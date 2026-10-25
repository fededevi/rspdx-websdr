#!/usr/bin/env python3
"""Small web receiver for the RSPdx: live waterfall, active channels, audio and transcripts.

Runs rspdx-stream and sends to the browser:
  - a waterfall line every 0.1 s,
  - the audio of the tuned channel (USB, LSB, AM, CW or NFM) at 8 kHz,
  - a list of the channels with voice activity of the last 10 s,
  - transcripts of the voice on all active channels (Whisper, on the CPU).
In "auto" mode, the receiver follows the most active channel.

An overlap-save FFT channelizer (250 Hz bins) cuts each audio channel out of
the wide band and decimates it to 8 kHz in one step. All channels use the
same FFT, so many channels cost little.
"""
import argparse
import base64
import collections
import calendar
import faulthandler
import glob
import json
import os
import queue
import re
import signal as signal_mod
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import wave
from types import SimpleNamespace
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import sys

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from hamtext import prompt as ham_prompt, resolve  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "sstv"))
from decoder import SSTVWatcher  # noqa: E402
from hamtext import CALL as CALL_RE, check as check_call  # noqa: E402


def load_tool(*path):
    """Load a decoder module by its path: several decoders are named decoder.py."""
    import importlib.util
    full = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", *path)
    spec = importlib.util.spec_from_file_location("_".join(path).replace(".py", ""), full)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ft8mod = load_tool("digimodes", "ft8", "decoder.py")
pskmod = load_tool("digimodes", "psk_rtty", "decoder.py")
cwmod = load_tool("digimodes", "cw", "decoder.py")
faxmod = load_tool("wefax", "decoder.py")

p = argparse.ArgumentParser()
p.add_argument("--bind", default="127.0.0.1")
p.add_argument("--port", type=int, default=8490)
p.add_argument("--antenna", default=os.environ.get("ANT", "Antenna C"))
p.add_argument("--center", type=float, default=6.95e6, help="first center frequency in Hz")
p.add_argument("--rate", type=float, default=2e6, help="first sample rate in S/s")
p.add_argument("--model", default="small", help="Whisper model: tiny, base, small, medium, ...")
p.add_argument("--lang", default=None, help="language code for the transcripts, for example en or it (default: detect)")
p.add_argument("--max-channels", type=int, default=8,
               help="channels that are transcribed at the same time; with --remote it changes with the GPU load, up to 24")
p.add_argument("--no-transcribe", action="store_true")
p.add_argument("--no-save-audio", action="store_true", help="do not keep the WAV file of each voice segment")
p.add_argument("--no-prompt", action="store_true", help="do not give Whisper the amateur radio prompt")
p.add_argument("--remote", default=None,
               help="URL of a Whisper service on a GPU computer (tools/whisper_service.py), for example http://127.0.0.1:8491")
p.add_argument("--lna", type=int, default=5, help="LNA state, 0 = most RF gain (the loop overloads the receiver at 0)")
args = p.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOGS = os.path.join(ROOT, "logs")
STREAM = os.path.join(ROOT, "local", "bin", "rspdx-stream")
AR = 8000                # audio rate
M = 32                   # bins in an audio channel: 32 * 250 Hz = 8 kHz
BIN = AR / M
FRAMES = 50              # frames for each read: 50 * 2 ms = 0.1 s
HIST = 470               # waterfall lines: 10 s for the voice detection, 47 s for the FT8/FT4 slot timing
MAX_WF = 1600

# sample rate, IF filter and the part of the band that the filter passes
SPANS = [(192000, 200000, 0.9), (384000, 300000, 0.75), (768000, 600000, 0.75),
         (2000000, 1536000, 0.76), (6000000, 5000000, 0.8), (8000000, 7000000, 0.85)]


def view_bins(rate, use):
    """First bin, last bin and pooling of the shown part of the band."""
    n = M * (rate // AR)
    lo = round(n * (1 - use) / 2)
    pool = max(1, (n - 2 * lo) // MAX_WF)
    wf = (n - 2 * lo) // pool
    return lo, lo + wf * pool, pool


SPAN_WIDTH = {r: (lambda v: (v[1] - v[0]) * BIN)(view_bins(r, u)) for r, b, u in SPANS}


class Plan:
    """FFT sizes and filters for one sample rate."""

    def __init__(self, rate):
        self.rate, self.bw, self.use = next(s for s in SPANS if s[0] == rate)
        self.D = self.rate // AR
        self.N = M * self.D                     # 250 Hz bins
        self.hop = self.N // 2
        h = signal.firwin(self.N - self.hop + 1, 3300, fs=self.rate)
        self.h_sel = np.fft.fft(h, self.N)[np.arange(-M // 2, M // 2) % self.N]
        self.win = np.hanning(self.N).astype(np.float32)
        self.lo, self.hi, self.pool = view_bins(self.rate, self.use)
        self.wf_bins = (self.hi - self.lo) // self.pool
        self.span = SPAN_WIDTH[self.rate]


lock = threading.Lock()
clients = []
state = {"center": args.center, "rate": int(args.rate), "tune": 7.1e6, "mode": "lsb", "auto": False,
         "antenna": args.antenna, "status": "starting", "transcribe": not args.no_transcribe, "refine": 0,
         "lna": args.lna}
plan = None
channels = {}            # rounded carrier -> {"f", "mode", "score", "first", "last"}
sstv_chans = []
digi_chans = []
transcripts = deque(maxlen=20000)  # today's segments, for the threads and the timeline
restart = threading.Event()
streams = []             # the running rspdx-stream process, stopped when the server stops


def default_mode(f):
    if 7.2e6 <= f <= 7.45e6 or 9.4e6 <= f <= 9.9e6 or 11.6e6 <= f <= 12.1e6 \
            or 13.57e6 <= f <= 13.87e6 or 15.1e6 <= f <= 15.8e6 or f < 1.8e6:
        return "am"      # broadcast bands
    if 26.9e6 <= f <= 27.5e6 or f >= 29.5e6:
        return "nfm"
    return "lsb" if f < 10e6 else "usb"


# CW and digital parts of the amateur bands (IARU region 1), and FT8 on 40 m: no voice there
NO_VOICE = [(3.5e6, 3.6e6), (7.0e6, 7.06e6), (7.07e6, 7.08e6), (10.1e6, 10.15e6), (14.0e6, 14.101e6),
            (18.068e6, 18.111e6), (21.0e6, 21.151e6), (24.89e6, 24.931e6), (28.0e6, 28.32e6)]


def voice_ok(f):
    return not any(lo <= f <= hi for lo, hi in NO_VOICE)


def broadcast(kind, item):
    msg = f"event: {kind}\ndata: {json.dumps(item)}\n\n".encode()
    for q in list(clients):
        try:
            q.put_nowait(msg)
        except queue.Full:
            pass


def public_state():
    s = dict(state)
    rate, bw, use = next(x for x in SPANS if x[0] == state["rate"])
    lo, hi, pool = view_bins(rate, use)
    s.update({"span": SPAN_WIDTH[rate], "bw": bw, "audio_rate": AR, "wf_bins": (hi - lo) // pool,
              "spans": [{"rate": r, "span": SPAN_WIDTH[r]} for r, b, u in SPANS]})
    return s


class Demod:
    """Audio from the 8 kHz complex channel. Keeps the filter states between calls."""

    def __init__(self, mode):
        self.mode = mode
        t = np.arange(255)
        if mode in ("usb", "lsb", "cw"):
            lo, hi = {"usb": (200, 2900), "lsb": (-2900, -200), "cw": (450, 950)}[mode]
            self.b = signal.firwin(255, (hi - lo) / 2, fs=AR) * np.exp(2j * np.pi * (lo + hi) / 2 * t / AR)
        else:
            self.b = signal.firwin(255, 3800 if mode == "am" else 3000, fs=AR).astype(complex)
        self.zi = np.zeros(254, complex)
        self.dc = np.zeros(1)
        self.env = np.zeros(1)
        self.prev = 1 + 0j

    def __call__(self, z):
        """Return the audio as int16 and its mean power before the AGC."""
        y, self.zi = signal.lfilter(self.b, 1, z, zi=self.zi)
        if self.mode == "am":
            a = np.abs(y)
            m, self.dc = signal.lfilter([0.002], [1, -0.998], a, zi=self.dc)
            a = a - m
        elif self.mode == "nfm":
            y2 = np.concatenate(([self.prev], y))
            a = np.angle(y2[1:] * np.conj(y2[:-1]))
            self.prev = y[-1]
        else:
            a = y.real
        env, self.env = signal.lfilter([0.0005], [1, -0.9995], a * a, zi=self.env)
        g = 1 / np.maximum(np.sqrt(env), 1e-4 if self.mode != "nfm" else 0.3)
        out = np.clip(a * g * 0.25, -1, 1)
        return (out * 32767).astype("<i2"), float(np.mean(a * a))


def ssb_carrier(z, mode):
    """Offset of the SSB carrier from the tuned frequency, from the edge of the voice band.

    The voice of an SSB signal starts about 300 Hz from the suppressed carrier:
    above it for USB, below it for LSB. Returns None if no clear signal is found.
    """
    f, p = signal.welch(z, AR, nperseg=1024, return_onesided=False)
    i = np.argsort(f)
    f, p = f[i], np.convolve(10 * np.log10(p[i] + 1e-30), np.ones(7) / 7, mode="same")
    inside = np.abs(f) < 3200
    noise = np.percentile(p[inside], 15)
    idx = np.flatnonzero((p > noise + 5) & inside)
    if len(idx) < 20:
        return None
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 20) + 1)
    g = max(groups, key=lambda g: np.sum(10 ** (p[g] / 10)))
    if f[g[-1]] - f[g[0]] < 600:
        return None
    return f[g[0]] - 250 if mode == "usb" else f[g[-1]] + 250


class EdgeTracker:
    """The carrier of an SSB voice, from the average spectrum of its voiced blocks.

    The voice of a transmitter starts about E0 above its carrier (USB) or below it (LSB).
    Stations are usually on a whole kHz or on a 2.5 kHz grid, so the carrier goes to
    the nearest multiple of 0.5 kHz. A change needs the same result two times.
    """
    E0 = 200.0
    NFFT = 1024                      # 7.8 Hz bins at 8 kHz

    def __init__(self):
        self.reset()

    def reset(self):
        self.P = np.zeros(self.NFFT)
        self.voiced = self.blocks = 0
        self.nf = None
        self.last = None

    def add(self, z):
        """One block of the complex channel."""
        p = float(np.mean(np.abs(z) ** 2))
        self.nf = p if self.nf is None else (p if p < self.nf else self.nf * 1.005)
        self.blocks += 1
        if p > 4 * self.nf:
            self.P += np.abs(np.fft.fft(z * np.hanning(len(z)), self.NFFT)) ** 2
            self.voiced += 1

    def update(self, f, mode):
        """Every 10 s with enough voice: a new carrier for a channel at f, or None."""
        if self.blocks < 100 or self.voiced < 30:
            return None
        self.blocks = 0
        fq = np.fft.fftfreq(self.NFFT, 1 / AR)
        i = np.argsort(fq)
        fq, db = fq[i], np.convolve(10 * np.log10(self.P[i] + 1e-30), np.ones(5) / 5, mode="same")
        if mode == "lsb":
            fq, db = -fq[::-1], db[::-1]        # the audio frequencies are positive in both modes
        self.P *= 0.7                            # newer blocks count more
        inside = np.abs(fq) < 3200
        plateau = np.median(db[(fq > 600) & (fq < 1800)])
        floor = np.percentile(db[inside], 10)
        if plateau - floor < 6:
            return None
        half = (plateau + floor) / 2
        # the edge can only be between -350 and +600 Hz (a channel is at most about 500 Hz off);
        # if the level stays high below -350 Hz, another station is there and the result is not clear
        j = int(np.argmax(fq > 600))
        while db[j] > half and fq[j] > -350:
            j -= 1
        if fq[j] <= -350:
            return None
        edge = fq[j] + (half - db[j]) / (db[j + 1] - db[j] + 1e-9) * (fq[j + 1] - fq[j])
        delta = (edge - self.E0) * (1 if mode == "usb" else -1)
        new = round((f + delta) / 500) * 500
        if new != self.last:
            self.last = new
            return None
        return new if new != f else None


def snap_grid(f, tol_whole, tol_half):
    """Operators usually use a whole kHz, or a 2.5 kHz grid (so a half kHz): use one if it is near."""
    k = round(f / 1000) * 1000
    if abs(f - k) <= tol_whole:
        return k
    h = round(f / 500) * 500
    return h if abs(f - h) <= tol_half else round(f / 10) * 10


def detect(hist, freqs, floor):
    """Score the voice channels in the history: level that changes in time."""
    lin = 10 ** (hist / 10)
    comb = np.abs(((freqs + 100e3) % 200e3) - 100e3) < 1.5e3   # local 200 kHz interference
    lin[:, comb] = 10 ** (floor / 10)
    cs = np.concatenate((np.zeros((len(lin), 1)), np.cumsum(lin, axis=1)), axis=1)
    w = int(2400 / BIN)
    band = (cs[:, w:] - cs[:, :-w]) / w                        # band power that starts at each bin
    ts = 10 * np.log10(band / 10 ** (floor / 10))
    act = (ts > 4).mean(0)
    score = ts.std(0) * np.minimum(act, 0.9) * (act > 0.1)
    out = []
    for i in np.argsort(-score)[:300]:
        if score[i] < 1.0:
            break
        lo_f, hi_f = freqs[i], freqs[i] + w * BIN
        mode = default_mode((lo_f + hi_f) / 2)
        if mode == "am":
            continue
        f = lo_f - 300 if mode in ("usb", "nfm") else hi_f + 300
        f = round(f / 500) * 500   # the 0.5 kHz grid; the tracker of the channel corrects it later
        if all(abs(f - o["f"]) > 3500 for o in out):
            out.append({"f": f, "mode": "usb" if mode == "nfm" else mode, "score": round(float(score[i]), 2),
                        "active": round(float(act[i]), 2)})
        if len(out) >= 20:
            break
    return out


class Transcriber:
    """Whisper on the CPU. Voice segments wait in a queue, and the oldest are dropped when it is full."""

    # short texts that Whisper often invents from noise
    INVENTED = {"i love you", "thank you", "thanks for watching", "thank you for watching", "you", "bye", "so",
                "thank you very much", "grazie", "grazie a tutti", "sottotitoli a cura di qtss", "amen", "okay",
                "subtitles by the amaraorg community", "hmm", "oh", "uh", "ah"}

    # on noise, Whisper can repeat the alphabet of the prompt
    # also subtitle credits, which Whisper learned from videos and often invents from noise
    PROMPT_COPY = re.compile(r"bravo,? charlie,? delta|foxtrot,? golf,? hotel|uniform,? victor,? whiskey"
                             r"|germany,? italy,? japan|santiago,? sugar|kilowatt,? london"
                             r"|字幕|subtit|sottotitol|amara\.org")

    def __init__(self):
        self.q = queue.Queue(maxsize=40)
        self.status = "off" if args.no_transcribe else "loading model"
        self.done = self.dropped = 0
        self.capacity = args.max_channels   # channels to follow; changes with the queue when the GPU is used
        self.model = None
        self.remote = None       # model name of the GPU service, when it answers
        self.label = args.model
        self.local_lock = threading.Lock()
        if not args.no_transcribe:
            threading.Thread(target=self.load, daemon=True).start()

    def load(self):
        if args.remote:
            try:
                with urllib.request.urlopen(args.remote + "/health", timeout=5) as r:
                    h = json.load(r)
                self.remote = h["model"].replace("\\", "/").rstrip("/").split("/")[-1]   # a folder name, not the path
                self.label = f"{h['model']} on GPU"
                self.status = "ready"
                for _ in range(2):   # two requests at a time keep the GPU busy during the network transfer
                    threading.Thread(target=self.work, daemon=True).start()
                return
            except Exception as e:
                print("GPU service does not answer, using the CPU:", e, flush=True)
        if not self.load_local():
            return
        self.status = "ready"
        for _ in range(2):
            threading.Thread(target=self.work, daemon=True).start()

    def load_local(self):
        with self.local_lock:
            if self.model is None:
                try:
                    from faster_whisper import WhisperModel
                    self.model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=4,
                                              num_workers=2)
                except Exception as e:  # missing package or model download error
                    self.status = f"error: {e}"
                    return False
        return True

    def transcribe(self, pcm, pr):
        """Whisper segments for 8 kHz int16 audio with the prompt pr: on the GPU service, or on the CPU."""
        if self.remote:
            q = {"prompt": pr} if pr else {}
            if args.lang:
                q["lang"] = args.lang
            try:
                req = urllib.request.Request(f"{args.remote}/transcribe?{urllib.parse.urlencode(q)}",
                                             data=pcm.astype("<i2").tobytes(), method="POST")
                with urllib.request.urlopen(req, timeout=20) as r:
                    d = json.load(r)
                self.label = f"{self.remote} on GPU"
                return [SimpleNamespace(**x) for x in d["segments"]], SimpleNamespace(language=d["language"])
            except Exception as e:
                # the CPU cannot transcribe many channels and keep the receiver in real time: wait for the GPU
                print("GPU service error, no transcription for 60 s:", e, flush=True)
                self.label = f"{self.remote} on GPU: no answer, retry in 60 s"
                self.status = "waiting for the GPU service"
                self.paused_until = time.time() + 60
                return [], None
        if not self.load_local():
            return [], None
        audio = signal.resample_poly(pcm.astype(np.float32) / 32768, 2, 1)
        parts, info = self.model.transcribe(audio, language=args.lang, beam_size=1, vad_filter=True,
                                            condition_on_previous_text=False, initial_prompt=pr or None)
        return list(parts), info

    def submit(self, seg):
        if self.model is None and self.remote is None:
            return
        if time.time() < getattr(self, "paused_until", 0):
            return                      # the GPU service does not answer
        if self.status != "ready" and self.remote:
            self.status = "ready"
            self.label = f"{self.remote} on GPU"
        while True:
            try:
                self.q.put_nowait(seg)
                return
            except queue.Full:
                try:
                    self.q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def work(self):
        # Whisper always processes a 30 s window, so short segments go in one window together,
        # with silence between them. Each result goes to the segment at its time position.
        gap = np.zeros(int(1.5 * AR), np.int16)
        while True:
            batch = [self.q.get()]
            total = len(batch[0]["pcm"]) / AR
            while total < 20:
                try:
                    nxt = self.q.get_nowait()
                except queue.Empty:
                    break
                if total + 1.5 + len(nxt["pcm"]) / AR > 26:
                    self.q.put(nxt)
                    break
                batch.append(nxt)
                total += 1.5 + len(nxt["pcm"]) / AR
            pieces, spans, t = [], [], 0.0
            for seg in batch:
                if pieces:
                    pieces.append(gap)
                    t += len(gap) / AR
                pieces.append(seg["pcm"])
                spans.append((t, t + len(seg["pcm"]) / AR))
                t = spans[-1][1]
            try:
                # the prompt makes Whisper prefer the NATO alphabet, CQ, QRZ, five nine and so on
                ham = not args.no_prompt and any(seg["mode"] in ("usb", "lsb") for seg in batch)
                pr = None
                if ham:
                    # the validated callsigns of these frequencies in the last 15 minutes, most frequent first
                    with lock:
                        heard = collections.Counter(
                            c["call"] for t in transcripts for seg in batch
                            if abs(t["f"] - seg["f"]) < 600 and t["ts"] > seg["ts"] - 900000
                            for c in t.get("callinfo", []) if c.get("status") in ("known", "fixed"))
                    pr = ham_prompt([c for c, _ in heard.most_common(6)])
                parts, info = self.transcribe(np.concatenate(pieces), pr)
            except Exception as e:
                parts, info = [], None
                print("transcribe error:", e, flush=True)
            keep = [[] for _ in batch]
            for s in parts:
                # Whisper sometimes copies the heading of the callsign list in the prompt: remove only the heading
                s.text = re.sub(r"(?i)\s*stations on the frequency[.:,]?", " ", s.text).strip()
                if not s.text:
                    continue
                if not (s.no_speech_prob < 0.6 and s.avg_logprob > -1.0 and s.compression_ratio < 2.4) \
                        or re.sub(r"[^\w ]", "", s.text.lower()).strip() in self.INVENTED \
                        or self.PROMPT_COPY.search(s.text.lower()):
                    continue
                mid = (s.start + s.end) / 2
                i = min(range(len(spans)), key=lambda k: 0 if spans[k][0] <= mid <= spans[k][1]
                        else min(abs(mid - spans[k][0]), abs(mid - spans[k][1])))
                keep[i].append(s)
            for seg, k in zip(batch, keep):
                text = " ".join(s.text.strip() for s in k).strip()
                # callsigns heard on this frequency in the last 15 minutes help to correct a callsign
                with lock:
                    ctx = {c for t in transcripts if abs(t["f"] - seg["f"]) < 1500 and t["ts"] > seg["ts"] - 900000
                           for c in t.get("calls", [])}
                info_calls = resolve(text, ctx)
                conf = float(np.exp(np.mean([s.avg_logprob for s in k]))) if k else 0.0
                item = {"ts": seg["ts"], "f": seg["f"], "mode": seg["mode"], "dur": round(len(seg["pcm"]) / AR, 1),
                        "shift": seg.get("shift", 0),
                        "text": text, "lang": info.language if info and k else None, "conf": round(conf, 2),
                        "audio": seg.get("audio"), "calls": [c["call"] for c in info_calls], "callinfo": info_calls}
                self.done += 1
                with lock:
                    old = next((t for t in transcripts if seg.get("replace") and t.get("audio") == item["audio"]), None)
                    if old is not None:
                        old.clear()           # the same dict stays in the list, with the new content
                        old.update(item)
                        item["replace"] = True
                    else:
                        transcripts.append(item)
                # every segment goes in the log, also without clear speech, so each WAV file has a line
                with open(os.path.join(LOGS, time.strftime("transcripts_%Y%m%d.jsonl")), "a") as fh:
                    fh.write(json.dumps({k: v for k, v in item.items() if k != "replace"}) + "\n")
                broadcast("tr", item)


tx = Transcriber()


class Chan:
    """One audio channel for the transcription: demodulation, fine tuning and voice segments."""

    def __init__(self, f, mode, pinned=False):
        self.f, self.mode, self.pinned = f, mode, pinned
        self.demod = Demod(mode)
        self.t = 0
        self.edge = EdgeTracker()
        self.nf = None
        self.pre = deque(maxlen=3)
        self.seg = None
        self.seen = time.time()

    def set(self, f, mode):
        if mode != self.mode:
            self.demod = Demod(mode)
            self.mode = mode
        self.f = f

    def feed(self, z, pcm, power, now):
        # the carrier from the average spectrum of the voice (the pinned channel follows the tuned one)
        if self.mode in ("usb", "lsb") and not self.pinned:
            self.edge.add(z)
            new = self.edge.update(self.f, self.mode)
            if new is not None:
                self.f = new
                self.edge.reset()
        # voice segments: level more than 6 dB over the noise floor of the channel
        self.nf = power if self.nf is None else (power if power < self.nf else self.nf * 1.005)
        active = power > 4 * self.nf
        if self.seg is None:
            if active:
                self.seg = {"blocks": list(self.pre) + [pcm], "ts": now - 0.1 * (len(self.pre) + 1),
                            "last": now, "voiced": 1, "levels": [power]}
            else:
                self.pre.append(pcm)
            return
        self.seg["blocks"].append(pcm)
        self.seg["levels"].append(power)
        if active:
            self.seg["last"] = now
            self.seg["voiced"] += 1
        if now - self.seg["last"] > 1.2 or len(self.seg["blocks"]) >= 250:
            seg, self.seg = self.seg, None
            self.pre.clear()
            # speech changes its level from syllable to syllable; steady noise and carriers do not
            lv = 10 * np.log10(np.array(seg["levels"]) + 1e-20)
            if seg["voiced"] >= 8 and np.std(lv) > 3 and state["transcribe"]:
                self.finish(seg)

    def finish(self, seg):
        pcm = np.concatenate(seg["blocks"])
        item = {"ts": seg["ts"] * 1000, "f": self.f, "mode": self.mode, "pcm": pcm}
        if not args.no_save_audio:
            day = time.strftime("%Y%m%d", time.localtime(seg["ts"]))
            name = f"{time.strftime('%H%M%S', time.localtime(seg['ts']))}_{self.f / 1e3:.2f}kHz_{self.mode}.wav"
            d = os.path.join(LOGS, "segments", day)
            os.makedirs(d, exist_ok=True)
            with wave.open(os.path.join(d, name), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(AR)
                w.writeframes(pcm.tobytes())
            item["audio"] = f"/seg/{day}/{name}"
        tx.submit(item)


def extract(X, frame_no, f, center, t0, pl):
    """The 8 kHz complex channel at f from the FFT frames X."""
    k0 = int(round((f - center) / BIN))
    fres = f - center - k0 * BIN
    Y = X[:, (k0 + np.arange(-M // 2, M // 2)) % pl.N] * pl.h_sel
    y = np.fft.ifft(np.fft.ifftshift(Y, axes=1), axis=1)[:, M // 2:]
    # bin k0 turns by pi * k0 for each hop of N/2
    sign = np.where(((k0 * (frame_no + np.arange(len(X)))) % 2) == 1, -1, 1)
    z = (y * sign[:, None]).ravel()
    return z * np.exp(-2j * np.pi * fres * (t0 + np.arange(len(z))) / AR)


def receiver():
    global plan
    proc = None
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(ROOT, "local", "lib"))
    errlog = open(os.path.join(LOGS, "websdr_rx.log"), "a", buffering=1)
    while True:
        restart.clear()
        with lock:
            center, rate = state["center"], state["rate"]
        plan = pl = Plan(rate)
        if proc is not None and proc.poll() is None:
            # the stream changes its settings without closing the device; drop the old samples in the pipe
            proc.stdin.write(f"{int(center)} {rate} {pl.bw}\n".encode())
            end = time.time() + 0.3
            while time.time() < end:
                proc.stdout.read(65536)
        else:
            # the MW/FM and DAB notch filters stop strong broadcast stations; they do not touch HF
            settings = f"rfgain_sel={state['lna']},rfnotch_ctrl=true,dabnotch_ctrl=true"
            cmd = [STREAM, str(int(center)), str(rate), str(pl.bw), state["antenna"], settings]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errlog,
                                    env=env, bufsize=0)
            streams[:] = [proc]
        state["status"] = "running"
        broadcast("state", public_state())
        tail = np.zeros(pl.hop, np.complex64)
        frame_no = 0
        listen, listen_key, listen_t = None, None, 0
        ltrack = EdgeTracker()
        hist = None
        hist_t = []
        last_fine, narrow_k, last_dbg = 0, np.zeros(0, int), 0
        global digi_chans
        for ch in digi_chans:
            ch.closed = True
            ch.close()
        digi_chans = []
        global sstv_chans
        sstv_chans = []
        # the last seconds of I/Q, so that a decoder that starts on a found SSTV signal also gets its VIS header
        replay = deque(maxlen=max(10, min(40, int(64e6 / (FRAMES * pl.hop * 8)))))
        txch = []
        last_detect = time.time()
        nbytes = FRAMES * pl.hop * 4
        view_freqs = center + (np.arange(pl.lo, pl.hi) - pl.N // 2) * BIN
        while not restart.is_set():
            buf = bytearray()
            while len(buf) < nbytes:
                chunk = proc.stdout.read(nbytes - len(buf))
                if not chunk:
                    break
                buf += chunk
            if len(buf) < nbytes:
                state["status"] = "receiver stopped, retry in 2 s (is the RSPdx used by another program?)"
                broadcast("state", public_state())
                proc.kill()
                proc.wait()
                proc = None
                time.sleep(2)  # the SDRplay service needs a moment before the device can be opened again
                break
            iq = np.frombuffer(buf, np.int16).astype(np.float32).view(np.complex64) / 32768
            x = np.concatenate((tail, iq))
            tail = x[-pl.hop:]
            frames = np.lib.stride_tricks.sliding_window_view(x, pl.N)[::pl.hop][:FRAMES]
            X = np.fft.fft(frames, axis=1)
            now = time.time()
            replay.append((frame_no, x, now))

            # the channel that the user listens to
            with lock:
                tune, mode = state["tune"], state["mode"]
            if (tune, mode) != listen_key:
                if listen is None or mode != listen.mode:
                    listen = Demod(mode)
                listen_key = (tune, mode)
                ltrack.reset()
            z = extract(X, frame_no, tune, center, listen_t, pl)
            listen_t += len(z)
            pcm, _ = listen(z)
            broadcast("audio", base64.b64encode(pcm.tobytes()).decode())
            # the digital channels (FT8, FT4)
            for ch in digi_chans:
                zd = extract(X, frame_no, ch.f, center, ch.t, pl)
                ch.t += len(zd)
                pcm_d, _ = ch.demod(zd)
                try:
                    digi_q.put_nowait((ch, pcm_d, now - 0.1))
                except queue.Full:
                    pass

            # the SSTV channels
            for ch in sstv_chans:
                zs = extract(X, frame_no, ch.f, center, ch.t, pl)
                ch.t += len(zs)
                pcm_s, _ = ch.demod(zs)
                try:
                    sstv_q.put_nowait((ch, pcm_s, now - 0.1))
                except queue.Full:
                    pass

            # the tuned channel also follows the carrier of the voice, on the 0.5 kHz grid
            if mode in ("usb", "lsb"):
                ltrack.add(z)
                new = ltrack.update(tune, mode)
                if new is not None:
                    with lock:
                        if state["tune"] == tune:
                            state["tune"] = new
                    ltrack.reset()
                    broadcast("state", public_state())

            while not retunes.empty():
                F, delta = retunes.get()
                for c in txch:
                    if not c.pinned and abs(c.f - F) < 600:
                        c.f = c.f + delta
                        c.edge.reset()
                with lock:
                    for v in channels.values():
                        if abs(v["f"] - F) < 600:
                            v["f"] += delta

            # the channels for the transcription; the pinned one follows the tuned channel
            if state["transcribe"]:
                pin = next((c for c in txch if c.pinned), None)
                if mode in ("usb", "lsb", "am"):
                    if pin is None:
                        txch.append(Chan(tune, mode, pinned=True))
                    else:
                        pin.set(tune, mode)
                elif pin:
                    txch.remove(pin)
                for c in txch:
                    zc = extract(X, frame_no, c.f, center, c.t, pl)
                    c.t += len(zc)
                    pcm_c, power = c.demod(zc)
                    c.feed(zc, pcm_c, power, now)
            frame_no += FRAMES

            # waterfall line
            P = np.mean(np.abs(np.fft.fft(frames[::5] * pl.win, axis=1)) ** 2, axis=0)
            P = np.fft.fftshift(P)[pl.lo:pl.hi]
            db = 10 * np.log10(P + 1e-20).astype(np.float32)
            hist = db[None] if hist is None else np.vstack((hist[-(HIST - 1):], db[None]))
            hist_t = (hist_t + [now])[-HIST:]
            pooled = db.reshape(pl.wf_bins, pl.pool).max(1)
            floor = float(np.percentile(pooled, 20))
            q = np.clip((pooled - floor + 10) * 4, 0, 255).astype(np.uint8)
            broadcast("wf", {"d": base64.b64encode(q.tobytes()).decode(), "floor": round(floor, 1)})

            # channel detection, auto mode and the list of channels for the transcription
            if now - last_detect >= 2 and len(hist) >= 30:
                last_detect = now
                found = detect(hist[-100:], view_freqs, float(np.median(hist[-100:])))
                with lock:
                    for c in found:
                        # one entry for each channel: merge detections less than 2 kHz apart
                        key = next((k for k, v in channels.items() if abs(v["f"] - c["f"]) < 2000), int(c["f"]))
                        old = channels.get(key)
                        channels[key] = dict(c, first=old["first"] if old else now, last=now)
                    for key in [k for k, v in channels.items() if now - v["last"] > 600]:
                        del channels[key]
                    if state["auto"] and found:
                        best = found[0]
                        here = next((c for c in found if abs(c["f"] - state["tune"]) < 1500), None)
                        held = now - state.get("auto_since", 0)
                        if (here is None and held > 6) or (here and best["score"] > 2 * here["score"] and held > 15):
                            state.update(tune=best["f"], mode=best["mode"], auto_since=now, refine=2)
                            broadcast("state", public_state())
                    listing = sorted(channels.values(), key=lambda c: -c["last"])
                if state["transcribe"]:
                    for c in found:
                        near = next((t for t in txch if abs(t.f - c["f"]) < 2000), None)
                        if near:
                            near.seen = now
                        elif voice_ok(c["f"]) and sum(not t.pinned for t in txch) < tx.capacity:
                            txch.append(Chan(c["f"], c["mode"]))
                    txch = [t for t in txch if t.pinned or t.seg is not None or now - t.seen < 30]
                    # with spare GPU time, follow more channels; when the queue grows, follow fewer
                    if tx.remote:
                        if tx.q.qsize() <= 2 and sum(not t.pinned for t in txch) >= tx.capacity:
                            tx.capacity = min(24, tx.capacity + 1)
                        elif tx.q.qsize() > 8:
                            tx.capacity = max(4, tx.capacity - 1)
                else:
                    txch = []
                broadcast("chans", {"now": now, "list": listing, "current": found})
                # automatic SSTV channels on signals that look like SSTV; kept 15 minutes for the next picture
                # the last 2 s: a new picture is found quickly, and the replay gives the decoder its VIS header
                found_sstv, tiles = sstv_candidates(hist[-20:], view_freqs, float(np.median(hist[-100:])))
                # FT8 and FT4 from their timing: a decoder for each tile where the energy stops at every slot end
                slots = slot_tiles(hist, hist_t, view_freqs, float(np.median(hist[-100:])))
                if now - last_dbg >= 30:      # what the detectors see, for the log
                    last_dbg = now
                    kk = lambda m: [round((float(view_freqs[0]) + k * 2500) / 1e3) for k in np.flatnonzero(m)][:12]
                    print(f"detect: drops (mean, per slot) {slot_tiles.debug}", flush=True)
                    print(f"detect: history {hist_t[-1] - hist_t[0]:.0f} s; FT8 {kk(slots.get('FT8', []))}; "
                          f"FT4 {kk(slots.get('FT4', []))}; narrow (peaks) "
                          f"{[(round((float(view_freqs[0]) + k * 2500) / 1e3), int(narrow_k[k])) for k in np.argsort(-narrow_k)[:8]] if len(narrow_k) else []}",
                          flush=True)
                for mode, mask in slots.items():
                    for k in np.flatnonzero(mask):
                        tiles[k] = 4 if mode == "FT8" else 5
                        c = float(view_freqs[0]) + k * 2500 - 200        # the audio covers the tile plus 200 Hz
                        ch = next((d for d in digi_chans if d.mode == mode and abs(d.f - c) < 100), None)
                        if ch:
                            ch.seen = now
                        elif digi_ignore.get((mode, round(c)), 0) < now and sum(d.mode != "PSK" for d in digi_chans) < 8:
                            # FT8 or FT4 there: a PSK/RTTY decoder on the same tile is not necessary
                            for d in [d for d in digi_chans if d.mode == "PSK" and abs(d.f - c) < 100]:
                                d.closed = True
                                digi_chans.remove(d)
                            digi_chans.append(DigiChan(c, mode))
                            print(f"{mode}: new decoder for {(c + 200) / 1e3:.1f}-{(c + 2900) / 1e3:.1f} kHz", flush=True)
                # PSK31 and RTTY on narrow signals where no FT8 or FT4 is
                slot_k = {round((d.f + 200 - float(view_freqs[0])) / 2500) for d in digi_chans if d.mode != "PSK"}
                if now - last_fine >= 10:
                    last_fine = now
                    fine = fine_narrow_tiles(replay, center, pl, view_freqs)
                    if fine is not None:
                        narrow_k = fine
                for k in np.argsort(-narrow_k)[:int(np.count_nonzero(narrow_k))]:   # the busiest tiles first
                    if k in slot_k or tiles[k] in (4, 5):
                        continue
                    c = float(view_freqs[0]) + k * 2500 - 200
                    ch = next((d for d in digi_chans if d.mode == "PSK" and abs(d.f - c) < 100), None)
                    if ch:
                        ch.seen, ch.peaks = now, int(narrow_k[k])
                        continue
                    if digi_ignore.get(("PSK", round(c)), 0) >= now:
                        continue
                    psk = [d for d in digi_chans if d.mode == "PSK"]
                    if len(psk) >= 8:
                        # full: a busier tile replaces a decoder that has found nothing, on a quieter tile
                        idle = [d for d in psk if d.count == 0 and d.peaks < narrow_k[k]]
                        if not idle:
                            continue
                        old = min(idle, key=lambda d: d.peaks)
                        old.closed = True
                        digi_chans.remove(old)
                    if True:
                        digi_chans.append(DigiChan(c, "PSK"))
                        digi_chans[-1].peaks = int(narrow_k[k])
                        print(f"PSK/RTTY/CW: new decoders for {(c + 200) / 1e3:.1f}-{(c + 2900) / 1e3:.1f} kHz", flush=True)
                for ch in [d for d in digi_chans if now - d.seen > 120
                           or (now - d.created > (120 if d.mode != "PSK" else 300) and d.count == 0)]:
                    if ch.count == 0:
                        digi_ignore[(ch.mode, round(ch.f))] = now + 600   # no message in 2 minutes: not FT8 there
                    ch.closed = True
                    ch.close()
                    digi_chans.remove(ch)
                for c, m, kind in found_sstv:
                    if abs(c - center) > pl.span / 2 - 3000:      # the view edges are filtered
                        continue
                    # SSTV is one tone at a time; AM music and data fill the band at each moment. A weather map
                    # is mostly white, so it looks like a carrier: the fax decoder checks its line structure itself
                    if kind == "sstv" and max(sstv_tonal(X, c + d, m, center, pl) for d in (-500, 0, 500)) < 0.8:
                        k = int(((c - 1700 if m == "lsb" else c + 1700) - view_freqs[0]) // 2500)
                        if 0 <= k < len(tiles) and tiles[k] == 3:
                            tiles[k] = 2      # steady, but not one moving tone: not SSTV
                        continue
                    near = next((ch for ch in sstv_chans if ch.mode == m and abs(ch.f - c) < 1500), None)
                    if near:
                        near.seen = now
                    elif sum(ch.auto for ch in sstv_chans) < 24:
                        # three decoders: the grid carrier can be 500 Hz off, and a decoder takes about +-260 Hz
                        new = [SSTVChannel(c + d, m, auto=True) for d in (-500, 0, 500)]
                        Xr = [(fno, np.fft.fft(np.lib.stride_tricks.sliding_window_view(xr, pl.N)[::pl.hop][:FRAMES],
                                               axis=1), tr) for fno, xr, tr in list(replay)]   # with the current block
                        for ch in new:
                            for fno, Xb, tr in Xr:
                                zr = extract(Xb, fno, ch.f, center, ch.t, pl)
                                ch.t += len(zr)
                                pr, _ = ch.demod(zr)
                                try:
                                    sstv_q.put_nowait((ch, pr, tr - 0.1))
                                except queue.Full:
                                    pass
                        sstv_chans.extend(new)
                        print(f"SSTV/fax: new signal {c / 1e3:.1f} kHz {m}, decoders at -500/0/+500 Hz", flush=True)
                sstv_chans[:] = [ch for ch in sstv_chans if not ch.auto or ch.prog or now - ch.seen < 900]
                broadcast("sstvstat", {"channels": [{"f": c.f, "mode": c.mode, "progress": c.prog, "auto": c.auto}
                                                    for c in sstv_chans],
                                       "tiles": tiles, "tile_hz": 2500, "tile_f0": float(view_freqs[0]),
                                       "replay_s": round(len(replay) * FRAMES * pl.hop / pl.rate, 1),
                                       "digi": [{"f": d.f, "mode": d.mode} for d in digi_chans]})
                broadcast("tx", {"status": tx.status, "model": tx.label, "queue": tx.q.qsize(), "capacity": tx.capacity,
                                 "done": tx.done, "dropped": tx.dropped, "on": state["transcribe"],
                                 "channels": [{"f": t.f, "mode": t.mode, "talking": t.seg is not None,
                                               "pinned": t.pinned} for t in txch]})


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        qs = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as fh:
                self.send(200, fh.read(), "text/html; charset=utf-8")
        elif u.path == "/api/state":
            with lock:
                body = json.dumps({"state": public_state(), "transcripts": list(transcripts)[-300:],
                                   "chans": sorted(channels.values(), key=lambda c: -c["last"])})
            self.send(200, body.encode(), "application/json")
        elif u.path == "/api/thread":
            # all segments of one frequency, in time order
            try:
                f, tol = float(qs["f"]), float(qs.get("tol", 600))
            except (KeyError, ValueError):
                self.send(400, b"bad value", "text/plain")
                return
            with lock:
                items = sorted((t for t in transcripts if abs(t["f"] - f) <= tol), key=lambda t: t["ts"])
            self.send(200, json.dumps(items).encode(), "application/json")
        elif m := re.fullmatch(r"/sstv/(\d{8})/([\w.\-]+\.png)", u.path):
            path = os.path.join(LOGS, "sstv", m.group(1), m.group(2))
            if os.path.isfile(path):
                with open(path, "rb") as fh:
                    self.send(200, fh.read(), "image/png")
            else:
                self.send(404, b"not found", "text/plain")
        elif u.path == "/api/digi":
            with lock:
                self.send(200, json.dumps(list(digi_msgs)[-1000:]).encode(), "application/json")
        elif u.path == "/api/sstv":
            with lock:
                self.send(200, json.dumps(list(sstv_images)).encode(), "application/json")
        elif u.path == "/api/sstv/preview":
            # the image that is coming in now, as PNG
            ch = next((c for c in sstv_chans if c.prog and abs(c.f - float(qs.get("f", 0))) < 100), None)
            rgb = (ch.watcher.preview() if not ch.prog.get("fax") else ch.fax.preview()) if ch else None
            if rgb is not None and ch.prog.get("fax") and ch.mode == "lsb":
                rgb = 255 - np.asarray(rgb, np.uint8)     # the same inversion as for the saved picture
            if rgb is None:
                self.send(404, b"no image", "text/plain")
                return
            import io
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(np.asarray(rgb, np.uint8)).save(buf, "PNG")
            self.send(200, buf.getvalue(), "image/png")
        elif u.path == "/api/retune":
            try:
                F, delta = float(qs["f"]), float(qs["delta"])
                if not 0 < abs(delta) <= 3000:
                    raise ValueError
            except (KeyError, ValueError):
                self.send(400, b"bad value", "text/plain")
                return
            threading.Thread(target=retune, args=(F, delta), daemon=True).start()
            self.send(200, json.dumps({"ok": True}).encode(), "application/json")
        elif u.path == "/api/timeline":
            # short form of the segments of the last minutes, for the timeline
            try:
                since = time.time() * 1000 - float(qs.get("minutes", 30)) * 60000
            except ValueError:
                self.send(400, b"bad value", "text/plain")
                return
            with lock:
                items = [{"ts": t["ts"], "f": t["f"], "dur": t["dur"], "mode": t["mode"], "text": t["text"][:200],
                          "audio": t.get("audio"), "calls": t.get("calls", []), "callinfo": t.get("callinfo", [])}
                         for t in transcripts if t["ts"] >= since]
            self.send(200, json.dumps(items).encode(), "application/json")
        elif m := re.fullmatch(r"/seg/(\d{8})/([\w.\-]+\.wav)", u.path):
            path = os.path.join(LOGS, "segments", m.group(1), m.group(2))
            if os.path.isfile(path):
                with open(path, "rb") as fh:
                    self.send(200, fh.read(), "audio/wav")
            else:
                self.send(404, b"not found", "text/plain")
        elif u.path == "/api/set":
            with lock:
                try:
                    if "rate" in qs:
                        r = int(float(qs["rate"]))
                        if r not in SPAN_WIDTH:
                            raise ValueError
                        if r != state["rate"]:
                            state["rate"] = r
                            channels.clear()
                            restart.set()
                    if "center" in qs:
                        c = float(qs["center"])
                        if not 0.1e6 <= c <= 2000e6:
                            raise ValueError
                        if c != state["center"]:
                            state["center"] = c
                            channels.clear()
                            restart.set()
                    half = SPAN_WIDTH[state["rate"]] / 2
                    if not abs(state["tune"] - state["center"]) < half:
                        state["tune"] = state["center"] + half / 2
                        state["mode"] = default_mode(state["tune"])
                    if "tune" in qs:
                        f = float(qs["tune"])
                        if abs(f - state["center"]) < half:
                            mode = qs.get("mode") or default_mode(f)
                            if qs.get("snap") == "1" and mode in ("usb", "lsb"):
                                # a click is usually on the middle of the voice band; then the 0.5 kHz grid
                                f = round((f + (-1300 if mode == "usb" else 1300)) / 500) * 500
                            state["tune"] = f
                            state["mode"] = mode
                            state["auto_since"] = time.time()
                            state["refine"] = 0
                    if qs.get("mode") in ("usb", "lsb", "am", "cw", "nfm"):
                        state["mode"] = qs["mode"]
                    if "auto" in qs:
                        state["auto"] = qs["auto"] == "1"
                        state["auto_since"] = 0
                    if "lna" in qs:
                        n = int(qs["lna"])
                        if not 0 <= n <= 27:
                            raise ValueError
                        state["lna"] = n
                        for proc in streams:
                            if proc.poll() is None:
                                proc.stdin.write(f"set rfgain_sel={n}\n".encode())
                    if "transcribe" in qs and not args.no_transcribe:
                        state["transcribe"] = qs["transcribe"] == "1"
                except ValueError:
                    self.send(400, b"bad value", "text/plain")
                    return
                s = public_state()
            broadcast("state", s)
            self.send(200, json.dumps(s).encode(), "application/json")
        elif u.path == "/api/events":
            q = queue.Queue(maxsize=300)
            clients.append(q)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with lock:
                    first = f"event: state\ndata: {json.dumps(public_state())}\n\n".encode()
                self.wfile.write(first)
                while True:
                    try:
                        msg = q.get(timeout=15)
                    except queue.Empty:
                        msg = b": keepalive\n\n"
                    self.wfile.write(msg)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                clients.remove(q)
        else:
            self.send(404, b"not found", "text/plain")


sstv_images = deque(maxlen=300)
sstv_q = queue.Queue(maxsize=2000)


def sstv_candidates(hist, freqs, floor):
    """SSTV (about 1.5 kHz wide) and weather fax (about 1 kHz wide) candidates; see one_tone_candidates."""
    a, tiles = one_tone_candidates(hist, freqs, floor, 1500, 1200)
    b, tiles_b = one_tone_candidates(hist, freqs, floor, 1000, 1400)
    tiles = [max(x, y) for x, y in zip(tiles, tiles_b)]
    out = [(c, m, "sstv") for c, m in a]
    for c, m in b:
        if all(abs(c - o[0]) > 1500 or o[1] != m for o in out):
            out.append((c, m, "fax"))
    return out, tiles


def one_tone_candidates(hist, freqs, floor, width, start):
    """Signals that look like SSTV: a band of about 1.5 kHz that is on nearly all the time, at a steady level.

    Returns the carrier of each, for the sideband of its band. Voice changes its level all the time,
    so it is not found; a picture takes 36 to 120 s, so the last 10 s show it clearly.
    """
    lin = 10 ** (hist / 10)
    comb = np.abs(((freqs + 100e3) % 200e3) - 100e3) < 1.5e3
    lin[:, comb] = 10 ** (floor / 10)
    cs = np.concatenate((np.zeros((len(lin), 1)), np.cumsum(lin, axis=1)), axis=1)
    w = int(width / BIN)               # the band that the tones fill, in 250 Hz bins
    ts = 10 * np.log10((cs[:, w:] - cs[:, :-w]) / w / 10 ** (floor / 10))
    act, sd, lvl = (ts > 6).mean(0), ts.std(0), ts.mean(0)
    # the tones fill the whole band; one carrier would also be steady, but in one bin only
    bins = 10 * np.log10(lin.mean(0) / 10 ** (floor / 10))
    filled = np.lib.stride_tricks.sliding_window_view(bins, w).min(1)[:len(act)]
    ok = (act > 0.85) & (sd < 3.0) & (filled > 3)
    # what the detector found in each 2.5 kHz tile of the view, for the page:
    # 0 quiet, 1 varying signal (voice), 2 steady signal that is not SSTV, 3 SSTV candidate
    ntile = int(np.ceil((freqs[-1] - freqs[0]) / 2500)) + 1
    tiles = np.zeros(ntile, np.int8)
    ti = ((freqs[:len(act)] - freqs[0]) // 2500).astype(int)
    np.maximum.at(tiles, ti, np.where(lvl > 6, np.where(ok, 2, 1), 0).astype(np.int8))
    out = []
    for i in np.argsort(-lvl):
        if not ok[i]:
            continue
        lo_f, hi_f = freqs[i], freqs[i] + w * BIN
        # SSTV is about 1.2 kHz wide, voice about 2.4 kHz: the 500 Hz on both sides of the window
        # must be much weaker than the window, and for voice one side is still in the voice band
        g = int(250 / BIN)
        side = int(500 / BIN)
        if i - g - side < 0 or i + w + g + side > lin.shape[1]:
            continue
        inner = lin[:, i:i + w].mean()
        left = lin[:, i - g - side:i - g].mean()
        right = lin[:, i + w + g:i + w + g + side].mean()
        inner_db = 10 * np.log10(inner) - floor
        side_db = 10 * np.log10(max(left, right)) - floor
        if side_db > max(3.0, 0.35 * inner_db):
            continue
        tiles[ti[i]] = 3
        # the tones are 1100-2300 Hz from the carrier; SSTV and weather fax can be in either sideband
        for c, mode in ((round((lo_f - start) / 500) * 500, "usb"), (round((hi_f + start) / 500) * 500, "lsb")):
            if all(abs(c - o[0]) > 2500 or o[1] != mode for o in out):
                out.append((c, mode))
        if len(out) >= 12:
            break
    return out, tiles.tolist()


def slot_tiles(hist, hist_t, freqs, floor):
    """FT8 and FT4 from their timing: all stations send in the same UTC slots and stop before the
    end of each slot, so the energy of a tile drops at the same point of every slot."""
    t = np.asarray(hist_t)
    out = {}
    if len(t) < 2 or t[-1] - t[0] < 46:      # three full slots, so that a chance pattern does not pass
        return out
    tile = ((freqs - freqs[0]) // 2500).astype(int)
    starts = np.r_[0, np.flatnonzero(np.diff(tile)) + 1]
    P = np.add.reduceat(10 ** (hist / 10), starts, axis=1) / np.diff(np.r_[starts, len(tile)])
    drops = {}
    for mode, period, on_win, off_win in (("FT8", 15.0, (2.0, 12.0), (13.6, 14.9)), ("FT4", 7.5, (1.2, 4.8), (6.0, 7.3))):
        ph = t % period
        on = (ph >= on_win[0]) & (ph <= on_win[1])
        off = (ph >= off_win[0]) & (ph <= off_win[1])
        if on.sum() < 20 or off.sum() < 6:
            continue
        p_on, p_off = P[on].mean(0), P[off].mean(0)
        drop = 10 * np.log10(p_on / (p_off + 1e-30))
        level = 10 * np.log10(p_on) - floor
        found = (drop > 3) & (level > 3)
        slot_tiles.debug[mode] = sorted(((round(float(drop[k]), 1), k) for k in np.argsort(-drop)[:4]), reverse=True)
        per_slot = []
        # the drop must be in each slot, not only on average: a voice station with a regular rhythm fails
        for n in np.unique(np.floor(t / period))[1:-1]:
            sl = np.floor(t / period) == n
            o, f = sl & on, sl & off
            if o.sum() >= 10 and f.sum() >= 3:
                d = 10 * np.log10(P[o].mean(0) / (P[f].mean(0) + 1e-30))
                per_slot.append(d)
                found &= d > 3
        slot_tiles.debug[mode] = [(dr, [round(float(p[k]), 1) for p in per_slot]) for dr, k in slot_tiles.debug[mode]]
        out[mode] = found
        drops[mode] = drop
    # an FT8 or FT4 activity is at most about 3 kHz wide: a longer run of marked tiles is another signal
    for mode, m in out.items():
        edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a > 3:
                m[a:b] = False
    # FT4 also stops at every 15 s, and FT8 half fits the 7.5 s pattern: the larger drop tells which it is
    if "FT8" in out and "FT4" in out:
        ft8 = out["FT8"] & (drops["FT8"] >= drops["FT4"])
        out["FT4"] = out["FT4"] & (drops["FT4"] > drops["FT8"])
        out["FT8"] = ft8
    return out


def fine_narrow_tiles(replay, center, pl, freqs):
    """Tiles with narrow signals (CW, PSK, RTTY), from a fine spectrum (about 16 Hz) of the last second.

    A narrow signal is a peak more than 10 dB above the local level within +-1 kHz. The local level
    must be near the noise floor: AM broadcast and voice raise it with their sidebands."""
    from scipy.ndimage import uniform_filter1d
    blocks = list(replay)[-10:]
    if len(blocks) < 5:
        return None
    iq = np.concatenate([x[pl.hop:] for _, x, _ in blocks])
    n = 1 << int(np.ceil(np.log2(pl.rate / 16)))
    seg = iq[:len(iq) // n * n].reshape(-1, n)
    if len(seg) == 0:
        return None
    P = np.fft.fftshift(np.mean(np.abs(np.fft.fft(seg * np.hanning(n), axis=1)) ** 2, axis=0))
    f = center + (np.arange(n) - n // 2) * pl.rate / n
    inside = (f >= freqs[0]) & (f <= freqs[-1])
    f, db = f[inside], 10 * np.log10(P[inside] + 1e-30)
    floor = np.percentile(db, 30)
    local = uniform_filter1d(db, int(2000 / (pl.rate / n)))      # the level within +-1 kHz, in dB
    peak = (db - local > 10) & (local - floor < 6)
    tile = ((f - freqs[0]) // 2500).astype(int)
    out = np.zeros(int((freqs[-1] - freqs[0]) // 2500) + 1, int)
    np.add.at(out, tile[peak], 1)
    return out          # the number of narrow peaks in each tile


def narrow_tiles(hist, freqs, floor):
    """Tiles with narrow signals (PSK, RTTY, CW, carriers): a 250 Hz bin well above the noise,
    with the bins 500 Hz to each side much lower."""
    db = 10 * np.log10((10 ** (hist / 10)).mean(0)) - floor
    side = np.maximum(np.r_[db[2:], [db[-1]] * 2], np.r_[[db[0]] * 2, db[:-2]])
    # an AM broadcast carrier is narrow too, but its sidebands fill 1-2 kHz on both sides
    lin = 10 ** (db / 10)
    k = np.ones(5) / 5                                  # 1.25 kHz of bins
    near = 10 * np.log10(np.convolve(lin, k, mode="same"))
    left, right = np.r_[[near[0]] * 6, near[:-6]], np.r_[near[6:], [near[-1]] * 6]   # centered 1.5 kHz away
    peak = (db > 8) & (db - side > 6) & (np.minimum(left, right) < 4)
    tile = ((freqs - freqs[0]) // 2500).astype(int)
    out = np.zeros(tile.max() + 1, bool)
    out[tile[peak]] = True
    return out


slot_tiles.debug = {}


class DigiChan:
    """A channel of 2.7 kHz USB audio for a digital decoder: FT8, FT4, or PSK/RTTY."""

    def __init__(self, f, mode):
        self.f, self.mode = f, mode
        self.demod = Demod("usb")
        self.t = 0
        self.seen = self.created = time.time()
        self.count = 0
        self.peaks = 0
        self.closed = False
        if mode in ("FT8", "FT4"):
            self.watchers = [ft8mod.FT8Watcher(label=f"{f / 1e3:.0f}kHz", modes=(mode,))]
        else:
            # narrow signals: PSK31/63, RTTY and CW; each decoder gives text only if the timing fits its mode
            self.watchers = [pskmod.DigiWatcher(label=f"{f / 1e3:.0f}kHz"), cwmod.CWWatcher(label=f"{f / 1e3:.0f}kHz")]

    def close(self):
        for w in self.watchers:
            if hasattr(w, "close"):
                w.close()


digi_q = queue.Queue(maxsize=4000)


def digi_worker():
    """Feed the digital decoders in their own thread, so that the receiver loop stays in real time."""
    while True:
        cw_flush(time.time())
        ch, pcm, ts = digi_q.get()
        if ch.closed:
            continue
        for w in ch.watchers:
            try:
                for ev in w.feed(pcm, ts):
                    if ch.mode == "PSK":     # PSK, RTTY and CW give text chunks, not slots
                        ev = dict(ev, slot_ts=ev["ts"], snr_db=round(ev.get("snr_db", 0)))
                    digi_event(ch, ev)
            except Exception as e:  # one decoder error must not stop the others
                print(f"{ch.mode} decoder error:", e, flush=True)


digi_msgs = deque(maxlen=5000)
digi_seen = deque(maxlen=2000)       # (mode, slot, text) of the last messages, against duplicates of overlapping channels


digi_ignore = {}                     # (mode, frequency) -> time until which no decoder starts there


cw_lines = {}                        # (channel, tone) -> CW words that wait for a pause


def cw_flush(now, force=False):
    """Send the CW lines that had a pause of 4 s, or that are long enough."""
    for key in [k for k, v in cw_lines.items() if force or now - v["last"] > 4 or len(v["text"]) >= 80]:
        v = cw_lines.pop(key)
        digi_event(v["ch"], dict(v["ev"], text=v["text"], joined=True))


def digi_event(ch, ev):
    if ev.get("mode") == "CW" and not ev.get("joined"):
        # the CW decoder gives one word at a time: join the words of one signal into a line
        key = (id(ch), round(ev["audio_hz"] / 20))
        v = cw_lines.setdefault(key, {"ch": ch, "ev": ev, "text": "", "last": 0})
        v["text"] = (v["text"] + " " + ev["text"]).strip()
        v["last"] = time.time()
        cw_flush(v["last"])
        return
    ch.count += 1
    key = (ev["mode"], round(ev["slot_ts"]), ev["text"])
    if key in digi_seen:
        return
    digi_seen.append(key)
    calls = []
    for tok in re.findall(r"[A-Z0-9/]{3,12}", ev["text"]):
        base = max(tok.split("/"), key=len)
        if CALL_RE.match(base):
            # an FT8 message has a checksum, so its callsign is exact: only mark it, never correct it
            best, status, alts = check_call(base)
            if status:
                known = status == "known"
                calls.append({"call": base, "status": "known" if known else "new", "heard": base,
                              "alts": [] if known else ([best] if status == "fixed" else alts)})
    item = {"ts": ev["slot_ts"] * 1000, "f": ch.f + ev["audio_hz"], "mode": ev["mode"], "snr": ev.get("snr_db"),
            "text": ev["text"], "callinfo": calls, "calls": [c["call"] for c in calls]}
    with lock:
        digi_msgs.append(item)
    with open(os.path.join(LOGS, time.strftime("digital_%Y%m%d.jsonl")), "a") as fh:
        fh.write(json.dumps(item) + "\n")
    broadcast("digi", item)


def sstv_tonal(X, c, mode, center, pl):
    """How much SSTV is one tone at a time: in each short FFT frame, the share of the
    strongest pair of bins in the picture band. SSTV (FM) is near 1; AM music, data and
    noise spread their energy over the band and give about 2 / (number of bins)."""
    band = np.arange(1100, 2400, BIN)
    f = c + band if mode == "usb" else c - band
    k = np.round((f - center) / BIN).astype(int) % pl.N
    P = np.abs(X[:, k]) ** 2
    pair = P[:, 1:] + P[:, :-1]
    tonal = float(np.median(pair.max(1) / (P.sum(1) + 1e-30)))
    # the SSTV tone moves with the picture; a carrier stays in the same bin
    pos = pair.argmax(1)
    moving = float(np.mean(np.abs(pos - np.bincount(pos).argmax()) > 1))   # more than one bin from its usual place
    return tonal if moving > 0.08 else 0.0     # a sync pulse or a change of brightness in the block


class SSTVChannel:
    def __init__(self, f, mode, auto=False):
        self.f, self.mode, self.auto = f, mode, auto
        self.seen = time.time()
        self.demod = Demod(mode)
        self.t = 0
        day = time.strftime("%Y%m%d")
        self.day = day
        self.watcher = SSTVWatcher(os.path.join(LOGS, "sstv", day), label=f"{f / 1e3:.0f}kHz")
        # a weather fax is also one moving tone, so the same audio goes to a fax decoder
        self.fax = faxmod.WefaxWatcher(os.path.join(LOGS, "sstv", day), label=f"fax_{f / 1e3:.0f}kHz_{mode}")
        self.prog = None


def sstv_item(path, f, mode, ts, partial, lines, offset):
    day, name = path.split(os.sep)[-2:]
    return {"ts": ts * 1000, "f": f, "mode": mode, "partial": partial, "lines": lines,
            "offset_hz": offset, "url": f"/sstv/{day}/{name}"}


def invert_png(path):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        out = ImageOps.invert(im.convert("L"))
    out.save(path)


def sstv_worker():
    """Feed the SSTV decoders in their own thread: a finished image takes up to some 100 ms to render."""
    while True:
        ch, pcm, ts = sstv_q.get()
        try:
            for img in ch.watcher.feed(pcm, ts):
                item = sstv_item(img["path"], ch.f, img["mode"], img["ts"], img["partial"], img["lines"],
                                 img.get("offset_hz"))
                with lock:
                    sstv_images.append(item)
                broadcast("sstv", item)
            for img in ch.fax.feed(pcm, ts):
                if ch.mode == "lsb":
                    invert_png(img["path"])     # a fax is sent in USB: in LSB, black and white change places
                item = sstv_item(img["path"], ch.f, f"Weather fax {img['lpm']} lpm {ch.mode.upper()}", img["ts"],
                                 img["partial"], img["lines"], img.get("offset_hz"))
                with lock:
                    sstv_images.append(item)
                broadcast("sstv", item)
            p = ch.watcher.progress()
            if p is None and ch.fax.progress():
                fp = ch.fax.progress()
                p = {"mode": f"Weather fax {fp.get('lpm', '')} lpm", "lines": fp.get("lines", 0), "total_lines": "?",
                     "ts": fp.get("ts"), "fax": True}
            ch.prog = p
        except Exception as e:  # a decoder error must not stop the other channels
            print("SSTV error:", e, flush=True)


retunes = queue.Queue()   # (frequency, delta) for the receiver: move the live channels too


def retune(F, delta):
    """Move the carrier of the frequency F by delta Hz, for the live channels and for today's segments.

    The audio of each segment is shifted (the first original stays as .orig.wav),
    the segment gets the new frequency, and it is transcribed again.
    """
    retunes.put((F, delta))
    with lock:
        items = [t for t in transcripts if abs(t["f"] - F) < 600 and t.get("audio")]
    for t in items:
        day, name = t["audio"].split("/")[2:]
        path = os.path.join(LOGS, "segments", day, name)
        orig = path[:-4] + ".orig.wav"
        if not os.path.exists(path):
            continue
        if not os.path.exists(orig):
            os.replace(path, orig)
        with wave.open(orig) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64) / 32768
        total = t.get("shift", 0) + delta          # the carrier offset from the original tuning
        # USB: a carrier that is delta higher means a voice that was delta too high; LSB: too low
        s = -total if t["mode"] == "usb" else total
        y = np.real(signal.hilbert(x) * np.exp(2j * np.pi * s * np.arange(len(x)) / AR))
        pcm = np.clip(y * 32767, -32768, 32767).astype("<i2")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(AR)
            w.writeframes(pcm.tobytes())
        with lock:
            t["f"] = t["f"] + delta
            t["shift"] = total
        tx.submit({"ts": t["ts"], "f": t["f"], "mode": t["mode"], "pcm": pcm, "audio": t["audio"], "replace": True,
                   "shift": total})
    return len(items)


def backfill():
    """Transcribe the saved WAV files of today that have no line in the log, when the queue is almost empty."""
    time.sleep(30)
    day = time.strftime("%Y%m%d")
    with lock:
        logged = {t.get("audio") for t in transcripts}
    d = os.path.join(LOGS, "segments", day)
    names = sorted(n for n in os.listdir(d) if n.endswith(".wav")) if os.path.isdir(d) else []
    todo = [n for n in names if f"/seg/{day}/{n}" not in logged]
    print(f"backfill: {len(todo)} saved segments without a transcript", flush=True)
    for n in todo:
        m = re.fullmatch(r"(\d{6})_([\d.]+)kHz_(\w+)\.wav", n)
        if not m:
            continue
        while tx.q.qsize() > 1 or not state["transcribe"]:
            time.sleep(1)    # live segments first
        with wave.open(os.path.join(d, n)) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2")
        ts = time.mktime(time.strptime(day + m.group(1), "%Y%m%d%H%M%S")) * 1000
        tx.submit({"ts": ts, "f": float(m.group(2)) * 1000, "mode": m.group(3), "pcm": pcm,
                   "audio": f"/seg/{day}/{n}"})


def validate_old():
    """Find and validate the callsigns of the segments that were loaded from the log."""
    with lock:
        old = [t for t in transcripts if "callinfo" not in t]
    for t in old:
        info_calls = resolve(t.get("text", ""))
        t["callinfo"], t["calls"] = info_calls, [c["call"] for c in info_calls]


def main():
    faulthandler.register(10)  # SIGUSR1 prints the stack of each thread
    if state["rate"] not in SPAN_WIDTH:
        raise SystemExit(f"--rate must be one of {sorted(SPAN_WIDTH)}")
    span = SPAN_WIDTH[state["rate"]]
    state["tune"] = args.center + 176.5e3 if args.center == 6.95e6 and span > 400e3 else args.center + span / 4
    state["mode"] = default_mode(state["tune"])
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, "websdr.pid"), "w") as fh:   # for rspdx-websdr-stop.sh
        fh.write(str(os.getpid()))
    try:  # show the transcripts of today again after a restart; a later line of the same WAV file wins
        last = {}
        with open(os.path.join(LOGS, time.strftime("transcripts_%Y%m%d.jsonl"))) as fh:
            for line in fh:
                try:
                    t = json.loads(line)
                except ValueError:
                    continue
                last[t.get("audio") or id(t)] = t
        for t in sorted(last.values(), key=lambda t: t["ts"]):
            transcripts.append(t)
    except FileNotFoundError:
        pass
    for path in sorted(glob.glob(os.path.join(LOGS, "sstv", "*", "*.png")), key=os.path.getmtime)[-300:]:
        # the decoder names the files sstv_<f>kHz_<UTC date>-<UTC time>_<mode>.png
        m = re.match(r"sstv_(\d+)kHz_(\d{8})-(\d{6})_(.+)\.png$", os.path.basename(path))
        if m:
            ts = calendar.timegm(time.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S"))
            sstv_images.append(sstv_item(path, float(m.group(1)) * 1000, m.group(4), ts, None, None, None))
        # the fax decoder names them wefax_fax_<f>kHz_<sideband>_<UTC date>-<UTC time>_<lpm>_<ioc>.png
        m = re.match(r"wefax_fax_(\d+)kHz_(usb|lsb)_(\d{8})-(\d{6})_(\d+)_(\d+)\.png$", os.path.basename(path))
        if m:
            ts = calendar.timegm(time.strptime(m.group(3) + m.group(4), "%Y%m%d%H%M%S"))
            sstv_images.append(sstv_item(path, float(m.group(1)) * 1000, f"Weather fax {m.group(5)} lpm {m.group(2).upper()}",
                                         ts, None, None, None))
    try:  # the digital messages of today
        with open(os.path.join(LOGS, time.strftime("digital_%Y%m%d.jsonl"))) as fh:
            for line in fh:
                try:
                    digi_msgs.append(json.loads(line))
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    threading.Thread(target=sstv_worker, daemon=True).start()
    threading.Thread(target=digi_worker, daemon=True).start()
    threading.Thread(target=validate_old, daemon=True).start()
    if args.remote:   # the CPU alone has no spare time for old segments
        threading.Thread(target=backfill, daemon=True).start()
    threading.Thread(target=receiver, daemon=True).start()
    srv = ThreadingHTTPServer((args.bind, args.port), Handler)
    srv.daemon_threads = True
    print(f"Web receiver: http://{args.bind}:{args.port}/  ({args.antenna})", flush=True)
    def on_term(*_):
        raise KeyboardInterrupt  # stop like Ctrl+C, so the stream also stops

    signal_mod.signal(signal_mod.SIGTERM, on_term)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for proc in streams:
            proc.terminate()


if __name__ == "__main__":
    main()
