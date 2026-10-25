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
from hamtext import PROMPT, resolve  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "sstv"))
from decoder import SSTVWatcher  # noqa: E402

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
HIST = 100               # waterfall lines for the channel detection (10 s)
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

    def transcribe(self, pcm, ham):
        """Whisper segments for 8 kHz int16 audio: on the GPU service, or on the CPU if it fails."""
        if self.remote:
            q = {"prompt": PROMPT} if ham else {}
            if args.lang:
                q["lang"] = args.lang
            try:
                req = urllib.request.Request(f"{args.remote}/transcribe?{urllib.parse.urlencode(q)}",
                                             data=pcm.astype("<i2").tobytes(), method="POST")
                with urllib.request.urlopen(req, timeout=60) as r:
                    d = json.load(r)
                self.label = f"{self.remote} on GPU"
                return [SimpleNamespace(**x) for x in d["segments"]], SimpleNamespace(language=d["language"])
            except Exception as e:
                print("GPU service error, this batch on the CPU:", e, flush=True)
                self.label = f"{args.model} on CPU (GPU service error)"
        if not self.load_local():
            return [], None
        audio = signal.resample_poly(pcm.astype(np.float32) / 32768, 2, 1)
        parts, info = self.model.transcribe(audio, language=args.lang, beam_size=1, vad_filter=True,
                                            condition_on_previous_text=False, initial_prompt=PROMPT if ham else None)
        return list(parts), info

    def submit(self, seg):
        if self.model is None and self.remote is None:
            return
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
                parts, info = self.transcribe(np.concatenate(pieces), ham)
            except Exception as e:
                parts, info = [], None
                print("transcribe error:", e, flush=True)
            keep = [[] for _ in batch]
            for s in parts:
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
        global sstv_chans
        sstv_chans = [SSTVChannel(f, m) for f, m in SSTV_FREQS if abs(f - center) < pl.span / 2 - 4000]
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
            pooled = db.reshape(pl.wf_bins, pl.pool).max(1)
            floor = float(np.percentile(pooled, 20))
            q = np.clip((pooled - floor + 10) * 4, 0, 255).astype(np.uint8)
            broadcast("wf", {"d": base64.b64encode(q.tobytes()).decode(), "floor": round(floor, 1)})

            # channel detection, auto mode and the list of channels for the transcription
            if now - last_detect >= 2 and len(hist) >= 30:
                last_detect = now
                found = detect(hist, view_freqs, float(np.median(hist)))
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
                broadcast("sstvstat", {"channels": [{"f": c.f, "mode": c.mode, "progress": c.prog}
                                                    for c in sstv_chans]})
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
        elif u.path == "/api/sstv":
            with lock:
                self.send(200, json.dumps(list(sstv_images)).encode(), "application/json")
        elif u.path == "/api/sstv/preview":
            # the image that is coming in now, as PNG
            ch = next((c for c in sstv_chans if c.prog and abs(c.f - float(qs.get("f", 0))) < 100), None)
            rgb = ch.watcher.preview() if ch else None
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


# SSTV calling frequencies; the receiver decodes those that are in the view
SSTV_FREQS = [(3.730e6, "lsb"), (3.735e6, "lsb"), (7.165e6, "lsb"), (7.171e6, "lsb"), (14.230e6, "usb"),
              (14.233e6, "usb"), (21.340e6, "usb"), (28.680e6, "usb")]
sstv_images = deque(maxlen=300)
sstv_q = queue.Queue(maxsize=200)


class SSTVChannel:
    def __init__(self, f, mode):
        self.f, self.mode = f, mode
        self.demod = Demod(mode)
        self.t = 0
        day = time.strftime("%Y%m%d")
        self.day = day
        self.watcher = SSTVWatcher(os.path.join(LOGS, "sstv", day), label=f"{f / 1e3:.0f}kHz")
        self.prog = None


def sstv_item(path, f, mode, ts, partial, lines, offset):
    day, name = path.split(os.sep)[-2:]
    return {"ts": ts * 1000, "f": f, "mode": mode, "partial": partial, "lines": lines,
            "offset_hz": offset, "url": f"/sstv/{day}/{name}"}


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
            ch.prog = ch.watcher.progress()
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
    for path in sorted(glob.glob(os.path.join(LOGS, "sstv", "*", "*.png")))[-300:]:
        # the decoder names the files sstv_<f>kHz_<UTC date>-<UTC time>_<mode>.png
        m = re.match(r"sstv_(\d+)kHz_(\d{8})-(\d{6})_(.+)\.png$", os.path.basename(path))
        if m:
            ts = calendar.timegm(time.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S"))
            sstv_images.append(sstv_item(path, float(m.group(1)) * 1000, m.group(4), ts, None, None, None))
    threading.Thread(target=sstv_worker, daemon=True).start()
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
