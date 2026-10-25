"""
Slot-aligned FT8 / FT4 decoder for 8 kHz mono SSB audio.

    w = FT8Watcher(label="14074kHz", modes=("FT8", "FT4"))
    for pcm, ts in audio:              # int16, 8000 Hz, any block length;
        for ev in w.feed(pcm, ts):     # ts = time.time() of the FIRST sample of pcm
            print(ev)                  # {"mode", "slot_ts", "audio_hz", "snr_db", "dt", "text", ...}

    python3 decoder.py recording.wav [FT8|FT4]     # offline decode of one slot (any rate)

The actual decoding is done by an external, proven decoder run as a separate
process on a 12 kHz WAV file of one UTC slot:

  * jt9 from WSJT-X (GPLv3, run as a separate program, not linked) - preferred,
    it is the reference decoder (multi-pass subtraction, OSD, a-priori CQ).
  * decode_ft8 from kgoba/ft8_lib (MIT) - fallback, about 3-4 dB less sensitive
    and its "SNR" is only a sync score.

Binary lookup: $FT8_JT9 / $FT8_DECODE_FT8, then <repo>/local/bin, then $PATH.

Timing: blocks are placed on a continuous sample counter.  A block whose ts
agrees with the counter (within 0.25 s) is appended contiguously, so jitter of
the time.time() stamps does not tear the audio; a smoothed clock error is
corrected in small steps; a larger disagreement (lost audio, restart) re-syncs
the counter to ts, leaving a gap of zeros in the slot.  The UTC slot of every
sample is computed from the counter, so gaps never shift the slot alignment.
A slot is decoded once audio past its end has arrived and at least
`min_coverage` of it is present.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.signal import resample_poly

FS = 8000
DEC_FS = 12000
MODES = {"FT8": 15.0, "FT4": 7.5}

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_LOCAL_BIN = os.path.join(_REPO, "local", "bin")


def _find(env, name):
    p = os.environ.get(env)
    if p and os.access(p, os.X_OK):
        return p
    p = os.path.join(_LOCAL_BIN, name)
    if os.access(p, os.X_OK):
        return p
    return shutil.which(name)


JT9 = _find("FT8_JT9", "jt9")
DECODE_FT8 = _find("FT8_DECODE_FT8", "decode_ft8")

# Decodes run in a shared pool so many watchers (bands) cannot oversubscribe the CPU.
_POOL = ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) // 2), thread_name_prefix="ft8dec")

# jt9:  "000000 -12  0.3 1234 ~  CQ DL1ABC JO62           a1"   (FT4 uses '+')
_JT9_RE = re.compile(r"^\s*\d{4,6}\s+(-?\d+)\s+(-?\d+\.\d)\s+(\d+)\s+[~+#:]\s+(.*?)\s*$")
# decode_ft8: "000000 +16.5 +1.36 1200 ~  CQ DL1ABC JO62"
_LIB_RE = re.compile(r"^\d{6}\s+([+-]?[\d.]+)\s+([+-]?[\d.]+)\s+(\d+)\s+~\s+(.*?)\s*$")
_FLAG_RE = re.compile(r"\s+(a\d+|\?)\s*$")


def _write_wav(path, x12):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(DEC_FS)
        w.writeframes(x12.astype("<i2").tobytes())


def _to_12k(pcm8, n_out):
    """8 kHz int16 -> 12 kHz int16 with a sane level, exactly n_out samples."""
    x = resample_poly(np.asarray(pcm8, np.float32), 3, 2)
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    if rms > 0:
        x *= min(3000.0 / rms, 1e3)          # jt9/ft8_lib don't care about level, int16 does
    x = np.clip(x, -32767, 32767)
    out = np.zeros(n_out, np.int16)
    n = min(n_out, x.size)
    out[:n] = x[:n]
    return out


def decode_slot(pcm8, mode="FT8", backend="auto", depth=3, ap=True, uncertain=False,
                tmpdir=None, timeout=None, stats=None):
    """Decode one slot of 8 kHz audio starting at the slot boundary.
    Returns a list of dicts {"audio_hz", "snr_db", "dt", "text", "ap", "uncertain", "decoder"}.
    ap: keep jt9 a-priori (AP) decodes ("ap" holds jt9's tag, e.g. "a1").
    uncertain: keep decodes jt9 marks with "?" (low confidence; off by default)."""
    period = MODES[mode]
    if backend == "auto":
        backend = "jt9" if JT9 else "ft8_lib"
    exe = JT9 if backend == "jt9" else DECODE_FT8
    if not exe:
        raise RuntimeError(f"FT8 decoder binary for backend {backend!r} not found")
    own = tmpdir is None
    d = tempfile.mkdtemp(prefix="ft8_", dir="/tmp") if own else tmpdir
    try:
        wav = os.path.join(d, "slot.wav")
        _write_wav(wav, _to_12k(pcm8, int(round(period * DEC_FS))))
        if backend == "jt9":
            cmd = [exe, "-5" if mode == "FT4" else "-8", "-p", "7" if mode == "FT4" else "15",
                   "-d", str(depth), "-L", "100", "-H", "3000", "-a", d, "-t", d, wav]
        else:
            cmd = [exe] + (["-ft4"] if mode == "FT4" else []) + [wav]
        p = subprocess.Popen(cmd, cwd=d, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, text=True)
        killer = threading.Timer(timeout or period * 2, p.kill)
        killer.start()
        try:
            stdout = p.stdout.read()
            _, status, ru = os.wait4(p.pid, 0)       # reap ourselves to get the child's CPU time
            p.returncode = os.waitstatus_to_exitcode(status)
        finally:
            killer.cancel()
            p.stdout.close()
        if stats is not None:
            stats["child_cpu_s"] = ru.ru_utime + ru.ru_stime
        out = []
        seen = set()
        for line in stdout.splitlines():
            m = (_JT9_RE if backend == "jt9" else _LIB_RE).match(line)
            if not m:
                continue
            snr, dt, f, text = float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4)
            flag, unsure = None, False
            while (fm := _FLAG_RE.search(text)):
                if fm.group(1) == "?":
                    unsure = True
                else:
                    flag = fm.group(1)
                text = text[:fm.start()].rstrip()
            if (flag and not ap) or (unsure and not uncertain):
                continue
            if backend != "jt9":
                dt -= 0.5 + 0.16 * (1 if mode == "FT8" else 0)   # ft8_lib reports raw offset
                if text.startswith("Error ["):
                    continue
            text = " ".join(text.split())
            if not text or text in seen:
                continue
            seen.add(text)
            out.append({"audio_hz": f, "snr_db": snr, "dt": round(dt, 2), "text": text,
                        "ap": flag, "uncertain": unsure, "decoder": backend})
        return out
    finally:
        if own:
            shutil.rmtree(d, ignore_errors=True)


class _Slots:
    """Per-mode slot buffers on the absolute 8 kHz sample grid (n = ts * FS)."""

    def __init__(self, mode):
        self.mode = mode
        self.L = int(round(MODES[mode] * FS))     # 120000 / 60000 samples
        self.bufs = {}                            # slot index -> [int16 array, filled count]

    def put(self, n0, x):
        i = 0
        while i < x.size:
            n = n0 + i
            k, off = divmod(n, self.L)
            take = min(x.size - i, self.L - off)
            b = self.bufs.get(k)
            if b is None:
                b = self.bufs[k] = [np.zeros(self.L, np.int16), 0]
            b[0][off:off + take] = x[i:i + take]
            b[1] += take
            i += take

    def complete(self, n_now):
        """Pop slots whose end lies before n_now."""
        done = []
        for k in sorted(self.bufs):
            if (k + 1) * self.L <= n_now:
                done.append((k, *self.bufs.pop(k)))
        return done


class FT8Watcher:
    def __init__(self, label="", modes=("FT8",), backend="auto", depth=3, ap=True, uncertain=False,
                 min_coverage=0.6, max_pending=4):
        self.label = label
        self.modes = tuple(m.upper() for m in modes)
        for m in self.modes:
            if m not in MODES:
                raise ValueError(f"unknown mode {m}")
        self.backend = backend
        self.depth = depth
        self.ap = ap
        self.uncertain = uncertain
        self.min_coverage = min_coverage
        self.max_pending = max_pending
        self._slots = {m: _Slots(m) for m in self.modes}
        self._n = None              # next expected absolute sample index
        self._err = 0.0             # smoothed clock error (samples)
        self._lock = threading.Lock()
        self._events = deque()
        self._futures = []
        self._tmp = tempfile.mkdtemp(prefix="ft8w_", dir="/tmp")
        self._job_lock = threading.Lock()    # one jt9 per watcher at a time (shared tmp dir)
        self.stats = {"slots": 0, "skipped": 0, "dropped": 0, "resyncs": 0, "cpu_s": 0.0, "wall_s": 0.0}

    # ---------------------------------------------------------------- feed
    def feed(self, pcm: np.ndarray, ts: float) -> list:
        x = np.asarray(pcm)
        if x.dtype != np.int16:
            x = np.clip(np.asarray(x, np.float64), -32768, 32767).astype(np.int16)
        x = x.reshape(-1)
        n_ts = int(round(ts * FS))
        if self._n is None:
            self._n = n_ts
        else:
            e = n_ts - self._n
            if abs(e) > 0.25 * FS:                  # lost audio / jump: re-sync
                self._n = n_ts
                self._err = 0.0
                self.stats["resyncs"] += 1
            else:
                self._err += 0.02 * (e - self._err)
                if abs(self._err) > 0.03 * FS:      # slow drift: step the counter
                    step = int(self._err)
                    self._n += step
                    self._err -= step
        if x.size:
            for s in self._slots.values():
                s.put(self._n, x)
            self._n += x.size
        for s in self._slots.values():
            for k, buf, filled in s.complete(self._n):
                self._submit(s.mode, k, buf, filled)
        return self._drain()

    def _submit(self, mode, k, buf, filled):
        slot_ts = k * MODES[mode]
        if filled < self.min_coverage * buf.size:
            self.stats["skipped"] += 1
            return
        with self._lock:
            self._futures = [f for f in self._futures if not f.done()]
            if len(self._futures) >= self.max_pending:
                self.stats["dropped"] += 1          # CPU can't keep up: drop this slot
                return
            self._futures.append(_POOL.submit(self._job, mode, slot_ts, buf))

    def _job(self, mode, slot_ts, buf):
        t0, c0 = time.time(), time.thread_time()
        st = {}
        try:
            with self._job_lock:
                res = decode_slot(buf, mode, self.backend, self.depth, self.ap, self.uncertain,
                                  tmpdir=self._tmp, stats=st)
        except Exception as e:                      # noqa: BLE001
            print(f"[ft8 {self.label}] decode error: {e}", file=sys.stderr)
            res = []
        cpu = st.get("child_cpu_s", 0.0) + time.thread_time() - c0   # decoder process + resampling
        with self._lock:
            self.stats["slots"] += 1
            self.stats["cpu_s"] += cpu
            self.stats["wall_s"] += time.time() - t0
            for d in res:
                d.update(mode=mode, slot_ts=slot_ts)
                self._events.append(d)

    def _drain(self):
        with self._lock:
            out = list(self._events)
            self._events.clear()
        return out

    # ---------------------------------------------------------------- misc
    def busy(self) -> bool:
        with self._lock:
            return any(not f.done() for f in self._futures)

    def poll(self) -> list:
        """Events of finished decodes without feeding audio."""
        return self._drain()

    def flush(self, timeout=60.0) -> list:
        """Wait for pending decodes (for tests / shutdown) and return their events."""
        end = time.time() + timeout
        while self.busy() and time.time() < end:
            time.sleep(0.02)
        return self._drain()

    def close(self):
        self.flush()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def __del__(self):
        try:
            shutil.rmtree(self._tmp, ignore_errors=True)
        except Exception:                           # noqa: BLE001
            pass


def _main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    mode = (argv[2] if len(argv) > 2 else "FT8").upper()
    with wave.open(argv[1]) as w:
        fs = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels())[:, 0]
    if fs != FS:
        from math import gcd
        g = gcd(FS, fs)
        x = resample_poly(x.astype(np.float32), FS // g, fs // g).astype(np.int16)
    print(f"backend: {'jt9 ' + JT9 if JT9 else 'ft8_lib ' + str(DECODE_FT8)}")
    for d in decode_slot(x, mode):
        print(f"{d['snr_db']:+4.0f} {d['dt']:+5.2f} {d['audio_hz']:5.0f}  {d['text']}"
              + (f"   [{d['ap']}]" if d["ap"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
