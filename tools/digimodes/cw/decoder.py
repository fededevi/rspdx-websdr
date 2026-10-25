"""
Streaming multi-signal CW (Morse) decoder for 8 kHz mono SSB audio.

    w = CWWatcher(label="14025kHz")
    for pcm, ts in audio:            # int16, 8000 Hz, blocks of 800 (any length works);
        for ev in w.feed(pcm, ts):   # ts = time.time() of the FIRST sample of pcm
            print(ev)                # {"mode":"CW","audio_hz","wpm","ts","text","snr_db","label"}
    w.active()                       # [{"audio_hz","wpm","snr_db","since"}]

    python3 decoder.py recording.wav            # offline decode of a WAV file

Pipeline
  1. Detection (every 0.5 s): short-time spectrum (25 ms Hann, 5 ms hop,
     31 Hz bins) over the last 3 s.  A bin is a CW candidate when its 90th
     percentile power is well above the local noise floor and above the
     local median of the 90th percentiles (AGC pumping, band edges), it is a
     local maximum, narrow (falls off on at least one side within ~100 Hz)
     and keyed (90th / 25th percentile contrast).  Steady carriers fail the
     contrast test, broadband voice fails the narrowness test.
  2. Tracking: up to MAX_TRACKERS tones.  Each tracker mixes its tone to
     baseband, low-pass filters it (4th order Butterworth, cut-off ~0.45 x WPM,
     close to a matched filter) and makes a 200 Hz envelope; a slow AFC
     follows drift.  A new tracker is back-filled from the last 3 s of audio.
  3. Keying: adaptive mark level (fast attack, limited to the recent envelope
     peak so it follows QSB) and noise level (key-up EMA, bounded by a low
     percentile of the last 3 s), threshold with hysteresis, de-glitching,
     interpolated edge times.  Weak short marks (noise blips) are folded
     into the surrounding space, short drop-outs into the mark.
  4. Timing: 2-means clustering of the recent mark lengths (log domain) gives
     dit and dah.  A tracker is "locked" (allowed to emit text) only when the
     statistics look like Morse: dah/dit 2.5..4.5, tight clusters, almost
     nothing at 2 units or > 5 units (RTTY), intra-element spaces ~1 unit,
     stable tone frequency from mark to mark (voice), mostly valid codes,
     speed 8..50 WPM.  At the first lock the last 15 s of audio are decoded
     again with the final parameters, so the start of the over is not lost.
  5. Decoding: a Viterbi search over a grid of speed scales (0.5x..2x of
     the reference speed, cheap changes after pauses) classifies marks and
     spaces, so run station and callers at different speeds on the same
     frequency are both read.  Words are emitted when followed by a word
     space and a few more elements, or after a pause.  Each word is checked
     (SNR, timing misfit, tone stability, garbage ratio) before it is emitted.

Unknown codes are emitted as '*'.  Prosigns: <AR> <SK> <KN> <AS> <HH> <SN> <CT>;
BT is emitted as '='.

snr_db is referred to a 3 kHz noise bandwidth (as in the test definition):
in-filter SNR minus 10*log10(3000 / filter noise bandwidth).
"""
from __future__ import annotations

import math
import sys
from collections import deque

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, sosfilt

FS = 8000
HOP = 40                     # samples per envelope / STFT step (5 ms)
ENV_DT = HOP / FS            # 0.005 s
WIN = 200                    # STFT window (25 ms)
NFFT = 256                   # 31.25 Hz bins
F_LO, F_HI = 280.0, 2850.0
K_LO = int(math.ceil(F_LO * NFFT / FS))
K_HI = int(math.floor(F_HI * NFFT / FS))
DET_FRAMES = 600             # 3 s of STFT frames
MED2MEAN = 1.19              # mean / median of the smoothed noise spectrum
BIN_ENBW = 1.5 * FS / WIN    # Hann window noise bandwidth (60 Hz)
DET_EVERY = 100              # run detection every 0.5 s
BACKFILL = 3 * FS            # audio used to back-fill a new tracker
REDECODE = 15 * FS           # audio kept to re-decode a tracker when it first locks

MAX_TRACKERS = 8
VISIBLE_S = 8.0              # emit only while the detector saw the tone recently
MIN_SEP_HZ = 55.0            # trackers closer than this are one signal

# detection thresholds (power ratios)
T_DET = 4.0                  # p90 above local noise floor (6 dB)
T_KEY = 3.0                  # p90 / p25 inside the bin (keyed, not a carrier)
T_DET90 = 2.0                # p90 above the local median of p90 (AGC pumping, band edges)
T_NARROW = 3.0               # p90 / neighbour p90 on at least one side

# keying
WORD_MISFIT = 0.2  # mean |log| timing misfit of a word
WORD_SNR = 4.0               # median mark power / noise power for a word
FC_START = 20.0              # initial envelope filter cut-off (Hz)
FC_K = 0.45   # cut-off = FC_K * WPM
FC_MIN, FC_MAX = 6.0, 25.0
ELEM_REL = 0.4               # a mark weaker than this x recent marks ...
ELEM_ABS = 6.0               # ... and than this x noise power is a noise blip
ELEM_ABS_MIN = 3.5          # a mark weaker than this x noise power is always a blip
AFC_GAIN = 0.1
DIT_MIN, DIT_MAX = 0.024, 0.15  # 50 .. 8 WPM
NOISE_CLAMP = 1.5
NOISE_N = 600                # envelope history for the noise floor (3 s)
HIST_N = 300                 # envelope history for the QSB peak limit (1.5 s)
FOFF_STD = 2.8               # max spread (Hz) of per-mark tone offsets
R_MIN = 2.0                  # minimum mark/noise amplitude ratio used for the threshold
THR_ON = 0.50
THR_OFF = 0.36

MORSE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F",
    "--.": "G", "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L",
    "--": "M", "-.": "N", "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R",
    "...": "S", "-": "T", "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
    "-.--": "Y", "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
    ".-.-.-": ".", "--..--": ",", "..--..": "?", "-..-.": "/", "-...-": "=",
    "-....-": "-", ".----.": "'", ".-..-.": '"', "-.--.-": ")", "---...": ":",
    "-.-.-.": ";", ".--.-.": "@", "-.-.--": "!", "..--.-": "_",
    ".-.-.": "<AR>", "...-.-": "<SK>", "-.--.": "<KN>", ".-...": "<AS>",
    "........": "<HH>", "...-.": "<SN>", "-.-.-": "<CT>",
}
UNKNOWN = "*"
DEBUG = None
SHORT_CODES = set("ETIANM")  # 1-2 element codes
SHORT_FRAC = 0.7
SCALES = np.exp(np.linspace(math.log(0.5), math.log(2.0), 21))   # speed scale grid
LSCALES = np.log(SCALES)
NSC = len(SCALES)
DLS = np.abs(LSCALES[:, None] - LSCALES[None, :])
LOG3 = math.log(3.0)
SW_IN = 3.0                  # cost per unit log-speed change inside a transmission
SW_PAUSE = 0.3               # ... after a pause (other station)
PRIOR_W = 1.0
LOOKAHEAD = 6                # elements decoded after a word before it is emitted
PAUSE_K = 2.0                # pause (x word space) that flushes the pending text


def _rstd(v):
    v = np.asarray(v, dtype=float)
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def _succ_std(v):
    """Robust spread from successive differences: insensitive to a few jumps
    (two stations taking turns a few Hz apart), large for random values."""
    d = np.diff(np.asarray(v, dtype=float))
    return float(1.4826 * np.median(np.abs(d)) / math.sqrt(2.0)) if len(d) else 0.0


def _fit_marks(marks):
    """2-means on log mark lengths.  Returns dict or None."""
    n = len(marks)
    if n < 6:
        return None
    lm = np.log(np.asarray(marks, dtype=float))
    lo, hi = np.percentile(lm, [15, 90])
    if hi - lo < math.log(1.7):
        return {"single": True, "center": float(np.exp(np.median(lm))),
                "spread": float(lm.std())}
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        a = lm[lm < mid]
        b = lm[lm >= mid]
        if len(a) == 0 or len(b) == 0:
            return {"single": True, "center": float(np.exp(np.median(lm))),
                    "spread": float(lm.std())}
        nlo, nhi = a.mean(), b.mean()
        if abs(nlo - lo) < 1e-4 and abs(nhi - hi) < 1e-4:
            break
        lo, hi = nlo, nhi
    mid = 0.5 * (lo + hi)
    a = lm[lm < mid]
    b = lm[lm >= mid]
    d = np.minimum(np.abs(lm - lo), np.abs(lm - hi))
    dit = (np.exp(lo) * len(a) + np.exp(hi) / 3.0 * len(b)) / n
    return {"single": False, "lo": float(np.exp(lo)), "hi": float(np.exp(hi)),
            "ratio": float(np.exp(hi - lo)), "na": len(a), "nb": len(b),
            "sa": float(a.std()) if len(a) > 1 else 0.0,
            "sb": float(b.std()) if len(b) > 1 else 0.0,
            "outliers": float(np.mean(d > 0.5)), "dit": float(dit),
            "split": float(np.exp(mid))}


class _Tracker:
    def __init__(self, freq, n_start):
        self.freq = float(freq)
        self.phase = 0.0
        self.fc = 0.0
        self.sos = None
        self.zi = None
        self._set_fc(FC_START)
        self.state = False
        self.last_tr = None        # time of last committed transition (s, watcher clock)
        self.cand = None           # pending transition time
        self.prev_x = 0.0
        self.acc_p = 0.0
        self.acc_n = 0
        self.acc_c = 0j
        self.acc_w = 0.0
        self.z_prev = 0j
        self.hist = np.zeros(0)
        self.mark_hist = deque(maxlen=256)   # (end, duration) for the SNR estimate
        self.phist = np.zeros(0)
        self.snr_spec = None       # SNR from the detection spectrum (3 kHz ref)
        self.marks_f = deque(maxlen=32)   # per-mark frequency offsets (Hz)
        self.ref_p = deque(maxlen=16)   # (time, power) of recent marks
        self.sig_ref = None
        self._seen_n = -1
        self.s_last = 1.0          # speed scale of the last decoded element
        self.nmarks = deque(maxlen=32)    # scale-normalised marks of decoded words
        self.nspaces = deque(maxlen=48)   # scale-normalised intra-word spaces
        self.word_scales = deque(maxlen=64)
        self.recent_chars = deque(maxlen=40)        # slow reference of the mark power of good words
        self.last_word_end = -1e9
        self.noise_a = None
        self.mark_a = None
        self.noise_p = None
        self.mark_p = None
        self.elems = deque()       # (is_mark, start, dur)
        self.marks = deque(maxlen=32)
        self.spaces = deque(maxlen=48)
        self.new_marks = 0
        self.dit = None
        self.unit = None           # (dit mark + intra space) / 2, for the speed
        self.split = None          # dit/dah boundary
        self.space_u = None
        self.locked = False
        self.ever_locked = False
        self.pass_count = 0
        self.fail_count = 0
        self.created = n_start / FS
        self.since = None
        self.last_mark_end = n_start / FS
        self.last_emit_end = -1.0
        self.last_seen = n_start / FS   # last time detector confirmed it

    # ---------------------------------------------------------------- dsp
    def _set_fc(self, fc):
        fc = float(fc)
        if self.sos is not None and abs(fc - self.fc) < 0.15 * self.fc:
            return
        old = self.fc
        self.fc = fc
        self.sos = butter(4, fc, fs=FS, output="sos")
        if self.zi is None:
            self.zi = np.zeros((self.sos.shape[0], 2), dtype=complex)
        if old and self.noise_a is not None:   # keep levels roughly consistent
            s = fc / old
            self.noise_a *= math.sqrt(s)
            self.noise_p *= s

    @property
    def enbw(self):
        return 2.05 * self.fc

    def envelope(self, x):
        """x: float audio, len multiple of HOP.  Returns (amp, pow, complex) per 5 ms."""
        n = len(x)
        w = 2 * math.pi * self.freq / FS
        ph = self.phase + w * np.arange(n)
        self.phase = float((self.phase + w * n) % (2 * math.pi))
        y, self.zi = sosfilt(self.sos, x * np.exp(-1j * ph), zi=self.zi)
        p = (y.real ** 2 + y.imag ** 2).reshape(-1, HOP).mean(axis=1)
        z = y.reshape(-1, HOP).mean(axis=1)
        return np.sqrt(p), p, z

    def init_levels(self, amp, pw):
        self.noise_a = float(np.percentile(amp, 30))
        self.mark_a = max(float(np.percentile(amp, 96)), R_MIN * self.noise_a)
        self.noise_p = float(np.percentile(pw, 30))
        self.mark_p = float(np.percentile(pw, 96))

    # ------------------------------------------------------------- keying
    def run(self, amp, pw, z, t0):
        """Run the keying state machine.  t0 = time of first envelope sample."""
        an = 1.0 - math.exp(-ENV_DT / 0.5)
        am = 1.0 - math.exp(-ENV_DT / 0.15)
        au = 1.0 - math.exp(-ENV_DT / 0.015)
        ad = 1.0 - math.exp(-ENV_DT / 2.5)
        g = 0.010 if self.dit is None else min(max(0.3 * self.dit, 0.008), 0.03)
        tail = 0.02 if self.dit is None else 0.6 * self.dit   # skip filter tails
        state = self.state
        cand = self.cand
        prev = self.prev_x
        acc_p, acc_n = self.acc_p, self.acc_n
        acc_c, acc_w, zp = self.acc_c, self.acc_w, self.z_prev
        na, ma, npw, mpw = self.noise_a, self.mark_a, self.noise_p, self.mark_p
        # QSB: the mark level may not exceed the envelope peak of the last ~1.5 s
        self.hist = np.concatenate([self.hist, amp])[-HIST_N:]
        # noise floor: low percentile of the recent envelope power (the EMA in the
        # key-up state is biased by the filter tails of strong signals)
        self.phist = np.concatenate([self.phist, pw])[-NOISE_N:]
        if len(self.phist) >= NOISE_N // 3:
            # the fast EMA follows AGC; the percentile bound stops it from
            # locking onto the signal itself (threshold above the dits)
            k = len(self.phist) // 5
            est = float(np.partition(self.phist, k)[k]) / 0.223
            if npw > NOISE_CLAMP * est:
                npw = NOISE_CLAMP * est
            ea = 0.886 * math.sqrt(est * NOISE_CLAMP)
            if na > ea:
                na = ea
        if len(self.hist) >= HIST_N // 2 and t0 - self.last_mark_end < 0.6:
            pk = float(self.hist.max())     # only while keying is going on
            if ma > pk:
                ma = pk
        zl = z.tolist()
        for i in range(len(amp)):
            x = float(amp[i])
            t = t0 + i * ENV_DT
            m = ma if ma > R_MIN * na else R_MIN * na
            thr = na + (THR_OFF if state else THR_ON) * (m - na)
            raw = x > thr
            if raw != state:
                if cand is None:
                    dx = x - prev
                    f = (thr - prev) / dx if dx != 0.0 else 1.0
                    f = 0.0 if f < 0.0 else (1.0 if f > 1.0 else f)
                    cand = t - ENV_DT + f * ENV_DT
                if t - cand >= g:
                    self._commit(state, cand, acc_p / acc_n if acc_n else 0.0,
                                 acc_c, acc_w)
                    state = not state
                    cand = None
                    acc_p, acc_n, acc_c, acc_w = 0.0, 0, 0j, 0.0
            else:
                cand = None
            zi = zl[i]
            if state:
                acc_p += float(pw[i])
                acc_n += 1
                if acc_n > 1:
                    q = zi * zp.conjugate()
                    acc_c += q
                    acc_w += abs(q)
            zp = zi
            if cand is None:
                if state:
                    ma += (au if x > ma else am) * (x - ma)
                    mpw += am * (float(pw[i]) - mpw)
                elif self.last_tr is None or t - self.last_tr > tail:
                    na += an * (x - na)
                    npw += an * (float(pw[i]) - npw)
                    if self.last_tr is not None and t - self.last_tr > 1.0:
                        floor = R_MIN * na
                        if ma > floor:
                            ma += ad * (floor - ma)
            prev = x
        self.state, self.cand, self.prev_x = state, cand, prev
        self.acc_p, self.acc_n = acc_p, acc_n
        self.acc_c, self.acc_w, self.z_prev = acc_c, acc_w, zp
        self.noise_a, self.mark_a, self.noise_p, self.mark_p = na, ma, npw, mpw

    def _commit(self, was_on, tc, mpow, acc_c=0j, acc_w=0.0):
        if self.last_tr is None:
            self.last_tr = tc
            return
        dur = tc - self.last_tr
        if was_on:
            foff = math.atan2(acc_c.imag, acc_c.real) / (2 * math.pi * ENV_DT) if acc_w else 0.0
            coh = abs(acc_c) / acc_w if acc_w else 0.0
            now = tc
            while self.ref_p and now - self.ref_p[0][0] > 8.0:
                self.ref_p.popleft()
            ref = float(np.median([v for _, v in self.ref_p])) if len(self.ref_p) >= 4 else None
            weak = mpow < ELEM_ABS_MIN * self.noise_p or (
                ref is not None and mpow < ELEM_REL * ref and mpow < ELEM_ABS * self.noise_p)
            if weak and (self.dit is None or dur < 2.0 * self.dit):
                # noise blip: fold it into the surrounding space
                if self.elems and not self.elems[-1][0]:
                    e = self.elems.pop()
                    if self.spaces and self.spaces[-1] == e[2]:
                        self.spaces.pop()
                    self.last_tr = e[1]
                return
            self.ref_p.append((now, mpow))
            self.elems.append((True, self.last_tr, dur, mpow, foff, coh))
            self.marks.append(dur)
            self.marks_f.append(foff)
            self.mark_hist.append((tc, dur))
            self.new_marks += 1
            self.last_mark_end = tc
            # AFC on clean marks
            if mpow > 6.0 * self.noise_p and coh > 0.8 and acc_w:
                self.freq += AFC_GAIN * max(-self.fc, min(self.fc, foff))
        elif self.elems:
            last = self.elems[-1]
            if self.dit is not None and last[0] and dur < 0.4 * self.dit:
                # drop-out inside a mark: join with the previous mark
                self.elems.pop()
                if self.marks and self.marks[-1] == last[2]:
                    self.marks.pop()
                    self.new_marks = max(self.new_marks - 1, 0)
                    if self.marks_f:
                        self.marks_f.pop()
                self.last_tr = last[1]
                return
            self.elems.append((False, self.last_tr, dur, 0.0, 0.0, 0.0))
            if dur < 2.0:
                self.spaces.append(dur)
        self.last_tr = tc

    # ------------------------------------------------------------- timing
    def evaluate(self):
        """Update dit estimate and the lock state from the recent statistics."""
        self.new_marks = 0
        use_norm = self.locked and len(self.nmarks) >= 10
        marks = self.nmarks if use_norm else self.marks
        spaces = self.nspaces if use_norm else self.spaces
        if not self.locked and self.ever_locked and len(self.elems) >= 16:
            # re-lock test on speed-normalised elements (stations at other speeds)
            el = list(self.elems)[-80:]
            path, _ = self._scale_path(el)
            sc = SCALES[path]
            marks = [e[2] / s for e, s in zip(el, sc) if e[0]][-32:]
            spaces = [e[2] / s for e, s in zip(el, sc) if not e[0] and e[2] < 2.0][-48:]
        f = _fit_marks(marks)
        if f is None:
            return
        if f["single"]:
            # all marks one length: only acceptable as continuation of a lock
            if self.dit is not None:
                c = f["center"]
                if not (0.7 * self.dit < c < 1.4 * self.dit
                        or 2.1 * self.dit < c < 4.2 * self.dit) or f["spread"] > 0.3:
                    self._fail()
            return
        if not self.locked and 2.0 <= f["ratio"] <= 5.0 and DIT_MIN <= f["dit"] <= DIT_MAX:
            self._set_fc(min(max(FC_K * 1.2 / f["dit"], FC_MIN), FC_MAX))
        strict = not self.locked
        rmin, rmax = (2.5, 4.5) if strict else (2.0, 5.0)
        smax = 0.26 if strict else 0.34
        omax = 0.12 if strict else 0.22
        dit = f["dit"]
        ok = (len(marks) >= (12 if strict else 8)
              and f["na"] >= 2 and f["nb"] >= 2
              and rmin <= f["ratio"] <= rmax
              and f["sa"] <= smax and f["sb"] <= smax
              and f["outliers"] <= omax
              and DIT_MIN <= dit <= DIT_MAX)
        if ok:
            # Morse marks are 1 or 3 units: few in between or longer (RTTY, voice)
            d = np.asarray(marks, dtype=float) / f["lo"]
            dz = np.mean((d > 1.5) & (d < min(2.3, 0.8 * f["ratio"])))
            lg = np.mean(d > 1.6 * f["ratio"])
            if dz > (0.10 if strict else 0.18) or lg > (0.05 if strict else 0.12):
                ok = False
            elif len(self.marks_f) >= 8 and _succ_std(self.marks_f) > (FOFF_STD if strict
                                                                    else 1.5 * FOFF_STD):
                ok = False
        if ok:
            sp = np.asarray(spaces, dtype=float)
            intra = sp[(sp > 0.35 * dit) & (sp < 1.9 * dit)]
            if len(sp) < (6 if not use_norm else 4) or len(intra) < (0.3 if strict else 0.2) * len(sp):
                ok = False
            else:
                u = float(np.median(intra))
                if not 0.5 * dit <= u <= 1.6 * dit:
                    ok = False
                else:
                    unit = 0.5 * (f["lo"] + u)
                    q = sp / unit
                    if unit < DIT_MIN or np.mean((q > 1.5) & (q < 2.2)) > (0.12 if strict
                                                                          else 0.2):
                        ok = False
        if ok:
            self.dit = dit
            self.unit = 0.5 * (f["lo"] + u)
            self.split = f["split"]
            self.space_u = u
            # code validity check on the buffered elements
            chars = list(self.recent_chars) if use_norm else self._chars(list(self.elems)[-120:])
            if len(chars) >= 3:
                bad = sum(1 for c in chars if c == UNKNOWN) / len(chars)
                if bad > (0.15 if strict else 0.3):
                    ok = False
            elif strict:
                ok = False
        if ok:
            self.fail_count = 0
            self.pass_count += 1
            if not self.locked and self.pass_count >= 2:
                self.locked = True
                self.ever_locked = True
            if not use_norm:
                wpm = 1.2 / self.unit
                self._set_fc(min(max(FC_K * wpm, FC_MIN), FC_MAX))
        else:
            self._fail()

    def _fail(self):
        self.pass_count = 0
        self.fail_count += 1
        if self.locked and self.fail_count >= 4:
            self.locked = False

    def _thresholds(self):
        d = self.dit
        u = self.space_u or d
        char_thr = math.sqrt(u * (2 * d + u))
        word_thr = math.sqrt((2 * d + u) * (6 * d + u))
        return self.split or 2 * d, char_thr, word_thr

    def _chars(self, elems):
        """Decode an element list into a list of chars (no word handling)."""
        if self.dit is None:
            return []
        split, char_thr, _ = self._thresholds()
        out, code = [], ""
        for e in elems:
            is_mark, dur = e[0], e[2]
            if is_mark:
                code += "." if dur < split else ("-" if dur < 6 * self.dit else "#")
            elif dur >= char_thr and code:
                out.append(MORSE.get(code, UNKNOWN))
                code = ""
        if code:
            out.append(MORSE.get(code, UNKNOWN))
        return out

    def _scale_path(self, elems):
        """Viterbi over a grid of speed scales (relative to the reference dit),
        so that stations at different speeds taking turns on one frequency are
        all decoded.  Returns (scale index per element, cost per element)."""
        d0 = self.dit
        u0 = self.space_u or d0
        c0 = 2 * d0 + u0
        wthr = math.sqrt(c0 * (6 * d0 + u0))
        n = len(elems)
        cost = np.empty((n, NSC))
        lam = np.full(n, SW_IN)
        for i, e in enumerate(elems):
            dur = e[2]
            if e[0]:
                l = math.log(dur / d0) - LSCALES
                c = np.minimum(np.abs(l), np.abs(l - LOG3))
            else:
                l = math.log(dur / u0) - LSCALES
                c = np.minimum(np.abs(l), np.abs(l - math.log(c0 / u0)))
                c[dur >= wthr * SCALES] = 0.0       # word gap / pause: no information
                if i + 1 < n and dur > 10 * d0:
                    lam[i + 1] = SW_PAUSE           # a new station may start here
            cost[i] = np.minimum(c, 0.8)
        acc = cost[0] + PRIOR_W * np.abs(LSCALES - math.log(self.s_last)) + 0.1 * np.abs(LSCALES)
        back = np.zeros((n, NSC), dtype=np.int16)
        for i in range(1, n):
            tr = acc[:, None] + lam[i] * DLS
            j = np.argmin(tr, axis=0)
            back[i] = j
            acc = tr[j, np.arange(NSC)] + cost[i]
        path = np.empty(n, dtype=np.int16)
        path[-1] = int(np.argmin(acc))
        for i in range(n - 1, 0, -1):
            path[i - 1] = back[i, path[i]]
        return path, cost[np.arange(n), path]

    def take_words(self, now, visible=True, force=False):
        """Decode complete words from the element buffer.
        Returns list of (start_time, word, wpm)."""
        # drop stale history (while unlocked or not visible)
        while self.elems and now - self.elems[0][1] > 15.0:
            self.elems.popleft()
        while self.elems and not self.elems[0][0]:
            self.elems.popleft()
        if not self.locked or self.dit is None or not self.elems:
            return []
        if not visible and not force:
            return []
        split, char_thr, word_thr = self._thresholds()
        n_el = len(self.elems)
        if not force and n_el == self._seen_n:
            # nothing new: only a pause can complete a word
            if (self.state or self.last_tr is None
                    or now - self.last_tr < PAUSE_K * word_thr * max(self.s_last, 0.5)):
                return []
        self._seen_n = n_el
        elems = list(self.elems)
        if not force and not (not self.state and self.last_tr is not None
                              and now - self.last_tr >= PAUSE_K * word_thr * 0.5):
            # a word can only be complete after a long space followed by LOOKAHEAD elements
            lim = 0.5 * word_thr
            if not any((not e[0]) and e[2] >= lim for e in elems[:len(elems) - LOOKAHEAD]):
                return []
        path, ecost = self._scale_path(elems)
        sc = SCALES[path]
        # pause: the current space is already longer than a word space
        pause = (not self.state and self.last_tr is not None
                 and now - self.last_tr >= PAUSE_K * word_thr * max(sc[-1], 1.0))
        cut = 0
        for i in range(len(elems) - 1 - LOOKAHEAD, -1, -1):
            if not elems[i][0] and elems[i][2] >= word_thr * sc[i]:
                cut = i + 1
                break
        if pause or force:
            cut = len(elems)
        if cut == 0:
            return []
        words = []
        cur = code = ""
        wd = None
        sr = split / self.dit
        for i in range(cut):
            is_mark, st, dur, mp, fo, _ = elems[i]
            s = sc[i]
            if wd is None:
                if not is_mark:
                    continue
                wd = {"st": st, "pws": [], "fos": [], "cost": [], "nm": [], "ns": [], "sc": []}
            if is_mark:
                dd = dur / (s * self.dit)
                code += "." if dd < sr else ("-" if dd < 6 else "#")
                wd["pws"].append(mp)
                wd["fos"].append(fo)
                wd["cost"].append(ecost[i])
                wd["nm"].append(dur / s)
                wd["sc"].append(s)
                wd["en"] = st + dur
            else:
                q = dur / s
                if q >= char_thr and code:
                    cur += MORSE.get(code, UNKNOWN)
                    code = ""
                if q >= word_thr:
                    if cur:
                        wd["text"] = cur
                        words.append(wd)
                    cur, wd = "", None
                else:
                    wd["cost"].append(ecost[i])
                    wd["ns"].append(q)
        if code:
            cur += MORSE.get(code, UNKNOWN)
        if cur and wd is not None:
            wd["text"] = cur
            words.append(wd)
        self.s_last = float(sc[cut - 1])
        for _ in range(cut):
            self.elems.popleft()
        while self.elems and not self.elems[0][0]:
            self.elems.popleft()
        self._seen_n = -1
        good = []
        for wd in words:
            w = wd["text"]
            self.recent_chars.extend(w.replace("<", "").replace(">", "")[:1] if w.startswith("<")
                                     else list(w))
            if self._word_ok(wd):
                self.nmarks.extend(wd["nm"])
                self.nspaces.extend(wd["ns"])
                s_w = float(np.median(wd["sc"]))
                self.word_scales.append((wd["st"], s_w))
                good.append((wd["st"], w.replace("#", UNKNOWN), 1.2 / (self.unit * s_w)))
                self.last_word_end = wd["en"]
        if good:
            self._recenter(now)
        return good

    def _recenter(self, now):
        """Keep the scale grid centred on the current speed and the envelope
        filter wide enough for the fastest station heard recently."""
        while self.word_scales and now - self.word_scales[0][0] > 30.0:
            self.word_scales.popleft()
        if not self.word_scales:
            return
        recent = [s for _, s in list(self.word_scales)[-4:]]
        sm = float(np.median(recent))
        if abs(math.log(sm)) > 0.12:
            self.dit *= sm
            self.unit *= sm
            self.split *= sm
            if self.space_u:
                self.space_u *= sm
            self.s_last /= sm
            self.nmarks = deque((v * sm for v in self.nmarks), maxlen=self.nmarks.maxlen)
            self.nspaces = deque((v * sm for v in self.nspaces), maxlen=self.nspaces.maxlen)
            self.word_scales = deque(((t, s / sm) for t, s in self.word_scales),
                                     maxlen=self.word_scales.maxlen)
        s_fast = min(s for _, s in self.word_scales)
        self._set_fc(min(max(FC_K * 1.2 / (self.unit * s_fast), FC_MIN), FC_MAX))

    def _word_ok(self, wd):
        w = wd["text"]
        pws = wd["pws"]
        mis = float(np.mean(wd["cost"])) if wd["cost"] else 0.0
        snr = float(np.median(pws)) / max(self.noise_p, 1e-30)
        if DEBUG is not None:
            DEBUG.append((w, mis, snr, len(pws)))
        rc = self.recent_chars
        if len(rc) >= 12 and sum(1 for c in rc if c in SHORT_CODES) > SHORT_FRAC * len(rc):
            return False            # only E/T/I/A/N/M lately: noise-like keying, not text
        nbad = w.count(UNKNOWN) + w.count("#")
        if nbad * 2 > len(w):                   # mostly garbage
            return False
        if snr < WORD_SNR or mis > WORD_MISFIT:     # too weak / timing does not fit
            return False
        mp = float(np.median(pws))
        if self.sig_ref is not None and mp < 0.12 * self.sig_ref:
            return False
        if len(wd["fos"]) >= 3 and _rstd(wd["fos"]) > 1.5 * FOFF_STD:
            return False                        # tone not stable (voice, noise)
        if len(pws) <= 1 and wd["st"] - self.last_word_end > 3.0:
            return False                        # isolated single blip
        self.sig_ref = mp if self.sig_ref is None else self.sig_ref + 0.3 * (mp - self.sig_ref)
        return True

    def snr_db(self):
        if self.snr_spec is not None:
            return self.snr_spec
        if self.noise_p is None or self.noise_p <= 0:
            return -99.0
        s = max(self.mark_p - self.noise_p, 1e-12 * self.noise_p) / self.noise_p
        return 10 * math.log10(s) - 10 * math.log10(3000.0 / self.enbw)

    @property
    def wpm(self):
        return 1.2 / self.unit if self.dit else 0.0


class CWWatcher:
    def __init__(self, label=""):
        self.label = label
        self._n = 0                         # samples processed (watcher clock)
        self._t0 = None                     # wall time of sample 0
        self._pend = np.zeros(0, dtype=np.float32)
        self._tail = np.zeros(WIN - HOP, dtype=np.float32)
        self._win = np.hanning(WIN + 2)[1:-1].astype(np.float32)
        self._spec = np.zeros((DET_FRAMES, K_HI - K_LO + 1), dtype=np.float32)
        self._spec_i = 0
        self._spec_fill = 0
        self._since_det = 0
        self._audio = deque()
        self._audio_len = 0
        self._trackers: list[_Tracker] = []

    # --------------------------------------------------------------- api
    def feed(self, pcm, ts):
        x = np.asarray(pcm, dtype=np.float32).ravel()
        if self._t0 is None:
            self._t0 = float(ts)
        else:
            exp = self._t0 + (self._n + len(self._pend)) / FS
            if abs(float(ts) - exp) > 1.0:   # lost audio / restart: re-sync clock
                self._t0 = float(ts) - (self._n + len(self._pend)) / FS
        if len(self._pend):
            x = np.concatenate([self._pend, x])
        n_use = (len(x) // HOP) * HOP
        self._pend = x[n_use:]
        x = x[:n_use]
        if n_use == 0:
            return []
        events = []
        n0 = self._n
        t_blk = n0 / FS
        # 1. spectrum frames for detection
        self._stft(x)
        # 2. running trackers
        for tr in self._trackers:
            amp, pw, z = tr.envelope(x)
            tr.run(amp, pw, z, t_blk + (HOP / 2) / FS)
        self._n += n_use
        now = self._n / FS
        self._audio.append(x)
        self._audio_len += n_use
        while self._audio_len - len(self._audio[0]) >= REDECODE:
            self._audio_len -= len(self._audio.popleft())
        # 3. detection
        self._since_det += n_use // HOP
        if self._since_det >= DET_EVERY and self._spec_fill >= DET_FRAMES // 2:
            self._since_det = 0
            self._detect(now)
        # 4. timing / text
        for tr in list(self._trackers):
            if tr.new_marks >= 3 or (tr.new_marks and not tr.state
                                     and now - tr.last_mark_end > 0.3):
                tr.evaluate()
                if tr.locked and tr.since is None:
                    tr.since = self._t0 + now
                    self._redecode(tr)
            vis = now - tr.last_seen < VISIBLE_S
            for st, w, wpm in tr.take_words(now, vis):
                events.append(self._event(tr, st, w, wpm))
            self._maybe_drop(tr, now)
        return events

    def active(self):
        out = []
        for tr in self._trackers:
            if tr.locked:
                out.append({"audio_hz": round(tr.freq, 1), "wpm": round(tr.wpm, 1),
                            "snr_db": round(tr.snr_db(), 1), "since": tr.since})
        return sorted(out, key=lambda d: d["audio_hz"])

    def flush(self):
        """Emit any text still buffered (e.g. at end of a recording)."""
        now = self._n / FS
        ev = []
        for tr in self._trackers:
            for st, w, wpm in tr.take_words(now, force=True):
                ev.append(self._event(tr, st, w, wpm))
        return ev

    # ---------------------------------------------------------- internals
    def _event(self, tr, st, w, wpm):
        return {"mode": "CW", "audio_hz": round(tr.freq, 1), "wpm": round(wpm, 1),
                "ts": round(self._t0 + (st if st is not None else self._n / FS), 3),
                "text": w, "snr_db": round(tr.snr_db(), 1), "label": self.label}

    def _stft(self, x):
        buf = np.concatenate([self._tail, x])
        self._tail = buf[-(WIN - HOP):]
        fr = np.lib.stride_tricks.sliding_window_view(buf, WIN)[::HOP]
        sp = np.fft.rfft(fr * self._win, NFFT, axis=1)[:, K_LO:K_HI + 1]
        p = (sp.real ** 2 + sp.imag ** 2).astype(np.float32)
        n = len(p)
        i = self._spec_i
        if i + n <= DET_FRAMES:
            self._spec[i:i + n] = p
        else:
            k = DET_FRAMES - i
            self._spec[i:] = p[:k]
            self._spec[:n - k] = p[k:]
        self._spec_i = (i + n) % DET_FRAMES
        self._spec_fill = min(self._spec_fill + n, DET_FRAMES)

    def _detect(self, now):
        if self._spec_fill < DET_FRAMES:
            S = self._spec[:self._spec_fill]
        else:
            S = np.roll(self._spec, -self._spec_i, axis=0)
        S = uniform_filter1d(S, 4, axis=0)
        p25, p50, p90 = np.percentile(S, [25, 50, 90], axis=0)
        floor = median_filter(p50, size=21, mode="nearest") + 1e-9
        floor90 = median_filter(p90, size=21, mode="nearest") + 1e-9
        nb = len(p90)
        cands = []
        for k in range(2, nb - 2):
            v = p90[k]
            if v < T_DET * floor[k] or v < T_KEY * p25[k] or v < T_DET90 * floor90[k]:
                continue
            if v < p90[k - 2:k + 3].max():
                continue
            left = p90[max(k - 5, 0):max(k - 2, 0)]
            right = p90[k + 3:k + 6]
            l = left.max() if len(left) else 0.0
            r = right.max() if len(right) else 0.0
            if min(l, r) * T_NARROW > v:
                continue
            a, b, c = np.log(p90[k - 1] + 1e-9), np.log(v + 1e-9), np.log(p90[k + 1] + 1e-9)
            den = a - 2 * b + c
            d = 0.5 * (a - c) / den if den < 0 else 0.0
            d = max(-0.5, min(0.5, d))
            cands.append(((K_LO + k + d) * FS / NFFT, float(v / floor[k])))
        cands.sort(key=lambda c: -c[1])
        # per-frame noise reference (robust to AGC pumping): median over the band
        nref = np.median(S, axis=1) * MED2MEAN + 1e-9 if self._trackers else None
        for t in self._trackers:
            on = sum(d for te, d in t.mark_hist if te > now - DET_FRAMES * ENV_DT)
            if on < 0.6:          # too little key-down time in the window
                continue
            k = int(round(t.freq * NFFT / FS)) - K_LO
            if not 0 <= k < nb:
                continue
            col = S[:, max(k - 1, 0):k + 2].max(axis=1) / nref
            frac = min(0.5 * on / (len(col) * ENV_DT), 0.25)
            top = np.sort(col)[-max(int(frac * len(col)), 4):]
            r = float(top.mean()) - 1.0
            v = 10 * math.log10(max(r, 1e-3)) - 10 * math.log10(3000.0 / BIN_ENBW)
            t.snr_spec = v if t.snr_spec is None else t.snr_spec + 0.3 * (v - t.snr_spec)
        for f, strength in cands:
            near = [t for t in self._trackers if abs(t.freq - f) < MIN_SEP_HZ]
            if near:
                t = min(near, key=lambda t: abs(t.freq - f))
                t.last_seen = now
                if abs(t.freq - f) > 3.0:
                    t.freq += 0.3 * (f - t.freq)
                continue
            if len(self._trackers) >= MAX_TRACKERS:
                victims = [t for t in self._trackers if not t.locked]
                if not victims:
                    continue
                v = min(victims, key=lambda t: t.last_seen)
                if now - v.last_seen < 1.0 and now - v.created < 5.0:
                    continue
                self._trackers.remove(v)
            self._new_tracker(f, now)
        # merge trackers that drifted together
        self._trackers.sort(key=lambda t: t.freq)
        i = 0
        while i < len(self._trackers) - 1:
            a, b = self._trackers[i], self._trackers[i + 1]
            if b.freq - a.freq < MIN_SEP_HZ * 0.7:
                drop = b if (a.locked, -a.created) >= (b.locked, -b.created) else a
                self._trackers.remove(drop)
            else:
                i += 1

    def _recent_audio(self, n_max):
        audio = np.concatenate(list(self._audio)) if self._audio else np.zeros(0, np.float32)
        audio = audio[-n_max:]
        return audio, self._n - len(audio)

    def _new_tracker(self, f, now):
        audio, n_start = self._recent_audio(BACKFILL)
        tr = _Tracker(f, n_start)
        if len(audio) >= 20 * HOP:
            amp, pw, z = tr.envelope(audio)
            tr.init_levels(amp, pw)
            tr.run(amp, pw, z, n_start / FS + (HOP / 2) / FS)
        else:
            tr.init_levels(np.ones(4), np.ones(4))
        tr.last_seen = now
        self._trackers.append(tr)

    def _redecode(self, tr):
        """Second pass at the first lock: re-run the keying on the last 15 s of
        audio with the final filter bandwidth, levels and dit length, so the
        start of the transmission is decoded as well as the rest."""
        audio, n_start = self._recent_audio(REDECODE)
        if len(audio) < FS:
            return
        sh = _Tracker(tr.freq, n_start)
        sh.dit = tr.dit
        sh._set_fc(tr.fc)
        amp, pw, z = sh.envelope(audio)
        # levels from robust statistics of the first second containing keying
        sh.init_levels(amp, pw)
        sh.run(amp, pw, z, n_start / FS + (HOP / 2) / FS)
        for k in ("phase", "sos", "zi", "fc", "state", "last_tr", "cand", "prev_x",
                  "acc_p", "acc_n", "acc_c", "acc_w", "z_prev", "hist", "phist", "noise_a",
                  "mark_a", "noise_p", "mark_p", "elems", "marks", "marks_f", "spaces",
                  "ref_p", "last_mark_end", "freq", "mark_hist"):
            setattr(tr, k, getattr(sh, k))
        tr.new_marks = 0

    def _maybe_drop(self, tr, now):
        idle = now - max(tr.last_mark_end, tr.last_tr or 0.0)
        if tr.locked:
            drop = idle > 60.0
        elif tr.ever_locked:
            drop = idle > 30.0
        else:
            drop = (now - tr.created > 20.0 and now - tr.last_seen > 5.0) or idle > 20.0
        if drop:
            self._trackers.remove(tr)


def _main(argv):
    import wave
    from scipy.signal import resample_poly
    if len(argv) < 2:
        print(__doc__)
        return 1
    with wave.open(argv[1]) as w:
        fs = w.getframerate()
        ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    x = x.reshape(-1, ch)[:, 0].astype(np.float64)
    if fs != FS:
        g = math.gcd(fs, FS)
        x = resample_poly(x, FS // g, fs // g)
    x = np.clip(x, -32768, 32767).astype(np.int16)
    w = CWWatcher(label=argv[1])
    for i in range(0, len(x), 800):
        for ev in w.feed(x[i:i + 800], i / FS):
            print(f"{ev['ts']:8.2f} {ev['audio_hz']:7.1f}Hz {ev['wpm']:4.1f}wpm "
                  f"{ev['snr_db']:+5.1f}dB  {ev['text']}")
    for ev in w.flush():
        print(f"{ev['ts']:8.2f} {ev['audio_hz']:7.1f}Hz {ev['wpm']:4.1f}wpm "
              f"{ev['snr_db']:+5.1f}dB  {ev['text']}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
