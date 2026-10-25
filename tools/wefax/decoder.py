"""
Streaming HF weather-fax (WEFAX / radiofax) decoder for 8 kHz mono USB audio.
numpy + scipy only (PIL used for PNG output when available).

    w = WefaxWatcher("/path/to/out", label="7880")
    for block, ts in audio:              # int16, 8000 Hz, any block length
        for img in w.feed(block, ts):    # finished (or partial) images
            print(img)   # {"path","ts","lpm","ioc","lines","offset_hz","partial",...}
        p = w.progress()                 # None or {"state","lpm","ioc","lines","ts","offset_hz"}
        a = w.preview()                  # None or uint8 grey (lines x width) array

    python3 decoder.py recording.wav [out_dir]     # offline decode of a file

Signal format (WMO / ITU-R, as used by DWD, NOAA, JMH, ...), USB dial 1.9 kHz
below the assigned frequency:  black 1500 Hz, white 2300 Hz.
  start   : 5 s alternating black/white at 300 Hz (IOC 576) or 675 Hz (IOC 288)
  phasing : ~30 s of lines = 5 % white pulse at the line start, rest black
            (the decoder accepts either polarity)
  image   : 60/90/120/240 lpm, width = pi * IOC pixels (1809 / 904)
  stop    : 5 s alternating black/white at 450 Hz, then ~10 s black

Signal path:
  * x is mixed by -1900 Hz and low-passed (+-1050 Hz, 65-tap FIR) -> y[n].
    p[n] = y[n] conj(y[n-1]); angle(sum p) over an interval is a power weighted
    instantaneous-frequency estimate (FM clicks have little weight).
  * d4 = frequency from p summed over 4 samples (2 kHz rate) is used for all
    detection work.  Every 0.1 s frame: Hann-windowed DFT of the last 0.2 s
    of d4 at 300/450/675 Hz; a "hit" needs >40 % of the AC power in that
    line and a swing consistent with +-400 Hz deviation.  Mean of d4 during
    the start tone = tuning offset.  An in-band / out-of-band audio power ratio
    per frame is the "signal present" indicator.
  * Phasing: d4 is folded at the period of every lpm candidate; the shortest
    period giving a strong 5 %-pulse profile gives lpm, polarity and line
    start.  Then every phasing line's pulse is located (box matched filter,
    parabolic sub-sample peak); a robust linear fit gives the true line period
    (slant correction) and phase.  Black/white levels measured on the phasing
    lines refine the offset and contrast.  The image starts at the first of 3
    consecutive non-phasing lines.
  * Image lines are rendered as soon as their samples are available: pixel =
    angle of the sum of p over a window around the pixel centre (cumsum +
    interpolation, so fractional line periods are exact).
  * End: stop tone (>= 0.8 s of 450 Hz hits), a new start tone, signal loss
    (< 10 % "signal" frames in 15 s, image trimmed to the last good frame),
    or a length cap.  Lost / capped images are saved with partial=True.
  * Running fax without start tone (optional, on by default): every 2 s the
    last 24 s of d4 are tested for line-periodic structure (normalised
    autocorrelation at the 120 lpm / 60 lpm line lag much higher than at
    non-harmonic reference lags, plus a good signal indicator).  If the folded
    data look like phasing, the normal phasing path is used; otherwise the
    image is decoded from the start of that 24 s window with the period from
    the autocorrelation peak (refined at 8 kHz), IOC 576 assumed, and the
    slant is refined again on the finished image from the line-to-line
    correlation.  The left edge is unknown in that mode (the picture can be
    rotated horizontally).
"""
from __future__ import annotations

import math
import os
import struct
import time
import zlib

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import firwin, lfilter

try:  # optional
    from PIL import Image as _PILImage
except Exception:  # pragma: no cover
    _PILImage = None

FS = 8000
FRAME = 800                 # analysis frame (0.1 s)
DEC = 4                     # d4 decimation
FS4 = FS // DEC             # 2000 Hz
F_CENTER = 1900.0
DEV = 400.0                 # black = -DEV, white = +DEV (relative to centre)
LPMS = (240, 120, 90, 60)   # shortest period first
TONES = (300.0, 450.0, 675.0)

_LP = firwin(65, 1050.0, fs=FS)
_MIX = np.exp(-2j * np.pi * F_CENTER / FS * np.arange(80))   # 1900/8000 = 19/80
_TWIN = 2 * FRAME // DEC                                    # 400 d4 samples
_HANN = np.hanning(_TWIN)
_TE = np.array([_HANN * np.exp(-2j * np.pi * f / FS4 * np.arange(_TWIN)) for f in TONES])
_TNORM = 2.0 / _HANN.sum()
_HZ = FS / (2 * np.pi)
_HZ4 = FS / (2 * np.pi)     # angle of p (lag 1 at 8 kHz) -> Hz, same for sums
# white input noise of variance s2 adds s2 * _R1 (real) to E[p]: it pulls every
# frequency estimate toward 1900 Hz.  s2 is estimated from the 300..1000 Hz
# audio band and subtracted (bias-free offset, levels and pixels at low SNR).
_R1 = float(np.sum(_LP[1:] * _LP[:-1]))
_W800 = np.hanning(FRAME)
_W2 = float(np.sum(_W800 ** 2))


def write_png(path, img):
    img = np.ascontiguousarray(img, dtype=np.uint8)
    if _PILImage is not None:
        _PILImage.fromarray(img).save(path)
        return
    h, w = img.shape[:2]
    ct = 0 if img.ndim == 2 else 2
    raw = b"".join(b"\x00" + img[i].tobytes() for i in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ct, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _box_peak(v, w):
    """Index (float) of the best placement of a width-w box on v (sub-sample)."""
    cs = np.concatenate(([0.0], np.cumsum(v)))
    m = cs[w:] - cs[:-w]
    k = int(np.argmax(m))
    h = max(1, w // 4)
    if h <= k < len(m) - h:
        a, b, c = m[k - h], m[k], m[k + h]
        den = a - 2 * b + c
        frac = 0.5 * (a - c) / den * h if den < 0 else 0.0
        frac = max(-h, min(h, frac))
    else:
        frac = 0.0
    return k + frac, m[k] / w


def fold_score(lum, P):
    """Fold lum (+-1 units) at period P (float). -> (score, polarity, edge) or None.
    score ~ pulse contrast (1 = full black/white swing) of a 5 % wide pulse."""
    m = int(len(lum) // P)
    if m < 4:
        return None
    Pi = int(P)
    starts = np.round(np.arange(m) * P).astype(np.int64)
    prof = lum[starts[:, None] + np.arange(Pi)[None, :]].mean(axis=0)
    w = max(2, int(round(0.05 * P)))
    ext = np.concatenate(([0.0], np.cumsum(np.concatenate((prof, prof[:w])))))
    mov = (ext[w:w + Pi] - ext[:Pi]) / w
    rest = (prof.sum() - mov * w) / (Pi - w)
    c = mov - rest
    k = int(np.argmax(np.abs(c)))
    return abs(c[k]) / 2.0, (1 if c[k] > 0 else -1), k


def _ncorr(x, L):
    a = x[:-L]
    b = x[L:]
    # einsum: avoids OpenBLAS thread start-up (ms per call on many-core hosts)
    den = math.sqrt(float(np.einsum("i,i", a, a)) * float(np.einsum("i,i", b, b))) + 1e-9
    return float(np.einsum("i,i", a, b)) / den


def _parab(r, k):
    if 0 < k < len(r) - 1:
        a, b, c = r[k - 1], r[k], r[k + 1]
        den = a - 2 * b + c
        if den < 0:
            return k + max(-0.5, min(0.5, 0.5 * (a - c) / den))
    return float(k)


def refine_slant(img, D=16, maxs=40, up=8, min_rho=0.15, max_delta=1e9):
    """Estimate residual slant (pixels per row) of a finished image from the
    correlation of the horizontal gradient of rows i and i+D (cross-spectrum
    summed over rows, 1/up pixel resolution); return (corrected image, delta)."""
    H, W = img.shape
    if H < 4 * D + 20:
        return img, 0.0
    g = np.diff(img.astype(np.float32), axis=1)
    g -= g.mean(axis=1, keepdims=True)
    F = np.fft.rfft(g, axis=1)
    C = (np.conj(F[:-D]) * F[D:]).sum(axis=0)
    n = up * (W - 1)
    S = np.fft.irfft(C, n=n) * up
    lags = np.arange(-maxs * up, maxs * up + 1)
    r = S[lags % n]
    k = int(np.argmax(r))
    if r[k] <= 0 or k in (0, len(r) - 1):
        return img, 0.0
    rho = r[k] / math.sqrt(float((g[:-D] ** 2).sum()) * float((g[D:] ** 2).sum()) + 1e-9)
    s = (lags[0] + _parab(r, k)) / up      # row i+D shifted right by s px
    delta = s / D
    if rho < min_rho or abs(delta) > max_delta:
        return img, 0.0
    if abs(delta) < 0.002:
        return img, delta
    flat = img.astype(np.float32).ravel()
    pos = (np.arange(H)[:, None] * (W + delta) + np.arange(W)[None, :]).ravel()
    pos = np.clip(pos, 0, flat.size - 1)
    out = np.interp(pos, np.arange(flat.size), flat).reshape(H, W)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), delta


class _Img:
    def __init__(self, lpm, ioc, L0, Pt, ts, off, black, white, mid=False):
        self.lpm = lpm
        self.ioc = ioc
        self.W = int(round(math.pi * ioc))
        self.L0 = float(L0)          # sample index (8 kHz) of line 0 start
        self.Pt = float(Pt)          # line period in samples (8 kHz)
        self.ts = ts
        self.off = off
        self.black = black           # Hz rel. 1900
        self.white = white
        self.mid = mid
        self.rows = []
        self.hw = max(0.5 * self.Pt / self.W, 2.0)
        self.max_lines = int(lpm * 30)       # 30 min cap

    def line_span(self, k):
        st = self.L0 + k * self.Pt
        return int(math.floor(st - 6.0)) - 2, int(math.ceil(st + self.Pt + 6.0)) + 2

    def render(self, k, p, pbase, bias=0.0):
        i0, i1 = self.line_span(k)
        seg = p[i0 - pbase:i1 - pbase].astype(np.complex128)
        cs = np.concatenate(([0j], np.cumsum(seg)))
        st = self.L0 + k * self.Pt - i0
        c = st + (np.arange(self.W) + 0.5) * (self.Pt / self.W)
        xs = np.arange(cs.size)
        a = c - self.hw
        b = c + self.hw
        S = (np.interp(b, xs, cs.real) - np.interp(a, xs, cs.real)) - 2 * self.hw * bias + \
            1j * (np.interp(b, xs, cs.imag) - np.interp(a, xs, cs.imag))
        f = np.angle(S) * _HZ
        v = (f - self.black) / (self.white - self.black)
        self.rows.append(np.clip(v * 255.0 + 0.5, 0, 255).astype(np.uint8))

    def array(self, n=None):
        rows = self.rows if n is None else self.rows[:n]
        if not rows:
            return np.zeros((0, self.W), np.uint8)
        return np.vstack(rows)


class WefaxWatcher:
    """Continuous WEFAX watcher for one audio channel (8 kHz int16)."""

    HIST_S = 32                  # d4/p history kept while idle (s)
    MID_WIN_S = 24               # window for running-fax detection
    MIN_LINES = 30               # discard shorter images
    HIT_RATIO = 0.40
    R_GOOD = 2.0
    REFINE_PPM = 40.0            # max. image-based slant correction after phasing (0 = off)
    PIX_BIAS = 0.7               # fraction of the noise bias removed from pixels
    PIX_R = [3.0, 6.0, 12.0, 25.0]   # pixel half-window (samples) vs in/out band ratio
    PIX_HW = [5.0, 3.5, 2.5, 2.0]

    def __init__(self, out_dir, label="", mid_image=True, debug=False):
        self.debug = debug
        self.out_dir = out_dir
        self.label = label
        self.mid_image = mid_image
        os.makedirs(out_dir, exist_ok=True)
        self._zi = np.zeros(len(_LP) - 1, np.complex128)
        self._ylast = 0j
        self._n = 0                                  # samples received
        self._p = np.zeros(0, np.complex64)          # p history (8 kHz)
        self._pbase = 0
        self._d4 = np.zeros(0, np.float32)           # frequency (Hz rel 1900) at 2 kHz
        self._d4base = 0
        self._xr = np.zeros(0)                       # raw audio not yet framed
        self._nf = 0                                 # frames done
        self._fr = []                                # (mu, h300, h450, h675, R) per frame
        self._frbase = 0
        self._anchor = (0, time.time())
        self._state = "idle"
        self._img: _Img | None = None
        self._cool = 0                               # no running-fax detection before this frame
        self._mid_prev = None
        self._nvar = []                              # per-frame noise variance estimates
        self._bias = 0.0
        self._off = 0.0
        self._off_true = None
        self._ioc = 576
        self._tone_ts = 0.0
        self._tone_first = 0
        self._tone_last = 0
        self._s0 = 0
        self.events = []                             # debug log

    # -- helpers ------------------------------------------------------------
    def _ts_of(self, n):
        n0, t0 = self._anchor
        return t0 + (n - n0) / FS

    def _frames(self, a, b):
        return self._fr[max(0, a - self._frbase):max(0, b - self._frbase)]

    def _hits(self, j, k):
        """number of hits of tone j in the last k frames"""
        return sum(1 for f in self._fr[-k:] if f[1 + j])

    def _d4seg(self, a, b):
        return self._d4[max(0, a - self._d4base):max(0, b - self._d4base)]

    # -- main entry ---------------------------------------------------------
    def feed(self, pcm, ts=None):
        out = []
        x = np.asarray(pcm).ravel()
        if x.size == 0:
            return out
        if ts is None:
            ts = time.time()
        if x.size > 1600:
            for i in range(0, x.size, 800):
                out += self.feed(x[i:i + 800], ts + i / FS)
            return out
        self._anchor = (self._n, float(ts))
        xf = x.astype(np.float64) * (1.0 / 32768.0)
        n0 = self._n
        mix = _MIX[(n0 + np.arange(xf.size)) % 80]
        y, self._zi = lfilter(_LP, 1.0, xf * mix, zi=self._zi)
        yprev = np.concatenate(([self._ylast], y[:-1]))
        self._ylast = y[-1]
        p = (y * np.conj(yprev)).astype(np.complex64)
        self._p = np.concatenate((self._p, p))
        self._n += xf.size
        # d4
        have = self._d4base + self._d4.size
        avail = self._n // DEC
        if avail > have:
            a = have * DEC - self._pbase
            s = self._p[a:a + (avail - have) * DEC].astype(np.complex128).reshape(-1, DEC).sum(axis=1)
            self._d4 = np.concatenate((self._d4, (np.angle(s) * _HZ4).astype(np.float32)))
        # frames
        self._xr = np.concatenate((self._xr, xf))
        while self._xr.size >= FRAME:
            fx = self._xr[:FRAME]
            self._xr = self._xr[FRAME:]
            self._frame(fx)
            r = self._step()
            if r:
                out.extend(r)
        if self._img is not None and self._state == "image":
            r = self._render_avail()
            if r:
                out.extend(r)
        self._trim()
        return out

    def progress(self):
        if self._state == "idle":
            return None
        im = self._img
        if im is not None:
            return {"state": self._state, "lpm": im.lpm, "ioc": im.ioc, "lines": len(im.rows),
                    "ts": float(im.ts), "offset_hz": round(float(getattr(im, "off_rep", im.off)), 1)}
        return {"state": self._state, "lpm": getattr(self, "_ph_lpm", 0) or 0,
                "ioc": self._ioc, "lines": 0, "ts": float(self._tone_ts),
                "offset_hz": round(float(self._off), 1)}

    def preview(self):
        """uint8 grey array (lines x width) of the image being received, or None."""
        if self._img is None or not self._img.rows:
            return None
        return self._img.array()

    def flush(self):
        """Finish an image in progress (e.g. at end of a recording)."""
        if self._state == "image" and self._img is not None:
            r = self._finish(len(self._img.rows), partial=True)
            return [r] if r else []
        self._state = "idle"
        return []

    # -- per frame analysis -------------------------------------------------
    def _frame(self, fx):
        f = self._nf
        self._nf += 1
        b = (f + 1) * (FRAME // DEC)
        w = self._d4seg(b - _TWIN, b).astype(np.float64)
        if w.size < _TWIN:
            self._fr.append((0.0, False, False, False, 0.0, False, 0j))
            return
        np.clip(w, -1200.0, 1200.0, out=w)
        mu = float(w.mean())
        xc = w - mu
        v = float(np.einsum("i,i", xc, xc)) / xc.size + 1e-9
        amps = np.abs(np.einsum("ij,j->i", _TE, xc)) * _TNORM
        ratio = amps * amps * 0.5 / v
        hits = (ratio > self.HIT_RATIO) & (amps > 280.0) & (amps < 760.0)
        weak450 = bool(ratio[1] > 0.22 and 220.0 < amps[1] < 760.0)
        # in-band / out-of-band power density of the audio
        S = np.abs(np.fft.rfft(fx * _W800)) ** 2       # 10 Hz bins
        off = self._off if self._state != "idle" else 0.0
        c = (F_CENTER + off) / 10.0
        pin = S[int(c - 45):int(c + 45)].mean()
        pout = S[30:100].mean()
        nv = min(pout / _W2, 0.9 * float(np.einsum("i,i", fx, fx)) / FRAME)
        self._nvar.append(nv)
        if len(self._nvar) > 30:
            del self._nvar[0]
        self._bias = float(np.median(self._nvar)) * _R1
        R = float(pin / (pout + 1e-12)) if pin > 1e-10 else 0.0
        a8 = f * FRAME - self._pbase
        ps = complex(np.sum(self._p[max(0, a8):a8 + FRAME].astype(np.complex128))) - FRAME * self._bias
        self._fr.append((mu, bool(hits[0]), bool(hits[1]), bool(hits[2]), R, weak450, ps))

    # -- state machine ------------------------------------------------------
    def _start_tone(self):
        """-> ioc if a start tone is present in the recent frames."""
        if len(self._fr) < 25:
            return None
        if self._hits(0, 25) >= 15:
            return 576
        if self._hits(2, 25) >= 15:
            return 288
        return None

    def _step(self):
        out = []
        f = self._nf - 1
        st = self._state
        if st == "idle":
            ioc = self._start_tone()
            if ioc:
                self._enter_tone(ioc)
            elif self.mid_image and f >= self._cool and f % 20 == 0:
                self._mid_check()
        elif st == "tone":
            j = 0 if self._ioc == 576 else 2
            last = self._fr[-1]
            if last[1 + j]:
                self._tone_last = f
            if f - self._tone_last >= 3:
                fr = self._frames(self._tone_first + 2, self._tone_last - 1)
                mus = [x[0] for x in fr if x[1 + j]] or [x[0] for x in self._frames(self._tone_first, self._tone_last + 1)]
                self._off = float(np.median(mus))
                pss = [x[6] for x in fr if x[1 + j]]
                self._off_true = float(np.angle(sum(pss)) * _HZ) if pss else self._off
                # window for frame k covers d4 of frames k-1..k -> tone ended near end of last hit frame
                self._s0 = (self._tone_last + 1) * FRAME - FRAME // 2
                self._enter_phasing(self._s0, None)
                self.events.append(("tone", self._tone_ts, self._ioc, self._off))
            elif f - self._tone_first > 200:
                self._state = "idle"
        elif st == "phasing":
            ioc = self._start_tone()
            if ioc and f - self._tone_last > 60:
                self._enter_tone(ioc)
            else:
                out += self._phasing_step()
        elif st == "image":
            out += self._image_checks()
        return out

    def _enter_tone(self, ioc):
        self._state = "tone"
        self._ioc = ioc
        j = 0 if ioc == 576 else 2
        k = self._nf - 25
        fr = self._frames(k, self._nf)
        first = next(i for i, x in enumerate(fr) if x[1 + j])
        self._tone_first = k + first
        self._tone_last = self._nf - 1
        self._tone_ts = self._ts_of(self._tone_first * FRAME)
        self._off = float(self._fr[-1][0])
        self._off_true = None
        self._img = None

    def _enter_phasing(self, s0, ts):
        self._state = "phasing"
        self._ph_lock = None
        self._ph_lpm = 0
        self._ph_meas = []          # (line index, pulse centre in d4 samples)
        self._ph_i = 0
        self._ph_fail = 0
        self._ph_ab = None
        if ts is not None:
            self._tone_ts = ts

    def _lum4(self, a, b):
        return np.clip((self._d4seg(a, b).astype(np.float64) - self._off) / DEV, -1.0, 1.0)

    def _phasing_step(self):
        out = []
        f = self._nf - 1
        avail4 = self._d4base + self._d4.size - 20
        a4 = self._s0 // DEC + int(0.3 * FS4)
        if self._ph_lock is None:
            if f % 5:
                return out
            if (avail4 - a4) > 25 * FS4:
                self.events.append(("phasing-timeout", self._tone_ts))
                self._state = "idle"
                self._cool = self._nf
                return out
            if (avail4 - a4) < 4.2 * (60.0 / 60) * FS4:
                return out
            lum = self._lum4(a4, avail4)
            res = []
            for lpm in LPMS:
                r = fold_score(lum, 60.0 / lpm * FS4)
                if r is None:
                    return out
                res.append((lpm, r))
            best = max(r[0] for _, r in res)
            if best < 0.45:
                return out
            for lpm, (sc, pol, k) in res:
                if sc >= 0.75 * best:
                    break
            P4 = 60.0 / lpm * FS4
            self._ph_lock = (lpm, P4, pol, a4 + k)
            self._ph_sc = sc
            self._ph_P4nom = P4
            self._ph_lpm = lpm
            self._ph_i = 0
            self.events.append(("phasing-lock", lpm, pol, round(sc, 2)))
        lpm, P4, pol, e0 = self._ph_lock
        w = max(2, int(round(0.05 * P4)))
        while True:
            i = self._ph_i
            cpred = self._ph_pred(i)
            lo = int(cpred - 1.5 * w)
            hi = int(cpred - 0.5 * w + P4) + 1
            if hi > avail4:
                break
            self._ph_i += 1
            if lo < self._d4base:
                continue
            lum = uniform_filter1d(pol * self._lum4(lo, hi), max(3, w // 4))
            win = lum[:3 * w]
            pk, height = _box_peak(win, w)
            centre = lo + pk + 0.5 * w
            ps = int(round(pk))
            rest = np.concatenate((win[:max(0, ps - w // 2)], win[ps + w + w // 2:], lum[3 * w:]))
            sides = np.concatenate((win[max(0, ps - w // 2):max(0, ps - w // 8)],
                                    win[ps + w + w // 8:ps + w + w // 2]))
            base = float(np.median(rest))
            amp = height - base
            sc = self._ph_sc
            ok = (abs(pk - w) <= 0.5 * w and amp > 1.1 * sc and base < -0.3 * sc
                  and (sides.size == 0 or float(np.mean(sides)) - base < 0.35 * amp)
                  and float(np.mean(rest > base + 0.5 * amp)) < 0.05)
            if self.debug:
                self.events.append(("ph", i, round(amp, 2), round(base, 2), round(float(np.mean(sides)) - base, 2) if sides.size else None, round(float(np.mean(rest > base + 0.5 * amp)), 3), ok))
            if ok:
                self._ph_meas.append((i, centre))
                self._ph_fail = 0
                if len(self._ph_meas) >= 4:
                    self._ph_fit()
            else:
                self._ph_fail += 1
                if self._ph_fail >= 3 and len(self._ph_meas) >= 6:
                    return self._phasing_to_image(i - 2)
            if i > lpm // 2 + 20 and len(self._ph_meas) < 3:      # never matched
                self.events.append(("phasing-nomatch", self._tone_ts))
                self._state = "idle"
                return out
            if i >= lpm * 1.5 and len(self._ph_meas) >= 6:                     # 90 s of phasing
                return self._phasing_to_image(i + 1)
        return out

    def _ph_pred(self, i):
        lpm, P4, pol, e0 = self._ph_lock
        w = max(2, int(round(0.05 * P4)))
        if self._ph_ab is not None:
            a, b = self._ph_ab
            return a + b * i
        return e0 + 0.5 * w + i * P4

    def _ph_fit(self):
        m = np.array(self._ph_meas, float)
        i, c = m[:, 0], m[:, 1]
        P4 = self._ph_lock[1]
        keep = np.ones(len(i), bool)
        for _ in range(3):
            if keep.sum() < 3:
                break
            ik, ck = i[keep], c[keep]           # closed-form LS line (np.polyfit -> threaded LAPACK)
            im_, cm_ = ik.mean(), ck.mean()
            b = float(((ik - im_) * (ck - cm_)).sum() / ((ik - im_) ** 2).sum())
            a = cm_ - b * im_
            res = c - (a + b * i)
            mad = np.median(np.abs(res[keep])) + 0.05
            keep = np.abs(res) < 4 * mad + 0.5
        if abs(b / P4 - 1) < 0.01:
            self._ph_ab = (a, b)

    def _phasing_to_image(self, i_img):
        lpm, P4, pol, e0 = self._ph_lock
        w = max(2, int(round(0.05 * P4)))
        a, b = self._ph_ab if self._ph_ab is not None else (e0 + 0.5 * w, P4)
        # measured levels on phasing lines
        pulse, rest = [], []
        for i, c in self._ph_meas[-40:]:
            c0 = int(round(c))
            pulse.append(self._d4seg(c0 - w // 3, c0 + w // 3))
            rest.append(self._d4seg(c0 + w, int(c0 + P4 - w)))
        # bias-free levels for the reported offset: angle of (sum p - n * bias)
        sp, sr = 0j, 0j
        for i, c in self._ph_meas[-40:]:
            c0 = int(round(c))
            for (u, v), acc in (((c0 - w // 3, c0 + w // 3), 0), ((c0 + w, int(c0 + P4 - w)), 1)):
                a8, b8 = u * DEC - self._pbase, v * DEC - self._pbase
                if a8 < 0:
                    continue
                z = complex(np.sum(self._p[a8:b8].astype(np.complex128))) - (b8 - a8) * self._bias
                if acc == 0:
                    sp += z
                else:
                    sr += z
        off_rep = None
        if sp != 0 and sr != 0:
            lvl = sorted((np.angle(sp) * _HZ, np.angle(sr) * _HZ))
            if 500 < lvl[1] - lvl[0] < 1100:
                off_rep = 0.5 * (lvl[0] + lvl[1])
        lp = float(np.median(np.concatenate(pulse))) if pulse else self._off + pol * DEV
        lr = float(np.median(np.concatenate(rest))) if rest else self._off - pol * DEV
        white, black = (lp, lr) if pol > 0 else (lr, lp)
        if 250.0 < white - black < 1000.0 and abs(0.5 * (white + black) - self._off) < 200:
            self._off = 0.5 * (white + black)
        else:
            white, black = self._off + DEV, self._off - DEV
        Pt = b * DEC
        # line start = leading edge of the pulse; d4 sample m covers p[4m..4m+3]
        L0 = (a + b * i_img - 0.5 * w) * DEC + 1.5
        ioc = getattr(self, "_ioc", 576) or 576
        self._img = _Img(lpm, ioc, L0, Pt, self._tone_ts, self._off, black, white)
        self._img.ppm = (b / self._ph_P4nom - 1) * 1e6
        self._img.off_rep = off_rep if off_rep is not None else (
            self._off_true if self._off_true is not None else self._off)
        self._img.phasing_lines = len(self._ph_meas)
        self._state = "image"
        self.events.append(("image", lpm, ioc, round(self._off, 1), round(self._img.ppm, 1)))
        return self._render_avail()

    # -- image ----------------------------------------------------------------
    def _render_avail(self):
        im = self._img
        Rm = float(np.median([x[4] for x in self._fr[-50:]])) if self._fr else 99.0
        im.hw = max(0.5 * im.Pt / im.W, float(np.interp(Rm, self.PIX_R, self.PIX_HW)))
        while True:
            k = len(im.rows)
            i0, i1 = im.line_span(k)
            if i1 > self._n:
                break
            if i0 < self._pbase:            # should not happen
                im.rows.append(np.full(im.W, 255, np.uint8))
                continue
            im.render(k, self._p, self._pbase, self.PIX_BIAS * self._bias)
            if len(im.rows) >= im.max_lines:
                return [x for x in [self._finish(len(im.rows), partial=True)] if x]
        return []

    def _line_at(self, n):
        im = self._img
        return max(0, int((n - im.L0) // im.Pt))

    def _image_checks(self):
        f = self._nf - 1
        out = []
        im = self._img
        weak = sum(1 for x in self._fr[-20:] if x[5]) >= 15
        if self._hits(1, 12) >= 8 or weak:          # stop tone
            k = 12 if not weak else 20
            fr = self._fr[-k:]
            first = f - (k - 1) + next(i for i, x in enumerate(fr) if (x[5] if weak else x[2]))
            nl = self._line_at(first * FRAME - 200)
            r = self._finish(nl, partial=False)
            self._cool = self._nf + 250
            return [r] if r else []
        ioc = self._start_tone()
        if ioc:
            nl = self._line_at((self._nf - 26) * FRAME)
            r = self._finish(nl, partial=True)
            self._enter_tone(ioc)
            return [r] if r else []
        if not hasattr(im, "good_f"):
            im.good_f = f
            im.start_f = f
        if self._fr[-1][4] > self.R_GOOD:
            im.good_f = f
        if f - im.start_f > 150:
            g = sum(1 for x in self._fr[-150:] if x[4] > self.R_GOOD)
            if g < 15:
                nl = self._line_at((im.good_f + 1) * FRAME)
                r = self._finish(nl, partial=True)
                return [r] if r else []
        return out

    def _finish(self, nlines, partial):
        im = self._img
        self._img = None
        self._state = "idle"
        if im is None:
            return None
        nlines = min(nlines, len(im.rows))
        if nlines < self.MIN_LINES:
            self.events.append(("discard", nlines))
            return None
        arr = im.array(nlines)
        delta = 0.0
        if im.mid:
            arr, delta = refine_slant(arr)
        elif self.REFINE_PPM > 0:          # small residual after the phasing fit
            arr, delta = refine_slant(arr, max_delta=self.REFINE_PPM * 1e-6 * im.W)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(im.ts))
        lab = ("%s_" % self.label) if self.label else ""
        fname = "wefax_%s%s_%d_%d.png" % (lab, stamp, im.lpm, im.ioc)
        path = os.path.join(self.out_dir, fname)
        write_png(path, arr)
        return {"path": path, "ts": float(im.ts), "lpm": int(im.lpm), "ioc": int(im.ioc),
                "lines": int(nlines), "offset_hz": round(float(getattr(im, "off_rep", im.off)), 1), "partial": bool(partial),
                "mid_start": bool(im.mid),
                "ppm": round(float(getattr(im, "ppm", 0.0)) + delta / im.W * 1e6, 1)}

    # -- running fax without start tone ----------------------------------------
    def _mid_check(self):
        N4 = self.MID_WIN_S * FS4
        end4 = self._d4base + self._d4.size
        if self._d4.size < N4 or (end4 - N4) * DEC < self._pbase:
            return
        nfr = self.MID_WIN_S * 10
        if len(self._fr) < nfr:
            return
        if sum(1 for x in self._fr[-nfr:] if x[4] > self.R_GOOD) < 0.8 * nfr:
            self._mid_prev = None
            return
        seg = self._d4[-N4:].astype(np.float64)
        np.clip(seg, -1200, 1200, out=seg)
        sm = np.convolve(seg, np.ones(8) / 8.0, "valid")
        lo, hi = np.percentile(sm, [0.5, 99.5])
        spread = hi - lo
        if spread < 450 or spread > 1300:
            self._mid_prev = None
            return
        off = 0.5 * (hi + lo) if spread > 650 else hi - DEV
        x = np.clip(seg - off, -DEV, DEV)
        x -= x.mean()
        x = x.reshape(-1, 2).mean(axis=1)          # 1 kHz is plenty here
        refs = max(_ncorr(x, L) for L in (250, 357, 643, 750, 1250))
        found = None
        for lpm in (120, 60):
            Lc = int(60000 / lpm)
            span = Lc // 100 + 1
            r = np.array([_ncorr(x, L) for L in range(Lc - span, Lc + span + 1)])
            k = int(np.argmax(r))
            if r[k] > 0.4 and r[k] - refs > 0.25:
                found = (lpm, Lc - span + _parab(r, k), float(r[k]))
                break
        if found is None:
            self._mid_prev = None
            return
        if self._mid_prev is None or self._mid_prev[0] != found[0]:
            self._mid_prev = found
            return
        lpm, L1k, rho = found
        self._mid_prev = None
        # refine the period at 8 kHz on a 4 kHz-ish smoothed frequency track
        a8 = (end4 - N4) * DEC
        p = self._p[a8 - self._pbase:end4 * DEC - self._pbase].astype(np.complex128)
        s = p.reshape(-1, 2).sum(axis=1)                       # 4 kHz
        z = np.clip(np.angle(s) * _HZ - off, -DEV, DEV)
        z -= z.mean()
        Lc = L1k * 4.0
        lags = np.arange(int(Lc) - 6, int(Lc) + 8)
        r = np.array([_ncorr(z, int(L)) for L in lags])
        k = int(np.argmax(r))
        Pt = (lags[0] + _parab(r, k)) * 2.0
        P4 = Pt / DEC
        # phasing?
        lum = np.clip(seg - off, -DEV, DEV) / DEV
        fs_ = fold_score(lum, P4)
        self.events.append(("mid", lpm, round(rho, 2), round(Pt, 2), fs_ and round(fs_[0], 2)))
        ts = self._ts_of(a8)
        self._off = off
        self._ioc = 576
        if fs_ is not None and fs_[0] > 0.75:
            self._s0 = a8
            self._tone_ts = ts
            self._tone_last = self._nf
            self._enter_phasing(a8, ts)
            self._ph_lock = (lpm, P4, fs_[1], self._s0 // DEC + fs_[2])
            self._ph_sc = fs_[0]
            self._ph_P4nom = 60.0 / lpm * FS4
            self._ph_lpm = lpm
            return
        # start the picture where the signal began (up to the history limit)
        fa = a8 // FRAME
        while fa - 1 - self._frbase >= 0 and (fa - 1) * FRAME >= self._pbase + 2 * FS:
            fr = self._frames(fa - 10, fa)
            if len(fr) < 10 or sum(1 for x in fr if x[4] > self.R_GOOD) < 8:
                break
            fa -= 10
        if fa * FRAME < a8:
            a8 = fa * FRAME
            ts = self._ts_of(a8)
        self._img = _Img(lpm, 576, a8, Pt, ts, off, off - DEV, off + DEV, mid=True)
        self._img.ppm = (Pt / (60.0 / lpm * FS) - 1) * 1e6
        self._state = "image"

    # -- buffers ------------------------------------------------------------
    def _trim(self):
        keep_from = self._n - self.HIST_S * FS
        if self._state == "image" and self._img is not None:
            i0, _ = self._img.line_span(len(self._img.rows))
            keep_from = min(keep_from, i0 - 64)
        elif self._state in ("tone", "phasing"):
            keep_from = min(keep_from, getattr(self, "_s0", keep_from) - FS,
                            self._tone_first * FRAME if self._state == "tone" else keep_from)
        keep_from = max(0, keep_from)
        keep_from -= keep_from % (DEC * 1024)
        if keep_from - self._pbase >= 8 * FS:
            cut = keep_from - self._pbase
            self._p = self._p[cut:]
            self._pbase = keep_from
            c4 = keep_from // DEC - self._d4base
            self._d4 = self._d4[c4:]
            self._d4base = keep_from // DEC
        if len(self._fr) > 3000:
            d = len(self._fr) - 2000
            self._fr = self._fr[d:]
            self._frbase += d


def _main(argv):
    """Decode a WAV file: python3 decoder.py file.wav [out_dir]"""
    from math import gcd
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    if len(argv) < 2:
        print(_main.__doc__)
        return 1
    rate, x = wavfile.read(argv[1])
    x = np.asarray(x, np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.abs(x).max() <= 1.0:
        x *= 32767.0
    if rate != FS:
        g = gcd(int(rate), FS)
        x = resample_poly(x, FS // g, int(rate) // g)
    pcm = np.clip(x, -32768, 32767).astype(np.int16)
    w = WefaxWatcher(argv[2] if len(argv) > 2 else ".", "wav")
    t0 = time.time()
    for i in range(0, len(pcm), 800):
        for r in w.feed(pcm[i:i + 800], t0 + i / FS):
            print(r)
    for r in w.flush():
        print(r)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv))
