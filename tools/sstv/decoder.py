"""
Streaming SSTV detector + decoder for 8 kHz mono SSB audio (numpy + scipy only).

    w = SSTVWatcher("/path/to/out", label="14230")
    for block, ts in audio:              # int16, 8000 Hz, any block length
        for img in w.feed(block, ts):    # finished (or partial) images
            print(img)                   # {"path", "mode", "ts", "offset_hz", "lines", "partial"}
        p = w.progress()                 # None, or {"mode", "lines", "total_lines", "ts", "offset_hz"}

    python3 decoder.py recording.wav [out_dir]     # offline decode of a file

Signal path (8 kHz, vectorised):
  * A complex band-pass FIR (800..3000 Hz, centred on 1900 Hz) gives the analytic
    signal y[n].  p[n] = y[n] conj(y[n-1]) carries the instantaneous frequency in
    its phase; the angle of a sum of p over an interval is a power-weighted
    frequency estimate for that interval.
  * Band-limited noise gives p a known mean (noise_power * _R1) that pulls every
    estimate toward 1900 Hz.  The noise power is solved from the VIS leader (and
    later from the sync pulses) and that bias is subtracted.
  * Idle: p is summed into 1 ms bins and two 1900 Hz leaders followed by a
    1200 Hz start bit are searched with cumulative sums over all candidate times
    (cheap).  Candidates are verified with coherent tone correlators on y
    (start/stop bit at 1200 Hz, data bits 1100/1300 Hz, offset-corrected),
    even parity and a known mode code.  The leader frequency gives the offset.
  * Image: y is buffered.  Each sync pulse is located with a matched filter
    near the predicted position, refined on its trailing edge and validated as
    a clean 1200 Hz tone; a robust linear fit of sync position vs. line index
    gives the line period (slant correction).  At the end the whole image is
    rendered from that fit: pixel value = angle of sum p over the pixel
    interval (interval widened at low SNR).  Signal loss (no valid sync for
    4 s) ends the image as partial.
"""
from __future__ import annotations

import math
import os
import struct
import time
import zlib
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import firwin, lfilter

try:  # optional
    from PIL import Image as _PILImage
except Exception:  # pragma: no cover
    _PILImage = None

FS = 8000
# pixel integration window vs. estimated in-band SNR (dB): wider at low SNR
PIX_WIN_SNR = [8.0, 12.0, 20.0, 28.0]
PIX_WIN_LEN = [7.0, 5.0, 3.0, 1.5]
_MS = FS / 1000.0          # samples per ms
_BIN = 8                   # samples per 1 ms bin

# ---------------------------------------------------------------------------
# Mode table
# ---------------------------------------------------------------------------


@dataclass
class Mode:
    name: str
    vis: int
    width: int
    height: int
    period_ms: float          # sync-to-sync period
    sync_ms: float
    first_sync_ms: float      # VIS stop-bit end -> first sync pulse start
    kind: str                 # 'rgb' | 'pd' | 'robot36' | 'robot72'
    channels: dict = field(default_factory=dict)   # name -> (start_ms rel. sync, dur_ms)

    @property
    def rows_per_sync(self):
        return 2 if self.kind == "pd" else 1

    @property
    def n_syncs(self):
        return self.height // self.rows_per_sync

    def chan_extent(self):
        lo = min(s for s, d in self.channels.values())
        hi = max(s + d for s, d in self.channels.values())
        return lo, hi


def _martin(name, vis, width, scan):
    sync, gap = 4.862, 0.572
    g = sync + gap
    b = g + scan + gap
    r = b + scan + gap
    period = sync + 4 * gap + 3 * scan
    return Mode(name, vis, width, 256, period, sync, 0.0, "rgb",
                {"G": (g, scan), "B": (b, scan), "R": (r, scan)})


def _scottie(name, vis, width, scan):
    sync, gap = 9.0, 1.5
    # line: gap G gap B sync gap R ; sync is mid-line
    r = sync + gap
    b = -scan
    g = -scan - gap - scan
    period = sync + 3 * gap + 3 * scan
    # standard: a 9 ms "start sync" follows the VIS, then line 0 (gap G gap B sync ...)
    first = sync + gap + scan + gap + scan
    return Mode(name, vis, width, 256, period, sync, first, "rgb",
                {"G": (g, scan), "B": (b, scan), "R": (r, scan)})


def _pd(name, vis, width, height, px):
    sync, porch = 20.0, 2.08
    s = width * px
    y0 = sync + porch
    period = sync + porch + 4 * s
    return Mode(name, vis, width, height, period, sync, 0.0, "pd",
                {"Y0": (y0, s), "Cr": (y0 + s, s), "Cb": (y0 + 2 * s, s), "Y1": (y0 + 3 * s, s)})


MODES = {}
for _m in (
    _martin("Martin M1", 44, 320, 146.432),
    _martin("Martin M2", 40, 160, 73.216),
    _scottie("Scottie S1", 60, 320, 138.240),
    _scottie("Scottie S2", 56, 160, 88.064),
    _scottie("Scottie DX", 76, 320, 345.600),
    Mode("Robot 36", 8, 320, 240, 150.0, 9.0, 0.0, "robot36",
         {"Y": (12.0, 88.0), "SEP": (100.5, 3.5), "C": (106.0, 44.0)}),
    Mode("Robot 72", 12, 320, 240, 300.0, 9.0, 0.0, "robot72",
         {"Y": (12.0, 138.0), "Cr": (156.0, 69.0), "Cb": (231.0, 69.0)}),
    _pd("PD50", 93, 320, 256, 0.286),
    _pd("PD90", 99, 320, 256, 0.532),
    _pd("PD120", 95, 640, 496, 0.190),
    _pd("PD160", 98, 512, 400, 0.382),
    _pd("PD180", 96, 640, 496, 0.286),
    _pd("PD240", 97, 640, 496, 0.382),
    _pd("PD290", 94, 800, 616, 0.286),
):
    MODES[_m.vis] = _m

# ---------------------------------------------------------------------------
# PNG output
# ---------------------------------------------------------------------------


def write_png(path, rgb):
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if _PILImage is not None:
        _PILImage.fromarray(rgb, "RGB").save(path)
        return
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_NTAPS = 41
_BP = (firwin(_NTAPS, 1100.0, fs=FS) *
       np.exp(2j * np.pi * 1900.0 * np.arange(_NTAPS) / FS)).astype(np.complex128)
_RAD2HZ = FS / (2 * np.pi)
# For white input noise of variance s:  E|y|^2 = s*_R0,  E[y[n] y*[n-1]] = s*_R1.
# The second term biases every frequency estimate towards 1900 Hz; it is
# removed by subtracting s*_R1 from p once s is known.
_R0 = float(np.sum(np.abs(_BP) ** 2))
_R1 = complex(np.sum(_BP[1:] * np.conj(_BP[:-1])))


def _noise_solve(psum, esum, n):
    """Noise power s from sums of p and |y|^2 over n samples of one steady tone.

    Model: mean(p) = A e^{jw} + s*R1, mean(|y|^2) = A + s*R0  ->  |pm - s R1| = P - s R0.
    """
    if n <= 0 or esum <= 0:
        return 0.0
    pm, P = psum / n, esum / n
    a = abs(_R1) ** 2 - _R0 ** 2
    b = 2 * P * _R0 - 2 * (pm * np.conj(_R1)).real
    c = abs(pm) ** 2 - P ** 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return 0.0
    roots = [(-b + sgn * math.sqrt(disc)) / (2 * a) for sgn in (1, -1)]
    roots = [r for r in roots if 0 <= r <= P / _R0]
    return float(min(roots)) if roots else 0.0


def _ang_hz(z):
    return np.angle(z) * _RAD2HZ


def _interp_cum(cum, x):
    """cum[i] = sum p[0:i]; linear interpolation at fractional positions x."""
    x = np.clip(x, 0, len(cum) - 1)
    return np.interp(x, np.arange(len(cum)), cum.real) + 1j * np.interp(x, np.arange(len(cum)), cum.imag)


def _ycc_to_rgb(y, cb, cr):
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    return np.stack([r, g, b], axis=-1)


# ---------------------------------------------------------------------------
# Image decoding state
# ---------------------------------------------------------------------------

class _Image:
    LOST_S = 4.0        # signal-lost timeout
    SYNC_THR = 0.45     # matched-filter score for a usable sync pulse
    TONE_THR = 0.35     # coherent 1200 Hz energy fraction inside the pulse

    def __init__(self, mode: Mode, t_img: int, start_n: int, offset: float, ts: float, noise=0.0):
        self.mode = mode
        self.offset = offset
        self.noise = noise                     # input noise power estimate (see _noise_solve)
        self.snr_db = 99.0                     # in-band SNR estimate (from sync pulses)
        self.ts = ts
        self.t_img = t_img                     # absolute sample index of VIS end
        self.start_n = start_n                 # absolute index of buf[0]
        lo, hi = mode.chan_extent()
        self.lo, self.hi = lo, hi
        total_ms = mode.first_sync_ms + mode.n_syncs * mode.period_ms + max(hi, 0) + 2000
        self.buf = np.zeros(int((t_img - start_n) + total_ms * _MS), np.complex64)  # analytic y
        self.fill = 0
        self.per = mode.period_ms * _MS        # nominal period in samples
        self.s0 = t_img + mode.first_sync_ms * _MS
        self.k = 0                             # next sync index to process
        self.sync_pos = np.full(mode.n_syncs, np.nan)
        self.sync_q = np.zeros(mode.n_syncs)
        self.fit = None                        # (a, b) absolute positions
        self.last_good = -1
        self.done = False
        self.lost = False

    # -- data -----------------------------------------------------------
    def append(self, y):
        n = min(len(y), len(self.buf) - self.fill)
        self.buf[self.fill:self.fill + n] = y[:n]
        self.fill += n
        return n < len(y)   # overflow

    def _p(self, a, b):
        """lag-1 products p[i] = y[i] conj(y[i-1]) for buffer indices a..b-1 (a >= 1)."""
        return self.buf[a:b] * np.conj(self.buf[a - 1:b - 1])

    def tone_frac(self, a, b, f):
        """Coherent energy fraction of a tone at f Hz in buffer samples [a, b)."""
        a, b = max(int(a), 0), min(int(b), self.fill)
        if b - a < 8:
            return 0.0
        seg = self.buf[a:b].astype(np.complex128)
        ref = np.exp(-2j * np.pi * f * np.arange(b - a) / FS)
        e = float(np.sum(seg.real ** 2 + seg.imag ** 2))
        return float(abs(np.dot(seg, ref)) ** 2 / ((b - a) * e)) if e > 0 else 0.0

    @property
    def end_n(self):
        return self.start_n + self.fill

    # -- sync tracking --------------------------------------------------
    def _predict(self, k):
        if self.fit is not None:
            return self.fit[0] + self.fit[1] * k
        if self.last_good >= 0:
            return self.sync_pos[self.last_good] + (k - self.last_good) * self.per
        return self.s0 + k * self.per

    def _refit(self, upto=None):
        ks = np.nonzero(self.sync_q[:upto] >= self.SYNC_THR)[0]
        if len(ks) < 3:
            self.fit = None
            return
        x = ks.astype(float)
        y = self.sync_pos[ks]
        w = self.sync_q[ks]
        keep = np.ones(len(ks), bool)
        a = b = None
        for _ in range(3):
            if keep.sum() < 3:
                break
            A = np.vstack([np.ones(keep.sum()), x[keep]]).T * w[keep, None]
            a, b = np.linalg.lstsq(A, y[keep] * w[keep], rcond=None)[0]
            bmin, bmax = self.per * 0.985, self.per * 1.015
            if not (bmin <= b <= bmax):
                b = min(max(b, bmin), bmax)
                a = np.median(y[keep] - b * x[keep])
            res = np.abs(y - (a + b * x))
            tol = max(1.5 * _MS, 2.5 * np.median(res[keep]))
            nk = res <= tol
            if (nk == keep).all():
                break
            keep = nk
        if a is not None:
            self.fit = (a, b)

    def _find_sync(self, pred, half):
        m = self.mode
        ns = int(round(m.sync_ms * _MS))
        nf = max(8, ns // 2)
        lo = int(pred - half) - nf - 8 - self.start_n
        hi = int(pred + half) + ns + nf + 8 - self.start_n
        lo0 = max(lo, 2)
        if hi > self.fill:
            return None
        seg = self._p(lo0 - 1, hi) - np.complex64(self.noise * _R1)
        cs = np.concatenate(([0], np.cumsum(seg)))
        f = _ang_hz(cs[8:] - cs[:-8]) - self.offset           # 1 ms moving sum
        ind = np.clip((1500.0 - f) / 300.0, 0.0, 1.0)
        base = lo0 - 1 + 4                                     # buf index of ind[0] (centred)
        ci = np.concatenate(([0], np.cumsum(ind)))
        j = np.arange(int(pred - half) - self.start_n, int(pred + half) + 1 - self.start_n) - base
        j = j[(j - nf >= 0) & (j + ns + nf < len(ind))]
        if len(j) == 0:
            return None
        inside = (ci[j + ns] - ci[j]) / ns
        left = (ci[j] - ci[j - nf]) / nf
        right = (ci[j + ns + nf] - ci[j + ns]) / nf
        score = inside - 0.5 * (left + right)
        i = int(np.argmax(score))
        # sub-sample refine (parabolic)
        jj = float(j[i])
        if 0 < i < len(score) - 1:
            d = score[i - 1] - 2 * score[i] + score[i + 1]
            if d < 0:
                jj += 0.5 * (score[i - 1] - score[i + 1]) / d
        # refine with the trailing edge (sync -> 1500 Hz porch in every mode):
        # edge = a + integral of the indicator from mid-sync to past the edge
        a = int(round(jj + ns / 2))
        b = min(int(round(jj + ns + nf)), len(ind))
        edge = a + float(ind[a:b].sum())
        pos = edge - ns + base - 0.5 + self.start_n
        # validate: the pulse interior must be a clean 1200 Hz tone
        i0 = pos - self.start_n
        tf = self.tone_frac(i0 + 0.15 * ns, i0 + 0.85 * ns, 1200.0 + self.offset)
        q = float(score[i]) if tf >= self.TONE_THR else 0.0
        return pos, q

    def process(self):
        """Process all syncs for which data is available. Returns True when finished."""
        m = self.mode
        while self.k < m.n_syncs:
            k = self.k
            pred = self._predict(k)
            ngood = int((self.sync_q[:k] >= self.SYNC_THR).sum())
            half = (6.0 if ngood >= 4 else 20.0) * _MS
            half = min(half, 0.3 * self.per)
            r = self._find_sync(pred, half)
            if r is None:
                break
            pos, q = r
            self.sync_pos[k] = pos
            self.sync_q[k] = q
            if q >= self.SYNC_THR:
                if self.fit is None or abs(pos - pred) < 2.5 * _MS:
                    self.last_good = k
                self._refit(k + 1)
            self.k += 1
            gap = k - self.last_good
            if gap * m.period_ms >= self.LOST_S * 1000 and gap >= 6:
                self.lost = True
                self.done = True
                return True
        if self.k >= m.n_syncs:
            last = self._predict(m.n_syncs - 1)
            if self.end_n >= last + self.hi * _MS * 1.02 + 16:
                self.done = True
                return True
        return False

    # -- rendering ------------------------------------------------------
    def rows_done(self):
        if self.lost:
            return (self.last_good + 1) * self.mode.rows_per_sync
        return min(self.k, self.mode.n_syncs) * self.mode.rows_per_sync

    def _update_noise(self, S, nsync):
        """Re-estimate the noise power from the interior of all good sync pulses."""
        good = np.nonzero(self.sync_q[:nsync] >= self.SYNC_THR)[0]
        if len(good) < 5:
            return
        ns = self.mode.sync_ms * _MS
        psum, esum, n = 0j, 0.0, 0
        for k in good:
            a = int(S[k] + 0.25 * ns) - self.start_n
            b = int(S[k] + 0.75 * ns) - self.start_n
            if a < 0 or b > self.fill:
                continue
            if a < 1:
                continue
            psum += complex(self._p(a, b).astype(np.complex128).sum())
            yy = self.buf[a:b].astype(np.complex128)
            esum += float(np.sum(yy.real ** 2 + yy.imag ** 2))
            n += b - a
        if n > 0:
            self.noise = _noise_solve(psum, esum, n)
            sig = esum / n - self.noise * _R0
            self.snr_db = 10 * math.log10(max(sig, 1e-12) / max(self.noise * _R0, 1e-12))

    def render(self, nsync=None):
        m = self.mode
        if nsync is None:
            nsync = self.k
        nsync = max(0, min(nsync, m.n_syncs))
        img = np.zeros((m.height, m.width, 3), np.uint8)
        if nsync == 0:
            return img
        self._refit()
        ks = np.arange(nsync)
        if self.fit is not None:
            S = self.fit[0] + self.fit[1] * ks
            scale = self.fit[1] / self.per
        else:
            S = np.where(np.isnan(self.sync_pos[:nsync]), self.s0 + ks * self.per, self.sync_pos[:nsync])
            scale = 1.0
        self._update_noise(S, nsync)
        p = np.zeros(self.fill, np.complex128)
        if self.fill > 1:
            p[1:] = self._p(1, self.fill)
        cum = np.concatenate(([0], np.cumsum(p)))
        del p
        cum -= self.noise * _R1 * np.arange(len(cum))

        # SNR-adaptive minimum integration window (samples) per pixel
        wmin = float(np.interp(self.snr_db, PIX_WIN_SNR, PIX_WIN_LEN))

        def chan(name, w=m.width):
            st, du = m.channels[name]
            pw = du / w * _MS                                # pixel width in samples
            half = 0.5 * max(pw, wmin) / _MS                 # ms
            cen = st + (np.arange(w) + 0.5) * (du / w)
            x = S[:, None] - self.start_n + scale * cen[None, :] * _MS
            dx = scale * half * _MS
            c0 = _interp_cum(cum, x - dx)
            c1 = _interp_cum(cum, x + dx)
            f = _ang_hz(c1 - c0) - self.offset
            return np.clip((f - 1500.0) * (255.0 / 800.0), 0, 255)

        if m.kind == "rgb":
            rgb = np.stack([chan("R"), chan("G"), chan("B")], axis=-1)
        elif m.kind == "pd":
            y0, cr, cb, y1 = chan("Y0"), chan("Cr"), chan("Cb"), chan("Y1")
            a = _ycc_to_rgb(y0, cb, cr)
            b = _ycc_to_rgb(y1, cb, cr)
            rgb = np.empty((2 * nsync, m.width, 3))
            rgb[0::2] = a
            rgb[1::2] = b
        elif m.kind == "robot72":
            rgb = _ycc_to_rgb(chan("Y"), chan("Cb"), chan("Cr"))
        else:  # robot36
            y = chan("Y")
            c = chan("C")
            sep = chan("SEP", 1)[:, 0]
            is_cb = sep > 127
            par = (is_cb ^ (ks % 2 == 1)).astype(int)
            p0 = 1 if par.sum() * 2 > len(par) else 0
            is_cb = ((ks % 2) ^ p0).astype(bool)        # parity-consistent
            cr = np.empty_like(c)
            cb = np.empty_like(c)
            for i in range(nsync):
                j = i ^ 1
                if j >= nsync:
                    j = i - 1
                if j < 0:
                    other = np.full(m.width, 128.0)
                else:
                    other = c[j]
                if is_cb[i]:
                    cb[i], cr[i] = c[i], other
                else:
                    cr[i], cb[i] = c[i], other
            rgb = _ycc_to_rgb(y, cb, cr)
        rows = rgb.shape[0]
        img[:rows] = np.clip(rgb, 0, 255).astype(np.uint8)
        return img


# ---------------------------------------------------------------------------
# Public watcher
# ---------------------------------------------------------------------------

class SSTVWatcher:
    """Continuous SSTV watcher for one audio channel (8 kHz int16)."""

    HIST_MS = 1500          # history kept for VIS search
    MIN_LINES = 8           # discard partial images with fewer rows

    def __init__(self, out_dir, label=""):
        self.out_dir = out_dir
        self.label = label
        os.makedirs(out_dir, exist_ok=True)
        self._zi = np.zeros(_NTAPS - 1, np.complex128)
        self._ylast = 0j
        self._n = 0                                   # samples of p produced
        self._hp = np.zeros(0, np.complex64)          # p history
        self._hp_base = 0
        self._he = np.zeros(0, np.float32)            # |y|^2 history
        self._hy = np.zeros(0, np.complex64)          # analytic signal history
        self._mse = np.zeros(0, np.float64)           # 1 ms bins of |y|^2
        self._msc = np.zeros(0, np.complex128)        # 1 ms bins of p
        self._ms_base = 0                             # ms index of _msc[0]
        self._next_T = 0                              # next VIS candidate (ms)
        self._anchor = (0, time.time())
        self._img: _Image | None = None
        self.vis_seen = []                            # debug: (ms, code, mode or None)

    # -- time mapping -----------------------------------------------------
    def _ts_of(self, n):
        n0, t0 = self._anchor
        return t0 + (n - n0) / FS

    # -- main entry -------------------------------------------------------
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
        x = x.astype(np.float64).ravel() * (1.0 / 32768.0)
        y, self._zi = lfilter(_BP, 1.0, x, zi=self._zi)
        yprev = np.concatenate(([self._ylast], y[:-1]))
        self._ylast = y[-1]
        p = (y * np.conj(yprev)).astype(np.complex64)
        e = (y.real ** 2 + y.imag ** 2).astype(np.float32)
        n0 = self._n
        self._n += len(p)

        # p history
        self._hp = np.concatenate((self._hp, p))
        self._he = np.concatenate((self._he, e))
        self._hy = np.concatenate((self._hy, y.astype(np.complex64)))
        keep = int(self.HIST_MS * _MS)
        if len(self._hp) > keep:
            cut = len(self._hp) - keep
            cut -= cut % _BIN
            self._hp = self._hp[cut:]
            self._he = self._he[cut:]
            self._hy = self._hy[cut:]
            self._hp_base += cut

        # 1 ms bins
        ms_have = self._ms_base + len(self._msc)
        ms_avail = self._n // _BIN
        if ms_avail > ms_have:
            a = ms_have * _BIN - self._hp_base
            seg = self._hp[a:a + (ms_avail - ms_have) * _BIN].astype(np.complex128)
            self._msc = np.concatenate((self._msc, seg.reshape(-1, _BIN).sum(axis=1)))
            sege = self._he[a:a + (ms_avail - ms_have) * _BIN].astype(np.float64)
            self._mse = np.concatenate((self._mse, sege.reshape(-1, _BIN).sum(axis=1)))
            if len(self._msc) > self.HIST_MS:
                d = len(self._msc) - self.HIST_MS
                self._msc = self._msc[d:]
                self._mse = self._mse[d:]
                self._ms_base += d

        # image in progress: append data
        if self._img is not None:
            overflow = self._img.append(y.astype(np.complex64))
            if self._img.process() or overflow:
                r = self._finish()
                if r:
                    out.append(r)

        vis = self._search_vis()
        if vis is not None:
            T, mode, off = vis
            if self._img is not None:
                self._img.lost = True
                r = self._finish()
                if r:
                    out.append(r)
            self._start_image(T, mode, off)
            if self._img.process():
                r = self._finish()
                if r:
                    out.append(r)
        return out

    def progress(self):
        im = self._img
        if im is None:
            return None
        return {"mode": im.mode.name, "lines": int(im.rows_done()), "total_lines": im.mode.height,
                "ts": float(im.ts), "offset_hz": round(float(im.offset), 1)}

    def preview(self):
        """RGB uint8 array of the image being received (or None)."""
        if self._img is None:
            return None
        return self._img.render()

    # -- VIS ----------------------------------------------------------------
    def _search_vis(self):
        c = self._msc
        base = self._ms_base
        nms = len(c)
        if nms < 920:
            return None
        # candidate T (ms, local index) = start of the start bit;
        # leader [T-285, T-5), start bit .. stop bit [T, T+300)
        tlo = max(self._next_T - base, 615)
        thi = nms - 300
        if thi <= tlo:
            return None
        self._next_T = base + thi
        f1 = _ang_hz(c[tlo - 615:thi])
        fcs = np.concatenate(([0], np.cumsum(f1)))
        f2cs = np.concatenate(([0], np.cumsum(f1 * f1)))

        def stats(l1, n):
            l0 = l1 - n
            m = (fcs[l1] - fcs[l0]) / n
            return m, np.sqrt(np.maximum((f2cs[l1] - f2cs[l0]) / n - m * m, 0))
        # cheap gate (vectorised): two steady leaders near 1900 Hz
        L1 = np.arange(tlo, thi) - 5 - (tlo - 615)
        fm, fsd = stats(L1, 280)
        fm1, fsd1 = stats(L1 - 310, 270)               # first leader [T-600, T-330)
        ok = ((fsd < 170) & (np.abs(fm - 1900) < 300) & (fsd1 < 170) &
              (np.abs(fm1 - fm) < 80))
        if not ok.any():
            return None
        cs = np.concatenate(([0], np.cumsum(c)))
        es = np.concatenate(([0], np.cumsum(self._mse)))
        T = np.arange(tlo, thi)[ok]
        # noise-bias-corrected leader frequency and start bit
        npl = 280 * _BIN
        lead = cs[T - 5] - cs[T - 285]
        lene = es[T - 5] - es[T - 285]
        noise = np.array([_noise_solve(lp, le, npl) for lp, le in zip(lead, lene)])
        nb = noise * _R1 * _BIN                      # bias per 1 ms bin
        fl = _ang_hz(lead - nb * 280)
        sb = _ang_hz(cs[T + 27] - cs[T + 3] - nb * 24)
        ok2 = (np.abs(fl - 1900) < 260) & (np.abs(sb - (fl - 700)) < 200)
        if not ok2.any():
            return None
        cand, cnb = T[ok2], nb[ok2]
        if cand[0] + 15 >= thi:
            self._next_T = base + cand[0]       # wait for the whole edge cluster
            return None
        # pick best edge position in the first cluster
        sel = cand < cand[0] + 15
        best, bestc, bnb = None, None, 0
        for t, b1 in zip(cand[sel], cnb[sel]):
            flt = _ang_hz(cs[t - 5] - cs[t - 285] - b1 * 280)
            fa = _ang_hz(c[t - 20:t] - b1)
            fb = _ang_hz(c[t:t + 20] - b1)
            e = (np.minimum(np.abs(fa - flt), 400).sum() +
                 np.minimum(np.abs(fb - (flt - 700)), 400).sum())
            if bestc is None or e < bestc:
                best, bestc, bnb = t, e, b1
        t = best
        flt = _ang_hz(cs[t - 5] - cs[t - 285] - bnb * 280)
        # bits: coherent tone correlators at the offset-corrected frequencies
        off = flt - 1900.0
        tones = np.array([1100.0, 1200.0, 1300.0]) + off
        nwin = 22 * _BIN
        k = np.arange(nwin)
        ref = np.exp(-2j * np.pi * tones[:, None] * k[None, :] / FS)
        bits = []
        good = True
        for i in range(10):
            a = (base + t + 30 * i + 4) * _BIN - self._hp_base
            seg = self._hy[a:a + nwin]
            if len(seg) < nwin:
                good = False
                break
            e1, e12, e0 = np.abs(ref @ seg) ** 2 / (nwin * float(np.sum(np.abs(seg) ** 2)) + 1e-30)
            if i in (0, 9):
                good &= (e12 > 0.5 * max(e1, e0)) and e12 > 0.25
            else:
                good &= (max(e1, e0) > 1.5 * min(e1, e0)) and max(e1, e0) > 0.25
                bits.append(1 if e1 > e0 else 0)
        code = sum(b << i for i, b in enumerate(bits[:7]))
        parity_ok = (sum(bits) % 2) == 0
        mode = MODES.get(code) if (good and parity_ok) else None
        self.vis_seen.append((int(base + t), int(code), mode.name if mode else None,
                              round(float(flt) - 1900, 1)))
        del self.vis_seen[:-50]
        if mode is None:
            # skip past this candidate but allow a later one
            self._next_T = base + t + 30
            return None
        self._next_T = base + t + 600
        return base + t, mode, float(flt - 1900.0)

    def _start_image(self, Tms, mode, off):
        # leader [T-285, T-5) ms: noise power + bias-free tuning offset
        i0 = Tms - 285 - self._ms_base
        i1 = Tms - 5 - self._ms_base
        noise = 0.0
        if i0 >= 0:
            psum = complex(self._msc[i0:i1].sum())
            esum = float(self._mse[i0:i1].sum())
            noise = _noise_solve(psum, esum, (i1 - i0) * _BIN)
            off = float(_ang_hz(psum - noise * _R1 * (i1 - i0) * _BIN) - 1900.0)
        T = Tms * _BIN                      # sample index of start-bit start
        t_img = T + int(round(300 * _MS))
        start = T
        im = _Image(mode, t_img, start, off, self._ts_of(T - int(610 * _MS)), noise)
        a = start - self._hp_base
        if a < 0:
            a = 0
            im.start_n = self._hp_base
        im.append(self._hy[a:])
        self._img = im

    def _finish(self):
        im = self._img
        self._img = None
        if im is None:
            return None
        if im.lost:
            nsync = im.last_good + 1
        else:
            nsync = im.k
        lines = nsync * im.mode.rows_per_sync
        if lines < self.MIN_LINES:
            return None
        rgb = im.render(nsync)
        im.ts = float(im.ts)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(im.ts))
        lab = ("%s_" % self.label) if self.label else ""
        fname = "sstv_%s%s_%s.png" % (lab, stamp, im.mode.name.replace(" ", ""))
        path = os.path.join(self.out_dir, fname)
        write_png(path, rgb)
        partial = bool(im.lost or lines < im.mode.height)
        return {"path": path, "mode": im.mode.name, "ts": float(im.ts),
                "offset_hz": round(float(im.offset), 1), "lines": int(lines), "partial": partial}


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
    pcm = np.concatenate([pcm, np.zeros(3 * FS, np.int16)])   # let the last image complete
    w = SSTVWatcher(argv[2] if len(argv) > 2 else ".", "wav")
    t0 = time.time()
    for i in range(0, len(pcm), 800):
        for r in w.feed(pcm[i:i + 800], t0 + i / FS):
            print(r)
    if w._img is not None:                    # flush an image still in progress
        w._img.lost = True
        w._img.last_good = max(w._img.last_good, w._img.k - 1)
        r = w._finish()
        if r:
            print(r)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv))
