"""
Streaming PSK31 / PSK63 / RTTY detector + decoder for 8 kHz mono USB audio.
numpy + scipy only, self-contained.

    w = DigiWatcher(label="14070")
    for block, ts in audio:                 # int16 (or float), 8000 Hz, any block length
        for ev in w.feed(block, ts):        # text chunks (per word / ~2-6 s / end of over)
            print(ev)                       # {"mode", "audio_hz", "ts", "text", "snr_db"}
        w.active()                          # [{"mode", "audio_hz", "snr_db", "since", "state"}]

    python3 decoder.py recording.wav        # offline decode of a file (8 kHz mono)

Signal path
  * Scan (every 1 s): Welch PSD of the last 2 s (2048-pt Hann, 3.9 Hz bins).
    Noise floor = median over 200..2900 Hz.  Local maxima >= SCAN_THR_DB above
    the floor become candidates; peaks closer than 30 Hz are merged (a PSK idle
    signal shows two tones +-baud/2) into a power-weighted centroid.  Two
    candidates 170 +- 15 Hz apart of similar level form an RTTY candidate.
    Everything else is a PSK candidate (31 or 63 is decided by decoding).
  * Each tracker: complex NCO mix to its centre, FIR low-pass, decimate to
    1000 Hz.  New trackers are primed with the last 2.5 s of audio, so the
    preamble and first characters are not lost.
  * PSK: per mode (31.25 / 62.5 Bd) a residual mixer (fine AFC), a Hann-2T
    matched filter, symbol timing from the per-phase energy histogram (energy
    peaks at symbol centres), differential detection, carrier-rotation
    estimate from the average of d^2 (d = s_k conj s_{k-1}), varicode.
    Quality per symbol m = cos(2(dphi - phi_est)): ~1 for clean BPSK, ~0 for
    noise.  While probing both modes run; the one that yields clean characters
    wins (a PSK31 signal in the PSK63 demod never produces a '00' delimiter,
    a PSK63 signal in the PSK31 demod gives low quality).
  * RTTY: mark/space mixers (+-85 Hz), boxcar filters matched to the baud,
    discriminator D = (|m|-|s|)/(|m|+|s|), async UART (falling edge -> sample
    bit centres, check start and 1-bit stop), ITA2 with LTRS/FIGS and USOS.
    While probing 45.45/50/75 Bd x both polarities run; best stop-bit
    validity wins.  AFC from the phase rotation of the mark/space filter
    outputs.
  * Text is emitted per chunk only when the chunk looks valid: mean symbol
    quality / discriminator contrast, stop-bit validity (RTTY), varicode
    validity (PSK) and printable ratio.  Failed candidates are black-listed
    for a while so voice or carriers do not churn CPU.
"""
from __future__ import annotations

import cmath
import math
import sys
import time

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import firwin, lfilter

FS = 8000
DEC = 8                    # tracker decimation -> 1000 Hz
FSD = FS // DEC
BAND_LO, BAND_HI = 200.0, 2900.0

SCAN_EVERY_S = 1.0
SCAN_WIN_S = 2.0
NFFT = 2048
SCAN_THR_DB = 6.0          # candidate peak threshold over noise floor (smoothed bins)
KEEP_THR_DB = 3.0          # tracker considered present while its peak is above this
HIST_S = 3.5               # audio replayed into a new tracker
EARLY_FAIL_S = 5.0         # probes with no sign of a valid signal are dropped after this
MAX_TRACKERS = 8
DEAD_SCANS = 4             # tracker removed after this many scans without signal
PSK_PROBE_MIN_S = 3.0
PSK_PROBE_MAX_S = 15.0
RTTY_PROBE_MIN_S = 3.0
RTTY_PROBE_MAX_S = 12.0
BLACKLIST_S = 30.0

# chunk acceptance
PSK_MIN_Q = 0.5            # mean symbol quality of a chunk
PSK_CHAR_Q = 0.35          # characters below this quality are dropped
PSK_EDGE_Q = 0.65          # leading/trailing characters below this are trimmed
PSK_MAX_INVALID = 0.15
RTTY_MIN_CONF = 0.45
RTTY_MIN_STOP = 0.8
MIN_PRINTABLE = 0.9
AMP_MIN = 0.4              # chars weaker than this x typical signal amplitude are noise

CHUNK_MIN_S = 2.0
CHUNK_MAX_S = 6.0
CHUNK_IDLE_S = 1.5

# ---------------------------------------------------------------------------
# Code tables
# ---------------------------------------------------------------------------

# G3PLX varicode, ASCII 0..127 (bits as transmitted, '0' = phase reversal)
_VARICODE_STR = """
1010101011 1011011011 1011101101 1101110111 1011101011 1101011111 1011101111 1011111101
1011111111 11101111 11101 1101101111 1011011101 11111 1101110101 1110101011 1011110111
1011110101 1110101101 1110101111 1101011011 1101101011 1101101101 1101010111 1101111011
1101111101 1110110111 1101010101 1101011101 1110111011 1011111011 1101111111 1 111111111
101011111 111110101 111011011 1011010101 1010111011 101111111 11111011 11110111 101101111
111011111 1110101 110101 1010111 110101111 10110111 10111101 11101101 11111111 101110111
101011011 101101011 110101101 110101011 110110111 11110101 110111101 111101101 1010101 111010111
1010101111 1010111101 1111101 11101011 10101101 10110101 1110111 11011011 11111101 101010101
1111111 111111101 101111101 11010111 10111011 11011101 10101011 11010101 111011101 10101111
1101111 1101101 101010111 110110101 101011101 101110101 101111011 1010101101 111110111 111101111
111111011 1010111111 101101101 1011011111 1011 1011111 101111 101101 11 111101 1011011 101011
1101 111101011 10111111 11011 111011 1111 111 111111 110111111 10101 10111 101 110111 1111011
1101011 11011111 1011101 111010101 1010110111 110111011 1010110101 1011010111 1110110101
"""
VARICODE = _VARICODE_STR.split()
assert len(VARICODE) == 128
VARI_DEC = {c: chr(i) for i, c in enumerate(VARICODE)}

ITA2_LTRS = ["\0", "E", "\n", "A", " ", "S", "I", "U", "\r", "D", "R", "J", "N", "F", "C", "K",
             "T", "Z", "L", "W", "H", "Y", "P", "Q", "O", "B", "G", None, "M", "X", "V", None]
ITA2_FIGS = ["\0", "3", "\n", "-", " ", "\a", "8", "7", "\r", "$", "4", "'", ",", "!", ":", "(",
             "5", '"', ")", "2", "#", "6", "0", "1", "9", "?", "&", None, ".", "/", ";", None]
ITA2_FIGS_CODE = 0x1B
ITA2_LTRS_CODE = 0x1F

_PRINTABLE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n"
                 ".,?/-=+:;'\"()!@#*&%$<>_")


def _printable_ratio(chars):
    if not chars:
        return 0.0
    return sum(1 for c in chars if c in _PRINTABLE) / len(chars)


def _clean_text(chars):
    out = []
    for c in chars:
        if c == "\r" or c == "\0":
            continue
        if c == "\n" or c == "\t":
            out.append(c if c == "\n" else " ")
        elif " " <= c <= "~":
            out.append(c)
    s = "".join(out)
    while "\n\n\n" in s:
        s = s.replace("\n\n\n", "\n\n")
    return s


# ---------------------------------------------------------------------------
# Front end: mix to centre, low-pass, decimate to 1000 Hz
# ---------------------------------------------------------------------------

class FrontEnd:
    def __init__(self, f0, cutoff=160.0, ntaps=63):
        self.f0 = float(f0)
        self.phase = 0.0
        self.taps = firwin(ntaps, cutoff, fs=FS).astype(np.float64)
        self.zi = np.zeros(ntaps - 1, dtype=np.complex128)
        self.dec_off = 0            # index of the next kept sample within the next block

    def process(self, x):
        n = len(x)
        if n == 0:
            return np.zeros(0, np.complex128)
        w = 2 * math.pi * self.f0 / FS
        ph = self.phase + w * np.arange(n)
        y = x * np.exp(-1j * ph)
        self.phase = float((self.phase + w * n) % (2 * math.pi))
        y, self.zi = lfilter(self.taps, 1.0, y, zi=self.zi)
        out = y[self.dec_off::DEC]
        self.dec_off = (self.dec_off - n) % DEC
        return out


# ---------------------------------------------------------------------------
# PSK demodulator (one baud rate)
# ---------------------------------------------------------------------------

class PskDemod:
    def __init__(self, baud):
        self.baud = baud
        self.mode = "PSK31" if baud < 40 else "PSK63"
        self.sps = int(round(FSD / baud))         # 32 or 16
        L = 2 * self.sps
        t = np.arange(L) - (L - 1) / 2.0
        mf = 0.5 * (1 + np.cos(np.pi * t / self.sps))
        self.mf = mf / mf.sum()
        self.zi = np.zeros(L - 1, np.complex128)
        self.delta = 0.0                            # residual frequency (Hz)
        self.rphase = 0.0
        self.energy = np.zeros(self.sps)
        self.buf = np.zeros(0, np.complex128)
        self.buf_abs = None
        self.next_sample = None
        self.prev = None
        self.afc = 0j
        self.bits = ""
        self.overflow = False
        self.carrier = False
        self.sym_m = []                             # qualities of symbols of current char
        self.sym_a = []                             # amplitudes of symbols of current char
        self.tokens = []                            # (abs_1k, char or None, q)
        self.q_avg = 0.0                            # slow symbol-quality average
        self.nsym = 0

    # coarse acquisition from the squared matched-filter output
    def acquire(self, z):
        if len(z) < 4 * self.sps:
            return
        h = np.hanning(self.sps + 2)[1:-1]          # wider than the matched filter:
        y = lfilter(h / h.sum(), 1.0, z)[2 * self.sps:]   # keeps both idle tones
        y2 = y * y
        nf = 8192
        Y = np.abs(np.fft.fft(y2 * np.hanning(len(y2)), nf))
        freqs = np.fft.fftfreq(nf, 1.0 / FSD)
        lim = self.baud * 1.1               # |2 delta| <= ~baud
        sel = np.flatnonzero(np.abs(freqs) <= lim)
        k = sel[np.argmax(Y[sel])]
        a, b, c = Y[(k - 1) % nf], Y[k], Y[(k + 1) % nf]
        den = a - 2 * b + c
        frac = 0.5 * (a - c) / den if den != 0 else 0.0
        f2 = (k + frac) * FSD / nf
        if f2 > FSD / 2:
            f2 -= FSD
        self.delta = f2 / 2.0

    def process(self, z, abs0):
        n = len(z)
        if n == 0:
            return
        w = 2 * math.pi * self.delta / FSD
        ph = self.rphase + w * np.arange(n)
        zz = z * np.exp(-1j * ph)
        self.rphase = float((self.rphase + w * n) % (2 * math.pi))
        y, self.zi = lfilter(self.mf, 1.0, zz, zi=self.zi)
        if self.buf_abs is None:
            self.buf_abs = abs0
        # timing energy histogram
        sps = self.sps
        idx = (abs0 + np.arange(n)) % sps
        self.energy *= math.exp(-n / (sps * 40.0))
        self.energy += np.bincount(idx, weights=(y.real ** 2 + y.imag ** 2), minlength=sps)
        self.buf = np.concatenate((self.buf, y))
        end = self.buf_abs + len(self.buf)
        c = np.sum(self.energy * np.exp(2j * np.pi * np.arange(sps) / sps))
        target = (np.angle(c) / (2 * np.pi) * sps) % sps
        if self.next_sample is None:
            if end - self.buf_abs < 8 * sps:
                return
            base = self.buf_abs + 2 * sps
            k = int(round(target - (base % sps))) % sps
            self.next_sample = base + k
        while self.next_sample < end:
            s = complex(self.buf[self.next_sample - self.buf_abs])
            self._symbol(s, self.next_sample)
            cur = self.next_sample % sps
            err = ((target - cur + sps / 2.0) % sps) - sps / 2.0
            step = sps + max(-2, min(2, int(round(err * 0.3))))
            self.next_sample += step
        keep_from = self.next_sample - 2 * sps
        if keep_from > self.buf_abs:
            self.buf = self.buf[keep_from - self.buf_abs:]
            self.buf_abs = keep_from

    def _symbol(self, s, at):
        if self.prev is None:
            self.prev = s
            return
        d = s * self.prev.conjugate()
        self.prev = s
        ad = abs(d)
        if ad <= 1e-20:
            m = 0.0
            bit = 1
        else:
            d2 = d * d / ad
            pa = abs(self.afc)
            # quality against the *previous* rotation estimate (unbiased for noise)
            m_prev = (d2 * self.afc.conjugate()).real / (ad * pa) if pa > 0 else 0.0
            self.afc = 0.9 * self.afc + 0.1 * d2
            aa = abs(self.afc)
            if aa > 0:
                rot = self.afc / aa                     # e^{j 2 phi_est}
                m = m_prev
                phi = cmath.phase(rot) / 2.0
                bit = 1 if (d * cmath.exp(-1j * phi)).real > 0 else 0
                # fine AFC: residual rotation per symbol -> Hz
                self.delta += 0.03 * phi * self.baud / (2 * math.pi)
            else:
                m = 0.0
                bit = 1 if d.real > 0 else 0
        self.nsym += 1
        self.q_avg = 0.97 * self.q_avg + 0.03 * m
        self.sym_m.append(m)
        self.sym_a.append(abs(s))
        if bit == 0:
            if self.bits == "":                     # idle (run of reversals)
                self.sym_m = []; self.sym_a = []
                return
            if self.bits.endswith("0"):             # '00' delimiter
                code = self.bits[:-1]
                q = sum(self.sym_m) / len(self.sym_m) if self.sym_m else 0.0
                amp = sum(self.sym_a) / len(self.sym_a) if self.sym_a else 0.0
                if self.overflow:
                    self.tokens.append((at, None, q, amp))
                elif not self.carrier:
                    self.tokens.append((at, VARI_DEC.get(code), q, amp))
                self.bits = ""
                self.overflow = False
                self.carrier = False
                self.sym_m = []; self.sym_a = []
                return
            self.bits += "0"
        else:
            self.bits += "1"
        if len(self.bits) >= 12 or (len(self.bits) == 11 and self.bits[-1] == "1"):
            if "0" in self.bits.rstrip("0") or self.overflow:
                self.overflow = True               # too long: not a varicode
            else:
                self.carrier = True                # unmodulated carrier
                self.sym_m = []; self.sym_a = []
            self.bits = self.bits[-2:]
            if len(self.sym_m) > 64:
                self.sym_m = self.sym_m[-64:]
                self.sym_a = self.sym_a[-64:]


# ---------------------------------------------------------------------------
# RTTY demodulator
# ---------------------------------------------------------------------------

class Uart:
    def __init__(self, baud, pol):
        self.baud = baud
        self.pol = pol
        self.spb = FSD / baud
        self.ptr = None             # absolute index to search from
        self.figs = False
        self.tokens = []            # (abs, char or None(shift), conf, stop_ok)
        self.nframes = 0
        self.nstop = 0

    @property
    def mode(self):
        return {45: "RTTY45", 50: "RTTY50", 75: "RTTY75"}[int(self.baud)]

    def run(self, D, A, d_abs):
        """D: discriminator samples (mark positive, pol applied), D[0] at abs d_abs."""
        n = len(D)
        end = d_abs + n
        if self.ptr is None or self.ptr < d_abs + 1:
            self.ptr = d_abs + 1
        spb = self.spb
        frame_len = int(math.ceil(6.6 * spb)) + 1
        neg = D < 0
        while True:
            i0 = self.ptr - d_abs
            if i0 >= n:
                break
            # falling edge: D[i-1] >= 0 and D[i] < 0
            sl = np.flatnonzero(neg[i0:] & ~neg[i0 - 1:n - 1])
            if len(sl) == 0:
                self.ptr = end
                break
            i = i0 + int(sl[0])
            if i + frame_len >= n:
                self.ptr = d_abs + i
                break
            pos = i + (np.arange(7) + 0.5) * spb
            pi = np.floor(pos).astype(int)
            fr = pos - pi
            vals = D[pi] * (1 - fr) + D[pi + 1] * fr
            amp = float(np.mean(A[pi]))
            if vals[0] >= 0:                         # glitch, not a start bit
                self.ptr = d_abs + i + 1
                continue
            code = 0
            for k in range(5):
                if vals[1 + k] > 0:
                    code |= 1 << k
            stop_ok = vals[6] > 0
            conf = float(np.mean(np.abs(vals)))
            self.nframes += 1
            self.nstop += stop_ok
            at = d_abs + i
            ch = None
            if code == ITA2_FIGS_CODE:
                self.figs = True
            elif code == ITA2_LTRS_CODE:
                self.figs = False
            else:
                ch = (ITA2_FIGS if self.figs else ITA2_LTRS)[code]
                if ch == " ":
                    self.figs = False                # unshift on space
            self.tokens.append((at, ch, conf, bool(stop_ok), amp))
            self.ptr = d_abs + i + int(6.5 * spb) if stop_ok else d_abs + i + int(5.5 * spb)


class RttyDemod:
    SHIFT = 170.0

    def __init__(self, bauds=(45.45, 50.0, 75.0), pols=(1, -1)):
        self.uarts = [Uart(b, p) for b in bauds for p in pols]
        self.filters = {}
        for b in bauds:
            L = int(round(FSD / b))
            self.filters[b] = (np.ones(L) / L, np.zeros(L - 1, np.complex128), np.zeros(L - 1, np.complex128))
        self.hist = {b: (np.zeros(0), np.zeros(0), None) for b in bauds}
        self.afc = 0j
        self.ferr = 0.0
        self.contrast = 0.0

    def process(self, z, abs0):
        n = len(z)
        if n == 0:
            return
        t = (abs0 + np.arange(n)) / FSD
        h = self.SHIFT / 2
        rm = z * np.exp(-2j * np.pi * h * t)
        rs = z * np.exp(2j * np.pi * h * t)
        ref = None
        for b, (taps, zm, zs) in list(self.filters.items()):
            m, zm = lfilter(taps, 1.0, rm, zi=zm)
            s, zs = lfilter(taps, 1.0, rs, zi=zs)
            self.filters[b] = (taps, zm, zs)
            am, as_ = np.abs(m), np.abs(s)
            D = (am - as_) / (am + as_ + 1e-12)
            if ref is None:
                ref = (m, s, D)
            old, olda, oabs = self.hist[b]
            if oabs is None:
                oabs = abs0
            Dh = np.concatenate((old, D))
            Ah = np.concatenate((olda, am + as_))
            for u in self.uarts:
                if u.baud == b:
                    u.run(Dh * u.pol, Ah, oabs)
            keep = int(8 * FSD / b)
            if len(Dh) > keep:
                oabs += len(Dh) - keep
                Dh = Dh[-keep:]
                Ah = Ah[-keep:]
            self.hist[b] = (Dh, Ah, oabs)
        m, s, D = ref
        w = np.sum(m[1:] * np.conj(m[:-1])) + np.sum(s[1:] * np.conj(s[:-1]))
        self.afc = 0.7 * self.afc + 0.3 * w
        self.ferr = float(np.angle(self.afc) * FSD / (2 * np.pi))
        self.contrast = 0.9 * self.contrast + 0.1 * float(np.mean(np.abs(D)))

    def keep_only(self, uart):
        self.uarts = [uart]
        b = uart.baud
        self.filters = {b: self.filters[b]}
        self.hist = {b: self.hist[b]}


# ---------------------------------------------------------------------------
# Trackers
# ---------------------------------------------------------------------------

class _Tracker:
    kind = ""

    def __init__(self, f0, since, snr_db):
        self.f0 = f0
        self.since = since
        self.snr_db = snr_db
        self.missing = 0
        self.state = "probe"
        self.pending = []           # accepted-candidate tokens of the current chunk
        self.emitted = 0
        self.dead = False
        self.failed = False

    def flush_due(self, now_abs, force=False):
        if not self.pending:
            return False
        first = self.pending[0][0]
        last = self.pending[-1][0]
        if force:
            return True
        if now_abs - last >= CHUNK_IDLE_S * FSD:
            return True
        if now_abs - first >= CHUNK_MAX_S * FSD:
            return True
        if now_abs - first >= CHUNK_MIN_S * FSD and self.pending[-1][1] in (" ", "\n"):
            return True
        return False


class PskTracker(_Tracker):
    kind = "PSK"

    def __init__(self, f0, since, snr_db):
        super().__init__(f0, since, snr_db)
        self.fe = FrontEnd(f0, cutoff=130.0)
        self.demods = [PskDemod(31.25), PskDemod(62.5)]
        self.ref_amp = None
        self.abs1k = 0
        self.probe_s = 0.0

    @property
    def mode(self):
        if len(self.demods) == 1:
            return self.demods[0].mode
        return self._best()[0].mode

    @property
    def audio_hz(self):
        d = self.demods[0] if len(self.demods) == 1 else self._best()[0]
        return self.fe.f0 + d.delta

    @staticmethod
    def _score(d):
        good = [t for t in d.tokens if t[1] is not None and t[2] > 0.6 and t[1] in _PRINTABLE]
        bad = [t for t in d.tokens if t[1] is None or t[2] <= 0.3]
        return len(good) - 2 * len(bad), len(good), len(d.tokens)

    def _best(self):
        sc = [(self._score(d), d) for d in self.demods]
        sc.sort(key=lambda x: x[0][0], reverse=True)
        return sc[0][1], sc[0][0]

    def prime(self, x, abs8):
        z = self.fe.process(x)
        self.abs1k = abs8 // DEC
        for d in self.demods:
            d.acquire(z)
        self._run(z)

    def feed(self, x):
        self._run(self.fe.process(x))

    def _run(self, z):
        for d in self.demods:
            d.process(z, self.abs1k)
        self.abs1k += len(z)
        self.probe_s += len(z) / FSD
        if self.state == "probe":
            best, (score, good, tot) = self._best()
            if self.probe_s >= PSK_PROBE_MIN_S and good >= 6 and good >= 0.6 * tot:
                self.demods = [best]
                self.state = "locked"
            elif self.probe_s >= PSK_PROBE_MAX_S or (
                    self.probe_s >= EARLY_FAIL_S and good < 2
                    and max(d.q_avg for d in self.demods) < 0.6):
                self.failed = True
        # recentre the front end if the residual got big
        if len(self.demods) == 1:
            d = self.demods[0]
            if abs(d.delta) > 15.0:
                self.fe.f0 += d.delta
                d.delta = 0.0

    def take_tokens(self):
        """Move decoded tokens into the pending chunk (only once locked)."""
        if self.state != "locked":
            return
        d = self.demods[0]
        for t in d.tokens:
            if t[1] is not None and t[2] > 0.75:
                self.ref_amp = t[3] if self.ref_amp is None else 0.95 * self.ref_amp + 0.05 * t[3]
        if self.ref_amp is None and d.tokens:
            self.ref_amp = float(np.median([t[3] for t in d.tokens if t[1] is not None] or [0.0]))
        self.pending.extend(d.tokens)
        d.tokens = []

    def finish_chunk(self):
        toks = self.pending
        self.pending = []
        # trim low-quality characters at the edges (noise before / after the over)
        amin = AMP_MIN * (self.ref_amp or 0.0)
        toks = [t for t in toks if t[3] >= amin]
        good = [i for i, t in enumerate(toks) if t[1] is not None and t[2] >= PSK_EDGE_Q]
        if not good:
            return None
        toks = toks[good[0]:good[-1] + 1]
        n = len(toks)
        q = float(np.mean([t[2] for t in toks]))
        inval = sum(1 for t in toks if t[1] is None) / n
        chars = [t[1] for t in toks if t[1] is not None and (t[2] >= PSK_CHAR_Q or t[1] == " ")]
        pr = _printable_ratio(chars)
        nalnum = sum(1 for c in chars if c.isalnum())
        ok = q >= PSK_MIN_Q and inval <= PSK_MAX_INVALID and pr >= MIN_PRINTABLE and nalnum >= 2
        if not ok:
            return None
        text = _clean_text(chars)
        if not text.strip():
            return None
        return toks[0][0], text


class RttyTracker(_Tracker):
    kind = "RTTY"

    def __init__(self, f0, since, snr_db):
        super().__init__(f0, since, snr_db)
        self.fe = FrontEnd(f0, cutoff=170.0)
        self.dem = RttyDemod()
        self.ref_amp = None
        self.abs1k = 0
        self.probe_s = 0.0

    def _best(self):
        best = None
        for u in self.dem.uarts:
            good = sum(1 for t in u.tokens if t[3] and t[2] > 0.5)
            tot = len(u.tokens)
            sc = good - (tot - good)
            if best is None or sc > best[0]:
                best = (sc, good, tot, u)
        return best

    @property
    def mode(self):
        return self._best()[3].mode

    @property
    def audio_hz(self):
        return self.fe.f0

    def prime(self, x, abs8):
        z = self.fe.process(x)
        self.abs1k = abs8 // DEC
        self._run(z)

    def feed(self, x):
        self._run(self.fe.process(x))

    def _run(self, z):
        self.dem.process(z, self.abs1k)
        self.abs1k += len(z)
        self.probe_s += len(z) / FSD
        if self.dem.contrast > 0.4:
            self.fe.f0 += 0.15 * float(np.clip(self.dem.ferr, -10, 10)) * len(z) / 100.0
        if self.state == "probe":
            sc, good, tot, u = self._best()
            if self.probe_s >= RTTY_PROBE_MIN_S and good >= 8 and good >= 0.75 * tot:
                self.dem.keep_only(u)
                self.state = "locked"
            elif self.probe_s >= RTTY_PROBE_MAX_S or (
                    self.probe_s >= EARLY_FAIL_S and (good < 2 or self.dem.contrast < 0.3)):
                self.failed = True

    def take_tokens(self):
        if self.state != "locked":
            return
        u = self.dem.uarts[0]
        for t in u.tokens:
            if t[3] and t[2] > 0.6:
                self.ref_amp = t[4] if self.ref_amp is None else 0.95 * self.ref_amp + 0.05 * t[4]
        if self.ref_amp is None and u.tokens:
            self.ref_amp = float(np.median([t[4] for t in u.tokens if t[3]] or [0.0]))
        self.pending.extend(u.tokens)
        u.tokens = []

    def finish_chunk(self):
        toks = self.pending
        self.pending = []
        n = len(toks)
        if n == 0:
            return None
        amin = AMP_MIN * (self.ref_amp or 0.0)
        toks = [t for t in toks if t[4] >= amin]
        good = [i for i, t in enumerate(toks) if t[3] and t[2] >= 0.6]
        if not good:
            return None
        toks = toks[good[0]:good[-1] + 1]
        n = len(toks)
        conf = float(np.mean([t[2] for t in toks]))
        stop = sum(1 for t in toks if t[3]) / n
        chars = [t[1] for t in toks if t[1] is not None and t[3] and t[2] >= 0.3]
        pr = _printable_ratio([c for c in chars if c not in "\r\0"])
        nalnum = sum(1 for c in chars if c.isalnum())
        ok = conf >= RTTY_MIN_CONF and stop >= RTTY_MIN_STOP and pr >= MIN_PRINTABLE and nalnum >= 2
        if not ok:
            return None
        text = _clean_text(chars)
        if not text.strip():
            return None
        return toks[0][0], text


# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------

class DigiWatcher:
    def __init__(self, label=""):
        self.label = label
        self.ring = np.zeros(0, np.float64)
        self.ring_abs = 0                    # abs index (8 kHz) of ring[0]
        self.abs = 0                         # abs index of next sample
        self.ts_ref = None                   # (abs, ts) of last block start
        self.next_scan = int(SCAN_EVERY_S * FS)
        self.trackers: list[_Tracker] = []
        self.blacklist = []                  # (kind, freq, until_abs)
        self.prev_cands = []                 # candidates of the previous scan (persistence check)
        self.win = np.hanning(NFFT)
        self.freqs = np.fft.rfftfreq(NFFT, 1.0 / FS)
        self.band = (self.freqs >= BAND_LO) & (self.freqs <= BAND_HI)
        self.cpu_s = 0.0

    # ---- public ---------------------------------------------------------
    def feed(self, pcm, ts):
        t0 = time.process_time()
        arr = np.asarray(pcm)
        x = arr.astype(np.float64)
        if arr.dtype == np.int16:
            x /= 32768.0
        self.ts_ref = (self.abs, float(ts))
        self.abs += len(x)
        self.ring = np.concatenate((self.ring, x))
        maxlen = int(max(SCAN_WIN_S, HIST_S) * FS) + 1600
        if len(self.ring) > maxlen:
            cut = len(self.ring) - maxlen
            self.ring = self.ring[cut:]
            self.ring_abs += cut
        events = []
        for tr in self.trackers:
            tr.feed(x)
        self._resolve()
        if self.abs >= self.next_scan:
            self.next_scan += int(SCAN_EVERY_S * FS)
            self._scan()
        events.extend(self._collect())
        self.cpu_s += time.process_time() - t0
        return events

    def flush(self):
        """Force out pending text (e.g. at end of a file)."""
        ev = []
        for tr in self.trackers:
            ev.extend(self._collect_one(tr, force=True))
        return ev

    def active(self):
        out = []
        for tr in self.trackers:
            out.append({"mode": tr.mode, "audio_hz": round(float(tr.audio_hz), 1),
                        "snr_db": round(float(tr.snr_db), 1), "since": tr.since,
                        "state": tr.state})
        return out

    # ---- internals ------------------------------------------------------
    def _ts_of(self, abs8):
        a, t = self.ts_ref
        return t + (abs8 - a) / FS

    def _collect(self):
        ev = []
        for tr in self.trackers:
            ev.extend(self._collect_one(tr, force=tr.dead))
        alive = []
        for tr in self.trackers:
            if tr.dead or tr.failed:
                if tr.failed:
                    self.blacklist.append((tr.kind, tr.audio_hz, self.abs + int(BLACKLIST_S * FS)))
                continue
            alive.append(tr)
        self.trackers = alive
        return ev

    def _collect_one(self, tr, force=False):
        tr.take_tokens()
        now1k = self.abs // DEC
        ev = []
        if tr.flush_due(now1k, force):
            r = tr.finish_chunk()
            if r is not None:
                at1k, text = r
                tr.emitted += 1
                ev.append({"mode": tr.mode, "audio_hz": round(float(tr.audio_hz), 1),
                           "ts": round(self._ts_of(at1k * DEC), 2), "text": text,
                           "snr_db": round(float(tr.snr_db), 1), "label": self.label})
        return ev

    def _psd(self):
        n = int(SCAN_WIN_S * FS)
        seg = self.ring[-n:]
        if len(seg) < n:
            return None
        hop = NFFT // 2
        nfr = (n - NFFT) // hop + 1
        idx = np.arange(NFFT)[None, :] + hop * np.arange(nfr)[:, None]
        X = np.fft.rfft(seg[idx] * self.win, axis=1)
        P = np.mean(X.real ** 2 + X.imag ** 2, axis=0)
        P *= 2.0 / (FS * np.sum(self.win ** 2))   # one-sided density (power/Hz)
        return P

    def _band_snr(self, P, nf, f, half):
        df = self.freqs[1]
        sel = (self.freqs >= f - half) & (self.freqs <= f + half)
        sig = float(np.sum(np.maximum(P[sel] - nf, 0.0)) * df)
        return 10 * math.log10(max(sig, 1e-30) / (nf * 3000.0))

    def _scan(self):
        P = self._psd()
        if P is None:
            return
        nf = float(np.median(P[self.band]))
        if nf <= 0:
            return
        Ps = np.convolve(P, np.ones(3) / 3, mode="same")
        r = Ps / nf
        f = self.freqs
        df = f[1]
        now = self.abs
        self.blacklist = [b for b in self.blacklist if b[2] > now]

        # update existing trackers
        for tr in self.trackers:
            if tr.kind == "RTTY":
                fs_ = [tr.audio_hz - 85, tr.audio_hz + 85]
                pk = max(float(np.max(r[(f >= x - 20) & (f <= x + 20)])) for x in fs_)
                snr = self._band_snr(P, nf, tr.audio_hz, 130)
            else:
                half = 25 if tr.mode == "PSK31" else 50
                sel = (f >= tr.audio_hz - half) & (f <= tr.audio_hz + half)
                pk = float(np.max(r[sel])) if np.any(sel) else 0.0
                snr = self._band_snr(P, nf, tr.audio_hz, half)
            tr.snr_db = 0.7 * tr.snr_db + 0.3 * snr if tr.snr_db > -90 else snr
            if 10 * math.log10(max(pk, 1e-12)) < KEEP_THR_DB:
                tr.missing += 1
                if tr.missing >= DEAD_SCANS:
                    tr.dead = True
            else:
                tr.missing = 0

        # drop duplicates (two trackers converged on the same signal)
        trs = sorted(self.trackers, key=lambda t: t.since)
        for i, a in enumerate(trs):
            for b in trs[i + 1:]:
                if a.dead or b.dead:
                    continue
                if a.kind != b.kind:
                    continue                 # handled by _resolve() once one of them locks
                lim = 60 if a.kind == "RTTY" else 10
                if abs(a.audio_hz - b.audio_hz) < lim:
                    # keep the older / locked one
                    if b.state == "locked" and a.state != "locked":
                        a.dead = True
                    else:
                        b.dead = True

        # candidate peaks
        thr = 10 ** (SCAN_THR_DB / 10)
        inb = np.flatnonzero(self.band[1:-1]) + 1
        loc = inb[(r[inb] > thr) & (r[inb] >= r[inb - 1]) & (r[inb] >= r[inb + 1])]
        if len(loc) == 0:
            self.prev_cands = []
            return
        # merge peaks closer than 30 Hz into a weighted centroid
        clusters = []
        for k in loc:
            if clusters and f[k] - f[clusters[-1][-1]] < 36.0:
                clusters[-1].append(k)
            else:
                clusters.append([k])
        cands = []
        for cl in clusters:
            lo, hi = f[cl[0]] - 20, f[cl[-1]] + 20
            sel = np.flatnonzero((f >= lo) & (f <= hi))
            wgt = np.maximum(P[sel] - 2 * nf, 0.0)
            if wgt.sum() <= 0:
                continue
            fc = float(np.sum(f[sel] * wgt) / np.sum(wgt))
            pk = float(np.max(r[cl]))
            width = f[cl[-1]] - f[cl[0]]
            cands.append({"f": fc, "pk": pk, "width": width})
        # RTTY: two tones 170 Hz apart, both above threshold, similar level, less power between
        rtty = []
        mx = maximum_filter1d(r, size=2 * int(round(12 / df)) + 1)
        grid = np.arange(BAND_LO + 85, BAND_HI - 85, df / 2)
        M = np.interp(grid + 85, f, mx)
        S = np.interp(grid - 85, f, mx)
        mid = np.interp(grid, f, np.convolve(r, np.ones(13) / 13, mode="same"))
        lo = np.minimum(M, S)
        ok = (lo > thr) & (np.maximum(M, S) < 10 * lo) & (mid < 0.5 * lo)
        score = np.where(ok, lo, 0.0)
        while np.any(score > 0):
            k = int(np.argmax(score))
            fc = float(grid[k])
            if not self._blacklisted("RTTY", fc, 20):
                rtty.append({"f": fc, "pk": float(score[k]), "kind": "RTTY"})
            score[np.abs(grid - fc) < 120] = 0.0
        # PSK candidates on the tones of an RTTY candidate are probed too (two PSK
        # stations ~140-170 Hz apart look like RTTY); the first to lock wins.
        psk = [dict(c, kind="PSK") for c in cands]
        allc = sorted(rtty + psk, key=lambda c: c["pk"], reverse=True)
        # a candidate must have been seen in the previous scan too (rejects voice transients)
        prev, self.prev_cands = self.prev_cands, [(c["kind"], c["f"]) for c in allc]
        allc = [c for c in allc if any(k == c["kind"] and abs(fp - c["f"]) < 15 for k, fp in prev)]
        for c in allc:
            if self._near_tracker(c):
                continue
            if c["kind"] == "PSK" and self._blacklisted("PSK", c["f"], 20):
                continue
            if len(self.trackers) >= MAX_TRACKERS:
                # replace the weakest unlocked tracker if this candidate is clearly stronger
                weak = [t for t in self.trackers if t.state != "locked"]
                if not weak:
                    break
                w = min(weak, key=lambda t: t.snr_db)
                csnr = self._band_snr(P, nf, c["f"], 130 if c["kind"] == "RTTY" else 25)
                if csnr < w.snr_db + 3:
                    continue
                self.trackers.remove(w)
            self._spawn(c, P, nf)

    def _resolve(self):
        """A locked RTTY tracker removes PSK probes on its tones and vice versa."""
        locked = [t for t in self.trackers if t.state == "locked"]
        if not locked:
            return
        for t in self.trackers:
            if t.state == "locked" or t.dead:
                continue
            for l in locked:
                if l.kind != t.kind and abs(l.audio_hz - t.audio_hz) < 130:
                    t.dead = True
                    if t.kind == "RTTY":
                        self.blacklist.append(("RTTY", t.audio_hz, self.abs + int(BLACKLIST_S * FS)))
                    break

    def _blacklisted(self, kind, fc, tol):
        return any(b[0] == kind and abs(b[1] - fc) < tol for b in self.blacklist)

    def _near_tracker(self, c):
        for t in self.trackers:
            if t.dead:
                continue
            dfz = abs(t.audio_hz - c["f"])
            if t.kind == "RTTY" and c["kind"] == "RTTY":
                if dfz < 120:
                    return True
            elif t.kind == "RTTY" or c["kind"] == "RTTY":
                if dfz < 130 and t.state == "locked":
                    return True
            else:
                guard = 25 if (t.state == "locked" and t.mode == "PSK31") else 45
                if dfz < guard:
                    return True
        return False

    def _spawn(self, c, P, nf):
        since = self._ts_of(self.abs)
        if c["kind"] == "RTTY":
            tr = RttyTracker(c["f"], since, self._band_snr(P, nf, c["f"], 130))
        else:
            tr = PskTracker(c["f"], since, self._band_snr(P, nf, c["f"], 25))
        n = int(HIST_S * FS)
        n -= n % DEC
        hist = self.ring[-n:]
        start_abs = self.abs - len(hist)
        # align history start to the decimation grid
        cut = (-start_abs) % DEC
        hist = hist[cut:]
        start_abs += cut
        tr.fe.dec_off = 0
        tr.prime(hist, start_abs)
        self.trackers.append(tr)


# ---------------------------------------------------------------------------
# CLI: offline decode of a WAV file
# ---------------------------------------------------------------------------

def _main(argv):
    import wave
    if len(argv) < 2:
        print(__doc__)
        return 1
    with wave.open(argv[1], "rb") as wf:
        assert wf.getframerate() == FS and wf.getnchannels() == 1 and wf.getsampwidth() == 2
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    w = DigiWatcher(label=argv[1])
    for i in range(0, len(data), 800):
        for ev in w.feed(data[i:i + 800], i / FS):
            print(ev)
    for ev in w.flush():
        print(ev)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
