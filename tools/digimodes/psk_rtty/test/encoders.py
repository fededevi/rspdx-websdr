"""Test signal generators: PSK31/PSK63 (varicode, Hann-shaped BPSK) and RTTY (ITA2 FSK)."""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from decoder import FS, VARICODE, ITA2_LTRS, ITA2_FIGS, ITA2_FIGS_CODE, ITA2_LTRS_CODE  # noqa: E402


def _carrier_phase(n, f0, drift_hz_s):
    t = np.arange(n) / FS
    f = f0 + drift_hz_s * t
    return 2 * np.pi * np.cumsum(f) / FS, f


def psk_bits(text, preamble_syms=40, postamble_syms=24):
    bits = "0" * preamble_syms
    for c in text:
        bits += VARICODE[ord(c) & 0x7F] + "00"
    bits += "0" * 8 + "1" * postamble_syms
    return bits


def psk_signal(text, baud, f0, drift_hz_s=0.0, rng=None):
    """Real passband PSK31/63 burst (unit-ish amplitude) + its instantaneous frequency track."""
    sps = int(round(FS / baud))
    bits = psk_bits(text)
    a = 1.0
    amps = []
    for b in bits:
        if b == "0":
            a = -a
        amps.append(a)
    n = (len(amps) + 1) * sps
    imp = np.zeros(n)
    imp[np.arange(len(amps)) * sps] = amps
    p = 0.5 * (1 - np.cos(2 * np.pi * np.arange(2 * sps) / (2 * sps)))
    bb = np.convolve(imp, p)[:n]
    ph, f = _carrier_phase(n, f0, drift_hz_s)
    ph0 = 0.0 if rng is None else rng.uniform(0, 2 * np.pi)
    return bb * np.cos(ph + ph0), f


def baudot_codes(text):
    codes = [ITA2_LTRS_CODE, ITA2_LTRS_CODE]
    figs = False
    for c in text.upper():
        if c in ITA2_LTRS and c is not None and c not in ("\0",):
            k = ITA2_LTRS.index(c)
            if figs and c not in (" ", "\n", "\r"):
                codes.append(ITA2_LTRS_CODE)
                figs = False
            codes.append(k)
            if c == " ":
                figs = False      # USOS
        elif c in ITA2_FIGS:
            k = ITA2_FIGS.index(c)
            if not figs:
                codes.append(ITA2_FIGS_CODE)
                figs = True
            codes.append(k)
    return codes


def rtty_signal(text, f_center, baud=45.45, shift=170.0, drift_hz_s=0.0, mark_upper=True,
                stop_bits=1.5, preamble_s=1.0):
    """Continuous-phase FSK, keying smoothed with a short raised-cosine filter."""
    spb = FS / baud
    seq = [1.0] * 0
    bits = []           # (value, duration in bits)
    bits.append((1, preamble_s * baud))
    for code in [ITA2_LTRS_CODE] * 4 + baudot_codes(text) + [ITA2_LTRS_CODE] * 3:
        bits.append((0, 1.0))
        for k in range(5):
            bits.append(((code >> k) & 1, 1.0))
        bits.append((1, stop_bits))
    bits.append((1, 0.5 * baud))
    total = sum(d for _, d in bits)
    n = int(math.ceil(total * spb)) + 1
    key = np.ones(n)
    pos = 0.0
    for v, d in bits:
        a, b = int(round(pos * spb)), int(round((pos + d) * spb))
        key[a:b] = 1.0 if v else -1.0
        pos += d
    L = max(3, int(spb * 0.3))
    h = np.hanning(L)
    key = np.convolve(key, h / h.sum(), mode="same")
    sgn = 1.0 if mark_upper else -1.0
    t = np.arange(n) / FS
    f = f_center + drift_hz_s * t + sgn * key * shift / 2
    ph = 2 * np.pi * np.cumsum(f) / FS
    env = np.ones(n)
    r = int(0.01 * FS)
    env[:r] = np.linspace(0, 1, r)
    env[-r:] = np.linspace(1, 0, r)
    return env * np.cos(ph), f_center + drift_hz_s * t


def place(total_len, sig, start):
    out = np.zeros(total_len)
    a = int(start * FS)
    b = min(total_len, a + len(sig))
    out[a:b] = sig[:b - a]
    return out


def mix(signals, snrs_db, total_len, rng, noise_sigma=0.05):
    """signals: list of (sig, start_s). SNR per signal in 3 kHz bandwidth (white noise, fs 8 kHz)."""
    noise = rng.normal(0, noise_sigma, total_len)
    pn3k = noise_sigma ** 2 * 3000.0 / (FS / 2)
    x = noise.copy()
    for (sig, start), snr in zip(signals, snrs_db):
        act = sig[np.abs(sig) > 1e-6]
        ps = float(np.mean(act ** 2))
        g = math.sqrt(pn3k * 10 ** (snr / 10) / ps)
        x += place(total_len, g * sig, start)
    return x


def to_int16(x, rms=0.12):
    """Crude AGC: scale to a target RMS and clip."""
    s = rms / max(1e-9, float(np.sqrt(np.mean(x ** 2))))
    return np.clip(x * s * 32767, -32768, 32767).astype(np.int16)
