"""Minimal WEFAX encoder for tests (8 kHz audio, USB, centre 1900 Hz).

encode(img, lpm, ioc, ...) -> float64 audio (amplitude 1.0)
  img: float array (lines x width) 0 = black .. 1 = white
  sequence: [lead silence/noise] start tone (5 s) | phasing (phasing_s) |
            image | stop tone (5 s at 450 Hz) | black (10 s)
  ppm: station clock error; the station's line period measured with our
       clock is nominal * (1 + ppm*1e-6).
  offset_hz: tuning offset added to every frequency.
"""
import numpy as np
from scipy.signal import firwin, lfilter

FS = 8000


def _square(t, f):
    return (np.floor(2 * f * t) % 2 == 0).astype(np.float64)


def luminance(img, lpm, ioc, ppm=0.0, start=True, phasing_s=30.0, stop=True,
              skip_lines=0, phasing_white_pulse=True):
    """Return luminance (0..1) at 8 kHz of the whole transmission plus the
    sample index where image line `skip_lines` begins."""
    T = 60.0 / lpm
    tone_f = 300.0 if ioc == 576 else 675.0
    H, W = img.shape
    segs = []
    if start:
        segs.append(("start", 5.0))
    if phasing_s > 0:
        segs.append(("phasing", round(phasing_s / T) * T))
    nimg = H - skip_lines
    segs.append(("image", nimg * T))
    if stop:
        segs.append(("stop", 5.0))
        segs.append(("black", 10.0))
    total = sum(d for _, d in segs)
    scale = 1.0 + ppm * 1e-6
    n = int(total * scale * FS)
    t = np.arange(n) / FS / scale          # station time
    lum = np.zeros(n)
    t0 = 0.0
    img_n0 = None
    flat = img.astype(np.float64).ravel()
    for name, d in segs:
        m = (t >= t0) & (t < t0 + d)
        tt = t[m] - t0
        if name in ("start", "stop"):
            lum[m] = _square(tt, tone_f if name == "start" else 450.0)
        elif name == "phasing":
            u = (tt / T) % 1.0
            pulse = u < 0.05
            lum[m] = np.where(pulse, 1.0, 0.0) if phasing_white_pulse else np.where(pulse, 0.0, 1.0)
        elif name == "image":
            img_n0 = int(np.argmax(m))
            k = np.floor(tt / T) + skip_lines
            u = (tt / T) % 1.0
            pos = k * W + u * W - 0.5
            lum[m] = np.interp(pos, np.arange(flat.size), flat)
        elif name == "black":
            lum[m] = 0.0
        t0 += d
    return lum, img_n0


def fm(lum, offset_hz=0.0):
    h = firwin(31, 1100.0, fs=FS)              # transmitter video bandwidth
    lum = lfilter(h, 1.0, lum)
    f = 1500.0 + 800.0 * np.clip(lum, 0, 1) + offset_hz
    ph = 2 * np.pi * np.cumsum(f) / FS
    return np.sin(ph)


def encode(img, lpm=120, ioc=576, ppm=0.0, offset_hz=0.0, **kw):
    lum, n0 = luminance(img, lpm, ioc, ppm, **kw)
    return fm(lum, offset_hz), n0
