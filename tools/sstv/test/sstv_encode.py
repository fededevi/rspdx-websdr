"""Reference SSTV encoder for tests (standard timings, phase-continuous FM).

Written independently of decoder.py's mode table, from the published timings
(Dayton paper / JL Barber).  Generated at 48 kHz, resampled to 8 kHz.
"""
import numpy as np
from scipy.signal import resample_poly

FS_GEN = 48000


def b2f(v):
    return 1500.0 + 800.0 * np.asarray(v, float) / 255.0


def vis_segments(code):
    seg = [(1900, 300), (1200, 10), (1900, 300), (1200, 30)]
    ones = 0
    for i in range(7):
        b = (code >> i) & 1
        ones += b
        seg.append((1100 if b else 1300, 30))
    seg.append((1100 if ones % 2 else 1300, 30))
    seg.append((1200, 30))
    return seg


def _px(seg, values, dur_ms):
    f = b2f(values)
    seg.extend(zip(f, [dur_ms / len(f)] * len(f)))


def _ycc(img):
    r, g, b = [img[..., i].astype(float) for i in range(3)]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.clip(y, 0, 255), np.clip(cb, 0, 255), np.clip(cr, 0, 255)


SPECS = {
    # name: (vis, w, h)
    "Martin M1": (44, 320, 256), "Martin M2": (40, 160, 256),
    "Scottie S1": (60, 320, 256), "Scottie S2": (56, 160, 256), "Scottie DX": (76, 320, 256),
    "Robot 36": (8, 320, 240), "Robot 72": (12, 320, 240),
    "PD90": (99, 320, 256), "PD120": (95, 640, 496), "PD180": (96, 640, 496),
}


def image_segments(mode, img):
    """img: uint8 HxWx3 of the mode's native size."""
    seg = []
    h = img.shape[0]
    if mode.startswith("Martin"):
        scan = 146.432 if mode == "Martin M1" else 73.216
        for l in range(h):
            seg += [(1200, 4.862), (1500, 0.572)]
            for ch in (1, 2, 0):
                _px(seg, img[l, :, ch], scan)
                seg.append((1500, 0.572))
    elif mode.startswith("Scottie"):
        scan = {"Scottie S1": 138.24, "Scottie S2": 88.064, "Scottie DX": 345.6}[mode]
        seg.append((1200, 9.0))                       # starting sync
        for l in range(h):
            seg.append((1500, 1.5))
            _px(seg, img[l, :, 1], scan)
            seg.append((1500, 1.5))
            _px(seg, img[l, :, 2], scan)
            seg += [(1200, 9.0), (1500, 1.5)]
            _px(seg, img[l, :, 0], scan)
    elif mode in ("Robot 36", "Robot 72"):
        y, cb, cr = _ycc(img)
        for l in range(h):
            seg += [(1200, 9.0), (1500, 3.0)]
            if mode == "Robot 36":
                _px(seg, y[l], 88.0)
                if l % 2 == 0:
                    seg += [(1500, 4.5), (1900, 1.5)]
                    _px(seg, (cr[l] + cr[min(l + 1, h - 1)]) / 2, 44.0)
                else:
                    seg += [(2300, 4.5), (1900, 1.5)]
                    _px(seg, (cb[l] + cb[l - 1]) / 2, 44.0)
            else:
                _px(seg, y[l], 138.0)
                seg += [(1500, 4.5), (1900, 1.5)]
                _px(seg, cr[l], 69.0)
                seg += [(2300, 4.5), (1900, 1.5)]
                _px(seg, cb[l], 69.0)
    elif mode.startswith("PD"):
        px = {"PD90": 0.532, "PD120": 0.19, "PD180": 0.286}[mode]
        w = img.shape[1]
        y, cb, cr = _ycc(img)
        for l in range(0, h, 2):
            seg += [(1200, 20.0), (1500, 2.08)]
            _px(seg, y[l], w * px)
            _px(seg, (cr[l] + cr[l + 1]) / 2, w * px)
            _px(seg, (cb[l] + cb[l + 1]) / 2, w * px)
            _px(seg, y[l + 1], w * px)
    else:
        raise ValueError(mode)
    return seg


def synth(segments, fs=FS_GEN):
    f = np.array([s[0] for s in segments], float)
    d = np.array([s[1] for s in segments], float)
    edges = np.concatenate(([0], np.cumsum(d))) * fs / 1000.0
    n = int(edges[-1])
    idx = np.searchsorted(edges[1:], np.arange(n) + 0.5, side="right")
    inst = f[np.minimum(idx, len(f) - 1)]
    phase = 2 * np.pi * np.cumsum(inst) / fs
    return np.sin(phase)


def encode(mode, img, fs_out=8000):
    vis = SPECS[mode][0]
    x = synth(vis_segments(vis) + image_segments(mode, img))
    return resample_poly(x, fs_out, FS_GEN)
