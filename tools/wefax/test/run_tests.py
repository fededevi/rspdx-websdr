#!/usr/bin/python3
"""WEFAX decoder test-suite (prints to stdout, temp files under /tmp only).

Generates radiofax transmissions with the own encoder (start tone, phasing,
test chart, stop tone), applies tuning offset, station clock error and white
noise, feeds the audio to WefaxWatcher in 800-sample int16 blocks and compares
the decoded PNG with the source image.  Also runs 10-minute false-alarm
recordings (noise, voice-like audio, RTTY, carrier) and measures CPU.

SNR definition: FM signal power / white-noise power over the full 0..4 kHz band.

    /usr/bin/python3 tools/wefax/test/run_tests.py [--quick]
"""
import os
import sys
sys.dont_write_bytecode = True
# test harness only: keep numpy's BLAS single-threaded so 8 worker processes
# do not oversubscribe the CPU (the decoder itself does not call BLAS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import zlib
import os
import shutil
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, lfilter, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from decoder import WefaxWatcher, FS  # noqa: E402
import wefax_encode as E  # noqa: E402

AMP = 0.3 * 32767
OUT = os.path.join(tempfile.gettempdir(), "wefax_test_out")


def make_chart(W, H, seed=3):
    """Weather-chart-like test card: white background, grid, text, ramps,
    diagonal lines, circles, checkerboard."""
    im = Image.new("L", (W, H), 255)
    dr = ImageDraw.Draw(im)
    sx, sy = W / 1810.0, H / 400.0
    for x in range(0, W, max(8, int(150 * sx))):
        dr.line([(x, 0), (x, H)], fill=0, width=max(1, int(3 * sx)))
    for y in range(0, H, max(4, int(60 * sy))):
        dr.line([(0, y), (W, y)], fill=0, width=2)
    try:
        font = ImageFont.load_default(size=max(10, int(36 * sy)))
    except TypeError:  # old Pillow
        font = ImageFont.load_default()
    dr.text((int(40 * sx), int(15 * sy)), "DWD PINNEBERG 7880 KHZ TEST 1234567890", fill=0, font=font)
    dr.text((int(900 * sx), int(140 * sy)), "WEFAX DECODER TEST CHART", fill=0, font=font)
    for i in range(4):
        cx, cy, r = int((300 + 350 * i) * sx), int(250 * sy), int((40 + 25 * i) * sy)
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=3)
    for i in range(6):
        dr.line([(int(1300 * sx), int(80 * sy)), (int((1450 + 60 * i) * sx), int(380 * sy))], fill=0, width=2)
    a = np.array(im, np.float64) / 255.0
    # horizontal grey ramp band and vertical ramp block
    y0, y1 = int(320 * sy), int(360 * sy)
    a[y0:y1, :] = np.linspace(0, 1, W)[None, :]
    x0, x1 = int(40 * sx), int(200 * sx)
    a[int(80 * sy):int(300 * sy), x0:x1] = np.linspace(0, 1, int(300 * sy) - int(80 * sy))[:, None]
    # checkerboard
    cb = ((np.arange(H)[:, None] // 8 + np.arange(W)[None, :] // 16) % 2).astype(float)
    a[int(370 * sy):, int(1400 * sx):int(1750 * sx)] = cb[int(370 * sy):, int(1400 * sx):int(1750 * sx)]
    return a


def add_noise(x, snr_db, rng):
    ps = AMP ** 2 / 2
    if snr_db is None:
        return x * AMP
    sig = ps / 10 ** (snr_db / 10)
    return x * AMP + rng.normal(0, np.sqrt(sig), x.size)


def to_pcm(x):
    return np.clip(np.round(x), -32768, 32767).astype(np.int16)


def run_watcher(pcm, out_dir, t0=1.7e9):
    w = WefaxWatcher(out_dir, "t")
    res = []
    tcpu = time.process_time()
    w.ui = {"progress_max_lines": 0, "preview_shape": None}
    for i in range(0, len(pcm), 800):
        res += w.feed(pcm[i:i + 800], t0 + i / FS)
        if i % (800 * 50) == 0:                  # what the web UI would poll
            p = w.progress()
            if p:
                w.ui["progress_max_lines"] = max(w.ui["progress_max_lines"], p["lines"])
                pv = w.preview()
                if pv is not None:
                    w.ui["preview_shape"] = pv.shape
    res += w.flush()
    return res, time.process_time() - tcpu, w


def _xcorr_cols(d, s):
    """cc[k] = sum over rows of the circular cross-correlation: d ~ roll(s, k)."""
    X = np.fft.rfft(d - d.mean(), axis=1)
    Y = np.fft.rfft(s - s.mean(), axis=1)
    return np.fft.irfft((X * np.conj(Y)).sum(axis=0), n=s.shape[1])


def best_align(dec, src, rows=4, cols=30, circular=False):
    """-> (corr, psnr, drow, dcol) for the best integer alignment (row shift
    +-rows; column shift +-cols, or any circular shift)."""
    best = (-2, 0, 0, 0)
    W = src.shape[1]
    for dr in range(-rows, rows + 1):
        a0, b0 = max(0, dr), max(0, -dr)
        n = min(dec.shape[0] - a0, src.shape[0] - b0)
        if n < 20:
            continue
        d = dec[a0:a0 + n]
        s = src[b0:b0 + n]
        cc = _xcorr_cols(d, s)
        ks = np.arange(W) if circular else np.r_[0:cols + 1, W - cols:W]
        k = int(ks[np.argmax(cc[ks])])
        ss = np.roll(s, k, axis=1)
        c = np.corrcoef(d.ravel(), ss.ravel())[0, 1]
        if c > best[0]:
            mse = np.mean((d - ss) ** 2)
            best = (c, 10 * np.log10(1.0 / max(mse, 1e-12)), dr, k if k <= W // 2 else k - W)
    return best


def col_shift(dec, src, cols=40):
    W = src.shape[1]
    cc = _xcorr_cols(dec, src)
    ks = np.r_[0:cols + 1, W - cols:W]
    k = int(ks[np.argmax(cc[ks])])
    return k if k <= W // 2 else k - W


def case(args):
    name, lpm, ioc, snr, off, ppm, H, kw = args
    kw = dict(kw)
    tail_s = kw.pop("tail_s", 5)
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    W = int(round(np.pi * ioc))
    src = make_chart(W, H)
    x, n0 = E.encode(src, lpm, ioc, ppm, off, **kw)
    lead = np.zeros(int(3 * FS))
    tail = np.zeros(int(tail_s * FS))
    x = np.concatenate([lead, x, tail])
    pcm = to_pcm(add_noise(x, snr, rng))
    od = os.path.join(OUT, name)
    shutil.rmtree(od, ignore_errors=True)
    res, cpu, w = run_watcher(pcm, od)
    dur = len(pcm) / FS
    r = {"name": name, "lpm": lpm, "ioc": ioc, "snr": snr, "off": off, "ppm": ppm,
         "n_img": len(res), "cpu": cpu / dur, "ui": w.ui}
    if not res:
        r["events"] = w.events[:6]
        return r
    best = max(res, key=lambda q: q["lines"])
    dec = np.asarray(Image.open(best["path"]), np.float64) / 255.0
    skip = kw.get("skip_lines", 0)
    src_c = src[skip:]
    mid = best.get("mid_start", False)
    if dec.shape[1] != W:
        dec = np.asarray(Image.fromarray((dec * 255).astype(np.uint8)).resize((W, dec.shape[0])), float) / 255
    c, psnr, drow, dcol = best_align(dec, src_c, rows=60 if mid else 4, circular=mid)
    r.update({"det_lpm": best["lpm"], "det_ioc": best["ioc"], "lines": best["lines"], "src_lines": H - skip,
              "det_off": best["offset_hz"], "det_ppm": best.get("ppm"), "partial": best["partial"],
              "mid": mid, "corr": c, "psnr": psnr, "drow": drow, "dcol": dcol})
    # slant: column shift in top vs bottom third (after row alignment)
    a0, b0 = max(0, drow), max(0, -drow)
    n = min(dec.shape[0] - a0, src_c.shape[0] - b0)
    d, s = dec[a0:a0 + n], np.roll(src_c[b0:b0 + n], dcol, axis=1)
    t = n // 3
    r["slant_px"] = col_shift(d[-t:], s[-t:]) - col_shift(d[:t], s[:t])
    return r


def voice_like(dur, rng):
    """Synthetic speech-like audio: glottal pulse train with drifting pitch,
    3 formant resonators changing per syllable, syllabic envelope with pauses."""
    n = int(dur * FS)
    out = np.zeros(n)
    i = 0
    while i < n:
        L = int(rng.uniform(0.12, 0.35) * FS)
        seg = np.zeros(L)
        f0 = rng.uniform(90, 240)
        ph = np.cumsum(np.full(L, f0) * (1 + 0.1 * np.sin(np.linspace(0, 3, L)))) / FS
        seg[np.where(np.diff(np.floor(ph)) > 0)[0]] = 1.0
        if rng.random() < 0.25:                   # unvoiced
            seg = rng.normal(0, 0.3, L)
        for fc, bw in ((rng.uniform(300, 900), 80), (rng.uniform(900, 2300), 120), (rng.uniform(2300, 3000), 160)):
            r = np.exp(-np.pi * bw / FS)
            a = [1, -2 * r * np.cos(2 * np.pi * fc / FS), r * r]
            seg = seg + 0.5 * lfilter([1 - r], a, seg)
        seg *= np.hanning(L)
        if rng.random() < 0.2:
            seg *= 0                              # pause
        out[i:i + L] = seg[:n - i]
        i += L
    sos = butter(4, [300, 2700], btype="band", fs=FS, output="sos")
    out = sosfilt(sos, out)
    return out / (np.std(out) + 1e-9) * AMP * 0.5


def rtty(dur, rng):
    n = int(dur * FS)
    baud = 45.45
    nb = int(dur * baud) + 1
    bits = rng.integers(0, 2, nb)
    b = bits[(np.arange(n) / FS * baud).astype(int)]
    f = 1900 + np.where(b, 85.0, -85.0)
    return np.sin(2 * np.pi * np.cumsum(f) / FS) * AMP


def false_alarm(args):
    kind, dur, seed = args
    rng = np.random.default_rng(seed)
    n = int(dur * FS)
    if kind == "noise":
        x = rng.normal(0, AMP * 0.5, n)
    elif kind == "voice":
        x = voice_like(dur, rng) + rng.normal(0, AMP * 0.05, n)
    elif kind == "rtty":
        x = rtty(dur, rng) + rng.normal(0, AMP * 0.2, n)
    elif kind == "carrier+noise":
        x = AMP * np.sin(2 * np.pi * 1850 * np.arange(n) / FS) + rng.normal(0, AMP * 0.3, n)
    elif kind == "silence":
        x = np.zeros(n)
    od = os.path.join(OUT, "fa_" + kind)
    shutil.rmtree(od, ignore_errors=True)
    res, cpu, w = run_watcher(to_pcm(x), od)
    starts = [e for e in w.events if e[0] in ("tone", "mid", "image")]
    return {"kind": kind, "dur": dur, "images": len(res), "triggers": len(starts),
            "cpu": cpu / dur, "events": w.events[:5]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="run only image cases containing this text; skip false-alarm runs")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    H120 = 240 if a.quick else 400
    cases = [
        ("120_576_clean", 120, 576, None, 0.0, 0.0, H120, {}),
        ("120_576_20dB_+150Hz_100ppm", 120, 576, 20, 150.0, 100.0, H120, {}),
        ("120_576_10dB_-100Hz_-100ppm", 120, 576, 10, -100.0, -100.0, H120, {}),
        ("120_576_5dB_+200Hz_50ppm", 120, 576, 5, 200.0, 50.0, H120, {}),
        ("120_576_0dB_-200Hz_100ppm", 120, 576, 0, -200.0, 100.0, H120, {}),
        ("120_576_-3dB_+100Hz", 120, 576, -3, 100.0, 0.0, H120, {}),
        ("60_576_10dB_+150Hz_100ppm", 60, 576, 10, 150.0, 100.0, H120 // 2, {}),
        ("120_288_10dB_-100Hz_100ppm", 120, 288, 10, -100.0, 100.0, H120, {}),
        ("90_576_15dB_+50Hz", 90, 576, 15, 50.0, -50.0, H120 * 3 // 4, {}),
        ("240_576_15dB_-50Hz", 240, 576, 15, -50.0, 80.0, H120 * 2, {}),
        ("120_576_15dB_blackpulse", 120, 576, 15, 80.0, 100.0, H120, {"phasing_white_pulse": False}),
        ("120_576_10dB_nostop(loss)", 120, 576, 10, 100.0, 100.0, H120, {"stop": False, "tail_s": 30}),
        ("mid_120_15dB_+120Hz_100ppm", 120, 576, 15, 120.0, 100.0, H120 + 100,
         {"start": False, "phasing_s": 0, "skip_lines": 60}),
        ("mid_120_8dB_-80Hz_-60ppm", 120, 576, 8, -80.0, -60.0, H120 + 100,
         {"start": False, "phasing_s": 0, "skip_lines": 60}),
        ("notone_phasing_120_15dB", 120, 576, 15, 60.0, 100.0, H120, {"start": False}),
    ]
    if a.only:
        cases = [c for c in cases if a.only in c[0]]
    t0 = time.time()
    fa_dur = 120 if a.quick else 600
    fas = [("noise", fa_dur, 1), ("voice", fa_dur, 2), ("rtty", fa_dur, 3),
           ("carrier+noise", fa_dur, 4), ("silence", fa_dur, 5)]
    if a.only:
        fas = []
    with Pool(min(8, os.cpu_count() or 2)) as pool:
        fr = pool.map_async(false_alarm, fas)
        res = pool.map(case, cases)
        fres = fr.get()
    print("\nImage tests (SNR over 0-4 kHz; corr/PSNR vs source after best integer alignment)")
    hdr = "%-30s %5s %5s %6s %7s %6s %6s %5s %6s %6s %6s %5s %4s %4s %5s"
    print(hdr % ("case", "lpm", "ioc", "lines", "off", "ppm", "dppm", "part", "corr", "PSNR", "slant", "drow", "dcol", "imgs", "cpu%"))
    ok_all = True
    for r in res:
        if "corr" not in r:
            print("%-30s  NO IMAGE  %s" % (r["name"], r.get("events")))
            ok_all = False
            continue
        print("%-30s %5s %5s %6s %7.1f %6.1f %6.1f %5s %6.3f %6.1f %6d %5d %4d %4d %5.2f" % (
            r["name"], r["det_lpm"], r["det_ioc"], "%d/%d" % (r["lines"], r["src_lines"]), r["det_off"],
            r["ppm"], r["det_ppm"] or 0, "y" if r["partial"] else "n", r["corr"], r["psnr"], r["slant_px"],
            r["drow"], r["dcol"], r["n_img"], 100 * r["cpu"]))
    ui_ok = all(r["ui"]["progress_max_lines"] > 0 and r["ui"]["preview_shape"] is not None
                for r in res if "corr" in r)
    print("progress()/preview() polled during reception: %s" % ("ok" if ui_ok else "FAILED"))
    print("\nFalse-alarm runs")
    for r in fres:
        print("%-14s %5.0f s  images=%d  start-triggers=%d  cpu=%.2f%%  %s" % (
            r["kind"], r["dur"], r["images"], r["triggers"], 100 * r["cpu"], r["events"] if r["triggers"] else ""))
    # CPU benchmark, sequential (no other load): 120 lpm fax at 10 dB, then idle noise
    src = make_chart(1810, 240)
    x, _ = E.encode(src, 120, 576, 100.0, 150.0)
    rng = np.random.default_rng(9)
    pcm = to_pcm(add_noise(np.concatenate([x, np.zeros(120 * FS)]), 10, rng))
    w = WefaxWatcher(os.path.join(OUT, "cpu"), "cpu")
    tim = []
    for i in range(0, len(pcm), 800):
        t1 = time.perf_counter()
        w.feed(pcm[i:i + 800], 1.7e9 + i / FS)
        tim.append(time.perf_counter() - t1)
    tim = np.array(tim)
    print("\nCPU (sequential, %.0f s audio): mean %.2f%% of one core, mean feed %.2f ms, "
          "p99 %.2f ms, max %.1f ms per 0.1 s block" % (
              len(pcm) / FS, 100 * tim.sum() / (len(pcm) / FS), 1e3 * tim.mean(),
              1e3 * np.percentile(tim, 99), 1e3 * tim.max()))
    print("\nwall %.0f s; images under %s" % (time.time() - t0, OUT))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
