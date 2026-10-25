#!/usr/bin/python3
"""SSTV decoder test-suite.

Generates SSTV signals (own standard-timing encoder, plus pysstv as a
cross-check), applies a tuning offset (analytic-signal frequency shift), an
optional sample-clock error, white noise at a given SNR, embeds them in noise,
feeds the result to SSTVWatcher in 800-sample int16 blocks and compares the
decoded PNG with the original.

SNR definition: tone power / white-noise power over the full 0..4 kHz band.

    /usr/bin/python3 tools/sstv/test/run_tests.py            # full suite
    /usr/bin/python3 tools/sstv/test/run_tests.py --quick
"""
import argparse
import os
import sys
import tempfile
import time
import zlib
from multiprocessing import Pool

import numpy as np
from scipy.signal import hilbert, resample_poly
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from decoder import SSTVWatcher, FS  # noqa: E402
import sstv_encode as E  # noqa: E402

AMP = 0.3 * 32767
OUT = os.path.join(tempfile.gettempdir(), "sstv_test_out")


def make_image(w, h, seed=1):
    """Deterministic test card: colour bars, grey ramp, smooth texture, text."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), float)
    tex = np.stack([gaussian_filter(rng.standard_normal((h, w)), (h / 30, w / 30)) for _ in range(3)], -1)
    img[:] = np.clip(128 + tex / tex.std() * 60, 0, 255)
    bars = np.array([[255, 255, 255], [255, 255, 0], [0, 255, 255], [0, 255, 0],
                     [255, 0, 255], [255, 0, 0], [0, 0, 255], [0, 0, 0]], float)
    bh = h // 6
    img[:bh] = bars[(np.arange(w) * 8) // w][None]
    img[bh:bh + h // 12] = (np.arange(w) * 255.0 / (w - 1))[None, :, None]
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(img.astype(np.uint8))
        d = ImageDraw.Draw(im)
        for i, y in enumerate(range(h // 2, h - 10, max(12, h // 12))):
            d.text((w // 10, y), "SSTV TEST %d" % i, fill=(255, 255, 255) if i % 2 else (0, 0, 0))
        img = np.asarray(im).astype(float)
    except ImportError:
        pass
    return img.astype(np.uint8)


def shift_freq(x, off):
    if off == 0:
        return x
    return np.real(hilbert(x) * np.exp(2j * np.pi * off * np.arange(len(x)) / FS))


def pysstv_encode(mode, img):
    from PIL import Image
    from pysstv import color
    cls = {"Martin M1": color.MartinM1, "Robot 36": color.Robot36, "PD90": color.PD90,
           "Martin M2": color.MartinM2, "PD120": color.PD120}[mode]
    s = cls(Image.fromarray(img), 48000, 16)
    x = np.fromiter(s.gen_values(), float)
    return resample_poly(x, FS, 48000)


def build(mode, img, snr, off, gen="own", ppm=0.0, cut=None, pre=3.0, post=6.0, seed=0, amp=AMP,
          then=None):
    x = E.encode(mode, img) if gen == "own" else pysstv_encode(mode, img)
    if then is not None:                 # a second image 2 s after the first one
        _, w2, h2 = E.SPECS[then]
        x = np.concatenate([x, np.zeros(2 * FS), E.encode(then, make_image(w2, h2, seed=2))])
    if ppm:
        # transmitter clock error: signal plays (1+ppm) faster/slower
        up = 100000
        x = resample_poly(x, up, int(round(up * (1 + ppm * 1e-6))))
    if cut is not None:
        x = x[:int(len(x) * cut)]
    x = shift_freq(x, off) * amp
    rng = np.random.default_rng(seed)
    x = np.concatenate([np.zeros(int(pre * FS)), x, np.zeros(int(post * FS))])
    sigma = amp / np.sqrt(2) / (10 ** (snr / 20.0)) if snr is not None else 0.0
    x = x + rng.standard_normal(len(x)) * sigma
    return np.clip(x, -32768, 32767).astype(np.int16)


def metrics(path, ref):
    from PIL import Image
    d = Image.open(path).convert("RGB")
    if d.size != (ref.shape[1], ref.shape[0]):
        d = d.resize((ref.shape[1], ref.shape[0]))
    d = np.asarray(d).astype(float)
    r = ref.astype(float)
    mse = ((d - r) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-9))
    corr = np.corrcoef(d.ravel(), r.ravel())[0, 1]
    return psnr, corr


def run_case(case):
    mode, snr, off = case["mode"], case["snr"], case["off"]
    _, w, h = E.SPECS[mode]
    img = make_image(w, h)
    seed = zlib.crc32(repr(sorted(case.items())).encode())
    pcm = build(mode, img, snr, off, case.get("gen", "own"), case.get("ppm", 0.0), case.get("cut"),
                seed=seed, amp=case.get("amp", AMP), then=case.get("then"))
    out_dir = os.path.join(OUT, "%s_%s_%s_%s" % (mode.replace(" ", ""), snr, off,
                                                 case.get("tag", "").replace(" ", "_").replace("%", "")))
    wch = SSTVWatcher(out_dir, label="t")
    res = []
    rng = np.random.default_rng(seed)
    blk = case.get("block", 800)
    i, peak, prog_max, prog_seen = 0, 0.0, 0, False
    t0 = time.process_time()
    while i < len(pcm):
        n = int(rng.integers(1, 4000)) if blk == "random" else blk
        t1 = time.perf_counter()
        res += wch.feed(pcm[i:i + n], 1.7e9 + i / FS)
        peak = max(peak, time.perf_counter() - t1)
        p = wch.progress()
        if p is not None:
            prog_seen = True
            prog_max = max(prog_max, p["lines"])
        i += n
    cpu = time.process_time() - t0
    r = dict(case)
    r.update(audio_s=len(pcm) / FS, cpu_s=cpu, n_img=len(res), peak_ms=peak * 1000,
             prog_ok=prog_seen and wch.progress() is None)
    if case.get("then"):
        r["second"] = [(x["mode"], x["lines"], x["partial"]) for x in res]
    if res:
        x = res[0]
        r.update(detected=True, mode_ok=(x["mode"] == mode), off_est=x["offset_hz"], lines=x["lines"],
                 partial=x["partial"])
        ref = img
        if case.get("cut") is not None:      # compare only the received rows
            n = x["lines"]
            from PIL import Image
            d = np.asarray(Image.open(x["path"]).convert("RGB"))[:n]
            tmp = x["path"] + ".crop.png"
            Image.fromarray(d).save(tmp)
            ref = img[:n]
            r["psnr"], r["corr"] = metrics(tmp, ref)
        else:
            r["psnr"], r["corr"] = metrics(x["path"], ref)
        r["path"] = x["path"]
    else:
        r.update(detected=False, mode_ok=False)
    return r


def cpu_idle(seconds=120.0):
    rng = np.random.default_rng(5)
    pcm = np.clip(rng.standard_normal(int(seconds * FS)) * 3000, -32768, 32767).astype(np.int16)
    w = SSTVWatcher(os.path.join(OUT, "idle"), "idle")
    t0 = time.process_time()
    n = 0
    for i in range(0, len(pcm), 800):
        n += len(w.feed(pcm[i:i + 800], i / FS))
    cpu = time.process_time() - t0
    return cpu / seconds, n, len(w.vis_seen)


def false_alarm(seconds=600.0):
    """Random tones (VIS/sync/video frequencies, 10..300 ms) in noise: worst case for VIS search."""
    rng = np.random.default_rng(3)
    segs, tot = [], 0.0
    while tot < seconds * 1000:
        d = float(rng.choice([10, 30, 30, 30, 60, 100, 300]))
        f = float(rng.choice([1100, 1200, 1300, 1500, 1900, 2300]) + rng.uniform(-20, 20))
        segs.append((f, d))
        tot += d
    x = E.synth(segs, FS) * 8000
    x = x + rng.standard_normal(len(x)) * 800
    w = SSTVWatcher(os.path.join(OUT, "falsealarm"), "fa")
    n = 0
    for i in range(0, len(x), 800):
        n += len(w.feed(x[i:i + 800].astype(np.int16), i / FS))
    return n, len(w.vis_seen)


def fmt_row(r):
    det = "yes" if r["detected"] else "NO"
    mok = "yes" if r["mode_ok"] else ("-" if not r["detected"] else "NO")
    q = ("%5.1f dB / %.3f" % (r["psnr"], r["corr"])) if r["detected"] else "-"
    snr = "clean" if r["snr"] is None else "%d dB" % r["snr"]
    extra = r.get("tag", "")
    if r.get("then"):
        extra += " images=%s" % r.get("second")
    if r["detected"]:
        extra += (" off_est=%+.1f" % r["off_est"]) + (" lines=%d" % r["lines"]) + (" PARTIAL" if r["partial"] else "")
    return "| %-10s | %-6s | %+5d Hz | %-4s | %-3s | %-18s | %s |" % (
        r["mode"], snr, r["off"], det, mok, q, extra.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("-j", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--md", default=os.path.join(HERE, "results.md"))
    a = ap.parse_args()

    modes = ["Martin M1", "Martin M2", "Scottie S1", "Scottie S2", "Scottie DX",
             "Robot 36", "Robot 72", "PD90", "PD120"]
    snrs = [20, 10, 5] if not a.quick else [10]
    offs = [120, -80]
    cases = [dict(mode=m, snr=s, off=o) for m in modes for s in snrs for o in offs]
    cases += [dict(mode=m, snr=None, off=0, tag="clean") for m in modes]
    if not a.quick:
        cases += [dict(mode=m, snr=0, off=150, tag="0dB") for m in ("Martin M1", "Scottie S1", "Robot 36", "PD120")]
        cases += [dict(mode=m, snr=10, off=-150, gen="pysstv", tag="pysstv")
                  for m in ("Martin M1", "Martin M2", "Robot 36", "PD90", "PD120")]
        cases += [dict(mode=m, snr=10, off=60, ppm=p, tag="clock %+dppm" % p)
                  for m in ("Martin M1", "Scottie S1", "PD120") for p in (-1000, 500)]
        cases += [dict(mode=m, snr=10, off=-40, cut=0.5, tag="signal cut at 50%")
                  for m in ("Martin M1", "PD120")]
        cases += [dict(mode="Martin M1", snr=10, off=250, tag="offset 250"),
                  dict(mode="Robot 36", snr=10, off=-250, tag="offset -250"),
                  dict(mode="Scottie S1", snr=10, off=50, amp=300.0, tag="low level (amp 300)"),
                  dict(mode="Robot 36", snr=10, off=-30, block="random", tag="random block sizes 1..4000"),
                  dict(mode="Martin M1", snr=10, off=70, then="Robot 36", tag="back-to-back")]

    t0 = time.time()
    with Pool(a.j) as pool:
        results = pool.map(run_case, cases, chunksize=1)
    wall = time.time() - t0

    idle_frac, idle_imgs, idle_vis = cpu_idle(60.0 if a.quick else 600.0)
    fa_imgs, fa_vis = false_alarm(60.0 if a.quick else 600.0)
    dec = [r for r in results if r["detected"] and not r.get("cut")]
    dec_frac = np.mean([r["cpu_s"] / r["audio_s"] for r in dec]) if dec else float("nan")

    lines = ["# SSTV decoder test results", "",
             "Generated by `run_tests.py` on %s (%.0f s wall). SNR = tone power / white noise power in 0-4 kHz."
             % (time.strftime("%Y-%m-%d %H:%M"), wall), "",
             "| mode       | SNR    | offset   | det  | mode| PSNR / corr        | notes |",
             "|------------|--------|----------|------|-----|--------------------|-------|"]
    lines += [fmt_row(r) for r in results]
    n_ok = sum(r["detected"] and r["mode_ok"] for r in results)
    peak = max(r["peak_ms"] for r in results)
    prog_bad = [r["mode"] for r in results if not r["prog_ok"]]
    lines += ["", "%d / %d cases detected with correct mode." % (n_ok, len(results)), "",
              "## CPU (one core, this machine)", "",
              "* idle (noise only, %d s): %.3f %% of one core, false images: %d, VIS candidates rejected: %d"
              % (60 if a.quick else 600, idle_frac * 100, idle_imgs, idle_vis),
              "* false-alarm test (random 1100..2300 Hz tone hopping, %d s): images: %d, "
              "headers reaching bit decode (all rejected unless an image was saved): %d"
              % (60 if a.quick else 600, fa_imgs, fa_vis),
              "* average over full decodes (incl. idle lead-in/out): %.2f %% of one core" % (dec_frac * 100),
              "* worst single feed() call (includes final PNG render; tests ran %d in parallel): %.0f ms"
              % (a.j, peak),
              "* progress() reported during reception and None afterwards: %s"
              % ("all cases" if not prog_bad else "FAILED for %s" % prog_bad)]
    txt = "\n".join(lines) + "\n"
    print(txt)
    with open(a.md, "w") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
