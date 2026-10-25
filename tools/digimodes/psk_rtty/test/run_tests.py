#!/usr/bin/env python3
"""
Tests for tools/digimodes/psk_rtty/decoder.py.  Prints results to stdout only.

    /usr/bin/python3 tools/digimodes/psk_rtty/test/run_tests.py [--quick]

  * multi-signal streams (PSK31/PSK63/RTTY at different audio frequencies,
    0.5 Hz/s drift on some, one RTTY with reversed polarity) at several SNRs
    (white noise, SNR per signal in 3 kHz), fed in 800-sample int16 blocks;
    character error rate (Levenshtein / reference length) per signal
  * noise only (10 min), voice-like noise, voiced (harmonic) speech-like signal,
    steady carrier and keyed CW: count of false text
  * CPU time per second of audio, idle and busy (8 signals)
"""
from __future__ import annotations

import math
import os
import sys
import time

sys.dont_write_bytecode = True   # keep the repository free of generated files

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from decoder import DigiWatcher, FS  # noqa: E402
from encoders import psk_signal, rtty_signal, mix, to_int16  # noqa: E402

QUICK = "--quick" in sys.argv

TEXT_A = ("CQ CQ CQ de IK3ABC IK3ABC IK3ABC pse k\n"
          "IK3ABC de DL1XYZ ur rst 599 599 name Hans qth Munich. "
          "The quick brown fox jumps over the lazy dog 1234567890. "
          "Rig 100 W into a dipole, wx sunny 22 C. 73 and good dx de DL1XYZ sk ")
TEXT_B = ("QRZ? de OE5KLM OE5KLM. Thanks for the call, my name is Eva and qth is Linz. "
          "PSK is a nice mode, 31 baud and very narrow. Antenna is a vertical. "
          "How copy? IK3ABC de OE5KLM k ")
TEXT_C = ("CQ TEST DE W1AW W1AW W1AW K\n"
          "W1AW DE K9ZZ 599 05 05 K\nTU 73 DE W1AW QRZ TEST. "
          "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG 1234567890 ")
TEXT_D = ("RYRYRY CQ CQ DE F5ABC F5ABC PSE K\n"
          "F5ABC DE G4XYZ GM OM UR RST 579 579 NAME JOHN QTH LONDON HW? "
          "BTU F5ABC DE G4XYZ K ")


def levenshtein(a, b):
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def norm(s):
    return " ".join(s.split())


def run_stream(pcm, block=800):
    w = DigiWatcher(label="test")
    evs = []
    peak_tr = 0
    for i in range(0, len(pcm), block):
        evs += w.feed(pcm[i:i + block], i / FS)
        peak_tr = max(peak_tr, len(w.trackers))
    evs += w.flush()
    return w, evs, peak_tr


def scenario(sigdefs, snrs, seed, tail_s=4.0):
    """sigdefs: list of dict(kind, text, f, baud, drift, start, mark_upper, mode)"""
    rng = np.random.default_rng(seed)
    sigs = []
    tracks = []
    for d in sigdefs:
        if d["kind"] == "PSK":
            s, f = psk_signal(d["text"], d["baud"], d["f"], d.get("drift", 0.0), rng)
        else:
            s, f = rtty_signal(d["text"], d["f"], d["baud"], drift_hz_s=d.get("drift", 0.0),
                               mark_upper=d.get("mark_upper", True))
        sigs.append((s, d["start"]))
        tracks.append((d["start"], f))
    total = int((max(st + len(s) / FS for s, st in sigs) + tail_s) * FS)
    x = mix(sigs, snrs, total, rng)
    pcm = to_int16(x)
    w, evs, peak = run_stream(pcm)
    res = []
    for d, (st, f), snr in zip(sigdefs, tracks, snrs):
        def freq_at(ts):
            k = int(np.clip((ts - st) * FS, 0, len(f) - 1))
            return f[k]
        tol = 60 if d["kind"] == "RTTY" else (12 if d["baud"] < 40 else 20)
        mine = [e for e in evs if abs(e["audio_hz"] - freq_at(e["ts"])) < tol]
        hyp = "".join(e["text"] for e in sorted(mine, key=lambda e: e["ts"]))
        ref = d["text"].upper() if d["kind"] == "RTTY" else d["text"]
        cer = levenshtein(norm(ref), norm(hyp)) / max(1, len(norm(ref)))
        modes = sorted(set(e["mode"] for e in mine))
        snr_est = float(np.median([e["snr_db"] for e in mine])) if mine else float("nan")
        res.append(dict(mode=d["mode"], f=d["f"], snr=snr, cer=cer, modes=modes, n=len(mine),
                        snr_est=snr_est, hyp=hyp))
    unmatched = [e for e in evs if not any(
        abs(e["audio_hz"] - (dd["f"])) < 80 + 0.6 * 60 for dd in sigdefs)]
    return res, unmatched, w, len(pcm) / FS


def print_results(title, res, unmatched, show_text=False):
    print(f"\n== {title}")
    print(f"  {'mode':7s} {'audio':>6s} {'SNR':>5s} {'est':>6s} {'CER':>7s}  events  modes reported")
    for r in res:
        print(f"  {r['mode']:7s} {r['f']:6.0f} {r['snr']:5.0f} {r['snr_est']:6.1f} {100 * r['cer']:6.1f}%  "
              f"{r['n']:5d}   {','.join(r['modes'])}")
        if show_text:
            print("      >", norm(r["hyp"])[:160])
    if unmatched:
        print(f"  unmatched events: {len(unmatched)}: " +
              "; ".join(f"{e['mode']}@{e['audio_hz']}:{e['text']!r}" for e in unmatched[:5]))


def false_text(name, pcm):
    w, evs, peak = run_stream(pcm)
    nch = sum(len(e["text"].strip()) for e in evs)
    dur = len(pcm) / FS
    print(f"  {name:38s} {dur / 60:5.1f} min  events={len(evs):3d} chars={nch:4d}  "
          f"max trackers={peak}  cpu={100 * w.cpu_s / dur:5.2f}% of a core")
    for e in evs[:5]:
        print(f"      false: {e['mode']}@{e['audio_hz']} snr={e['snr_db']} {e['text']!r}")
    return len(evs), nch


def voice_noise(dur_s, rng):
    """Speech-shaped noise: band-pass + two formant resonances, syllable (3-6 Hz) envelope, pauses."""
    n = int(dur_s * FS)
    x = rng.normal(0, 1, n)
    sos = butter(4, [300, 3000], btype="band", fs=FS, output="sos")
    x = sosfilt(sos, x)
    for fc in (500, 1500):
        sos = butter(2, [fc * 0.8, fc * 1.25], btype="band", fs=FS, output="sos")
        x = x + 2 * sosfilt(sos, rng.normal(0, 1, n))
    env = _syllable_env(n, rng)
    bg = rng.normal(0, 0.03 * np.std(x), n)
    return x * env + bg


def _syllable_env(n, rng):
    env = np.zeros(n)
    t = 0
    while t < n:
        phrase = int(rng.uniform(1.5, 5) * FS)
        end = min(n, t + phrase)
        while t < end:
            syl = int(rng.uniform(0.12, 0.3) * FS)
            a = rng.uniform(0.4, 1.0)
            seg = a * np.sin(np.pi * np.arange(syl) / syl) ** 2
            m = min(syl, n - t)
            env[t:t + m] = seg[:m]
            t += syl + int(rng.uniform(0.0, 0.08) * FS)
        t += int(rng.uniform(0.3, 2.0) * FS)
    return env


def voiced_speech(dur_s, rng):
    """Harmonic voice: glottal pulse train, f0 90-220 Hz with intonation, moving formants."""
    n = int(dur_s * FS)
    t = np.arange(n) / FS
    f0 = 140 + 40 * np.sin(2 * np.pi * 0.3 * t) + 25 * np.sin(2 * np.pi * 1.7 * t + 1) \
        + 5 * np.sin(2 * np.pi * 5.5 * t)
    ph = np.cumsum(f0) / FS
    pulses = np.diff(np.floor(ph), prepend=0.0)
    src = lfilter([1.0], [1, -0.95], pulses)          # glottal tilt
    out = np.zeros(n)
    seg = int(0.2 * FS)
    zi = None
    vowels = [(700, 1200, 2500), (300, 2300, 3000), (500, 900, 2400), (400, 1900, 2600), (600, 1000, 2700)]
    for a in range(0, n, seg):
        F = vowels[rng.integers(len(vowels))]
        y = np.zeros(min(seg, n - a))
        for k, fc in enumerate(F):
            sos = butter(2, [fc * 0.9, fc * 1.1], btype="band", fs=FS, output="sos")
            y += (0.8 ** k) * sosfilt(sos, src[a:a + len(y)])
        out[a:a + len(y)] = y
    env = _syllable_env(n, rng)
    bg = rng.normal(0, 0.02 * np.std(out), n)
    return out * env + bg


def cw_signal(dur_s, f, wpm, rng):
    n = int(dur_s * FS)
    dot = 1.2 / wpm
    key = []
    while len(key) * dot < dur_s:
        for _ in range(rng.integers(1, 5)):
            key += [1] * (1 if rng.random() < 0.5 else 3) + [0]
        key += [0, 0] if rng.random() < 0.8 else [0] * 6
    k = np.repeat(np.array(key, float), int(dot * FS))[:n]
    k = np.pad(k, (0, n - len(k)))
    k = np.convolve(k, np.hanning(41) / np.hanning(41).sum(), mode="same")
    return k * np.cos(2 * np.pi * f * np.arange(n) / FS)


def main():
    t_start = time.time()
    print("PSK31/PSK63/RTTY decoder tests (white noise, SNR per signal in 3 kHz, 800-sample int16 blocks)")

    # --- 1. PSK mix --------------------------------------------------------
    psk_defs = [
        dict(kind="PSK", mode="PSK31", text=TEXT_A, f=600, baud=31.25, drift=0.5, start=2.0),
        dict(kind="PSK", mode="PSK31", text=TEXT_B, f=690, baud=31.25, drift=0.0, start=3.3),
        dict(kind="PSK", mode="PSK63", text=TEXT_A + TEXT_B, f=1150, baud=62.5, drift=-0.5, start=1.0),
        dict(kind="PSK", mode="PSK31", text=TEXT_B, f=2400, baud=31.25, drift=0.2, start=5.0),
    ]
    table = []
    psk_snrs = [10, 0, -5] if QUICK else [10, 0, -5, -8, -10]
    for snr in psk_snrs:
        res, un, w, dur = scenario(psk_defs, [snr] * len(psk_defs), seed=100 + snr)
        print_results(f"PSK mix (4 signals) @ {snr} dB", res, un, show_text=(snr in (0, -8)))
        table += res

    # --- 2. RTTY + PSK mix -------------------------------------------------
    rtty_defs = [
        dict(kind="RTTY", mode="RTTY45", text=TEXT_C, f=1500, baud=45.45, drift=0.5, start=2.0),
        dict(kind="RTTY", mode="RTTY45", text=TEXT_D, f=2300, baud=45.45, drift=0.0, start=3.0,
             mark_upper=False),
        dict(kind="PSK", mode="PSK31", text=TEXT_B, f=900, baud=31.25, drift=0.0, start=2.5),
    ]
    rtty_snrs = [10, 3] if QUICK else [10, 3, 0, -3]
    for snr in rtty_snrs:
        res, un, w, dur = scenario(rtty_defs, [snr, snr, 0], seed=200 + snr)
        print_results(f"RTTY45 x2 (one reversed) @ {snr} dB + PSK31 @ 0 dB", res, un, show_text=(snr == 3))
        table += res[:2]

    # --- 3. RTTY 50 / 75 ---------------------------------------------------
    r2_defs = [
        dict(kind="RTTY", mode="RTTY50", text=TEXT_C, f=1200, baud=50.0, start=2.0),
        dict(kind="RTTY", mode="RTTY75", text=TEXT_D, f=2000, baud=75.0, start=2.5, drift=-0.5),
    ]
    for snr in ([10, 3] if not QUICK else [10]):
        res, un, w, dur = scenario(r2_defs, [snr, snr], seed=300 + snr)
        print_results(f"RTTY50 + RTTY75 @ {snr} dB", res, un)
        table += res

    # --- 4. adjacent PSK31 stations ---------------------------------------
    for sp, lv in ((40, (3, 3, 3)), (50, (20, 0, 20)), (30, (3, 3, 3))):
        adj = [dict(kind="PSK", mode="PSK31", text=TEXT_A, f=1000, baud=31.25, start=2.0),
               dict(kind="PSK", mode="PSK31", text=TEXT_B, f=1000 + sp, baud=31.25, start=3.0),
               dict(kind="PSK", mode="PSK31", text=TEXT_B, f=1000 + 2 * sp, baud=31.25, start=2.5)]
        res, un, w, dur = scenario(adj, list(lv), seed=5)
        print_results(f"3 x PSK31 spaced {sp} Hz, SNR {lv} dB (not in summary)", res, un)

    # --- summary table -----------------------------------------------------
    print("\n== CER summary (mode x SNR, mean over signals)")
    agg = {}
    for r in table:
        agg.setdefault((r["mode"], r["snr"]), []).append(r)
    for (mode, snr), rs in sorted(agg.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        cer = np.mean([r["cer"] for r in rs])
        okm = all(r["modes"] == [mode] for r in rs)
        print(f"  {mode:7s} {snr:4d} dB  CER {100 * cer:5.1f}%  ({len(rs)} signal(s), mode id {'ok' if okm else 'WRONG/none'})")

    # --- false text --------------------------------------------------------
    print("\n== False text")
    rng = np.random.default_rng(7)
    noise_min = 2 if QUICK else 10
    pcm = to_int16(rng.normal(0, 0.05, int(noise_min * 60 * FS)))
    false_text("white noise", pcm)
    vmin = 1 if QUICK else 5
    false_text("speech-shaped noise, syllabic", to_int16(voice_noise(vmin * 60, rng)))
    false_text("voiced harmonic speech-like", to_int16(voiced_speech(vmin * 60, rng)))
    x = np.cos(2 * np.pi * 1234.5 * np.arange(int(60 * FS)) / FS) + rng.normal(0, 0.05, int(60 * FS))
    false_text("steady carrier 1234.5 Hz (+noise)", to_int16(x))
    x = cw_signal(120, 800, 20, rng) + cw_signal(120, 1700, 28, rng) * 0.5 + rng.normal(0, 0.05, 120 * FS)
    false_text("keyed CW 20+28 wpm (+noise)", to_int16(x))

    # --- CPU, busy ---------------------------------------------------------
    print("\n== CPU (busy: 6 x PSK31 + 1 x PSK63 + 1 x RTTY45 at 10 dB, 60 s)")
    rng = np.random.default_rng(9)
    busy = []
    for i, f in enumerate([500, 640, 780, 920, 1060, 2600]):
        busy.append(dict(kind="PSK", mode="PSK31", text=(TEXT_A + TEXT_B)[:240], f=f, baud=31.25, start=1 + 0.3 * i))
    busy.append(dict(kind="PSK", mode="PSK63", text=(TEXT_A + TEXT_B) * 2, f=1400, baud=62.5, start=1.0))
    busy.append(dict(kind="RTTY", mode="RTTY45", text=(TEXT_C + TEXT_D) * 2, f=2000, baud=45.45, start=1.0))
    res, un, w, dur = scenario(busy, [10] * len(busy), seed=9, tail_s=1.0)
    print(f"  audio {dur:.1f} s, decoder CPU {w.cpu_s:.2f} s -> {100 * w.cpu_s / dur:.2f}% of one core")
    print(f"  CER per signal: " + ", ".join(f"{r['mode']}@{r['f']}:{100 * r['cer']:.1f}%" for r in res))
    print(f"\n(total test wall time {time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
