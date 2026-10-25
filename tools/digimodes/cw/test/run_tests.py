#!/usr/bin/python3
"""
Synthetic tests for tools/digimodes/cw/decoder.py.  Prints results to stdout
only (no files are written anywhere).

    /usr/bin/python3 tools/digimodes/cw/test/run_tests.py            # full run (~ a few minutes)
    /usr/bin/python3 tools/digimodes/cw/test/run_tests.py --quick    # shorter scenarios
    /usr/bin/python3 tools/digimodes/cw/test/run_tests.py --only false,matrix,qsb,qso,multi

Signals: random callsigns / contest exchanges / plain text, raised-cosine
keying edges (5 ms), per-element timing jitter, per-station dah weight,
2-4 simultaneous signals at different tones and speeds, white noise at a given
SNR in 3 kHz bandwidth (SNR = key-down carrier power / noise power in 3 kHz),
optional QSB, SSB channel filter 200-2900 Hz and a simple AGC.
CER = Levenshtein distance / reference length, spaces removed.
"""
from __future__ import annotations

import argparse
import sys
sys.dont_write_bytecode = True   # keep the repository clean
import os
import random
import string
import sys
import time

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from decoder import CWWatcher, MORSE, FS  # noqa: E402

CODE = {v: k for k, v in MORSE.items() if len(v) == 1}

# ---------------------------------------------------------------- text
PREFIXES = ["DL", "DK", "DJ", "OH", "OK", "OM", "SP", "G", "M", "F", "I", "IK", "IZ", "EA",
            "K", "W", "N", "AA", "VE", "JA", "LZ", "YO", "UA", "S5", "9A", "HA", "PA", "ON"]
PLAIN = ["NAME FRED QTH MUNICH", "RST 599 599", "TNX FER CALL", "GM OM", "HW CPY?",
         "WX SUNNY TEMP 18C", "RIG IC7300 PWR 100W", "ANT DIPOLE", "73 ES GL", "BK",
         "QSL VIA BURO", "UR RST 579", "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"]


def callsign(rng):
    return (rng.choice(PREFIXES) + str(rng.randint(0, 9))
            + "".join(rng.choice(string.ascii_uppercase) for _ in range(rng.randint(1, 3))))


def message(rng, call):
    r = rng.random()
    other = callsign(rng)
    if r < 0.25:
        return f"CQ TEST {call} {call}"
    if r < 0.45:
        return f"{other} 5NN {rng.randint(1, 40)}"
    if r < 0.55:
        return f"TU {call}"
    if r < 0.65:
        return f"R 5NN {rng.randint(1, 40):02d} TU"
    if r < 0.75:
        return f"CQ CQ DE {call} {call} K"
    return f"{other} DE {call} " + rng.choice(PLAIN)


# ------------------------------------------------------------ keying
def key_timeline(text, wpm, rng, jitter=0.15, dah=3.0):
    """Return list of (on, duration_s)."""
    u = 1.2 / wpm
    out = []

    def j(d):
        return max(d * (1.0 + rng.uniform(-jitter, jitter)), 0.25 * u)
    words = text.split()
    for wi, w in enumerate(words):
        for ci, ch in enumerate(w):
            code = CODE[ch]
            for ei, e in enumerate(code):
                out.append((1, j(u if e == "." else dah * u)))
                if ei < len(code) - 1:
                    out.append((0, j(u)))
            if ci < len(w) - 1:
                out.append((0, j(3 * u)))
        if wi < len(words) - 1:
            out.append((0, j(7 * u)))
    return out


def make_signal(n, tone, wpm, rng, jitter=0.15, start=0.5, gap=(1.0, 4.0), pause_frac=1.0):
    """Keyed tone, unit amplitude.  Returns (signal, reference text)."""
    key = np.zeros(n, dtype=np.float32)
    call = callsign(rng)
    dah = rng.uniform(2.8, 3.3)
    t = start
    sent = []
    while True:
        msg = message(rng, call)
        tl = key_timeline(msg, wpm, rng, jitter, dah)
        dur = sum(d for _, d in tl)
        if (t + dur + 0.3) * FS >= n:
            break
        for on, d in tl:
            i0, i1 = int(t * FS), int((t + d) * FS)
            if on:
                key[i0:i1] = 1.0
            t += d
        sent.append(msg)
        t += rng.uniform(*gap) * pause_frac
    edge = np.hanning(int(0.005 * FS))
    key = np.convolve(key, edge / edge.sum(), mode="same")
    ph = rng.uniform(0, 2 * np.pi)
    tt = np.arange(n) / FS
    drift = 2 * np.pi * rng.uniform(-0.05, 0.05) * tt ** 2 / 2    # slight drift (Hz/s)
    return key * np.cos(2 * np.pi * tone * tt + ph + drift), " ".join(sent)


def qsb(n, rng, depth_db=10.0, f=None):
    f = f or rng.uniform(0.05, 0.25)
    tt = np.arange(n) / FS
    g = -depth_db * (0.5 - 0.5 * np.cos(2 * np.pi * f * tt + rng.uniform(0, 2 * np.pi)))
    return 10 ** (g / 20)


def channel(x, rng, agc=True):
    """SSB filter 200-2900 Hz, AGC, int16."""
    sos = butter(6, [200, 2900], btype="band", fs=FS, output="sos")
    x = sosfilt(sos, x)
    if agc:
        blk = 80                               # 10 ms
        nb = len(x) // blk
        r = np.sqrt(np.mean(x[:nb * blk].reshape(nb, blk) ** 2, axis=1) + 1e-12)
        env = np.empty(nb)
        e = r[0]
        att, dec = 0.5, 1 - np.exp(-0.01 / 0.8)  # fast attack, 0.8 s decay
        for i in range(nb):
            e = e + att * (r[i] - e) if r[i] > e else e + dec * (r[i] - e)
            env[i] = e
        g = np.repeat(3000.0 / env, blk)
        g = np.concatenate([g, np.full(len(x) - len(g), g[-1])])
        x = x * g
    else:
        x = x * (3000.0 / (np.std(x) + 1e-12))
    return np.clip(x, -32767, 32767).astype(np.int16)


def noise_sigma(snr_db):
    # SNR = (A^2/2) / (sigma^2 * 3000/4000), A = 1
    return np.sqrt(0.5 / (0.75 * 10 ** (snr_db / 10)))


def scenario(dur, sigs, snr_db, seed, jitter=0.15, qsb_db=0.0, agc=True):
    """sigs: list of (tone, wpm).  Returns pcm, [(tone, wpm, ref)]."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    n = int(dur * FS)
    x = np.zeros(n)
    refs = []
    for tone, wpm in sigs:
        s, ref = make_signal(n, tone, wpm, rng, jitter, start=rng.uniform(0.3, 3.0))
        if qsb_db:
            s = s * qsb(n, rng, qsb_db)
        x += s
        refs.append((tone, wpm, ref))
    x += nrng.normal(0, noise_sigma(snr_db), n)
    return channel(x, rng, agc), refs


# ---------------------------------------------------------------- metrics
def lev(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    bb = np.frombuffer(b.encode(), dtype=np.uint8)
    prev = np.arange(len(b) + 1)
    ar = np.arange(len(b) + 1)
    for i, ch in enumerate(a.encode(), 1):
        cost = (bb != ch).astype(int)
        r = np.empty_like(prev)
        r[0] = i
        r[1:] = np.minimum(prev[1:] + 1, prev[:-1] + cost)
        r = np.minimum.accumulate(r - ar) + ar
        prev = r
    return int(prev[-1])


def run_decoder(pcm, label=""):
    w = CWWatcher(label)
    evs = []
    t0 = 1_700_000_000.0
    c0 = time.process_time()
    maxtr = 0
    for i in range(0, len(pcm) - 799, 800):
        evs += w.feed(pcm[i:i + 800], t0 + i / FS)
        maxtr = max(maxtr, len(w._trackers))
    evs += w.flush()
    cpu = time.process_time() - c0
    for e in evs:
        e["t"] = e["ts"] - t0
    return evs, cpu, w, maxtr


def score(evs, refs, tol=40.0):
    """Per reference signal: (cer, decoded text).  Plus unassigned text."""
    res = []
    used = set()
    for tone, wpm, ref in refs:
        mine = [e for e in evs if abs(e["audio_hz"] - tone) < tol]
        for e in mine:
            used.add(id(e))
        dec = " ".join(e["text"] for e in sorted(mine, key=lambda e: e["ts"]))
        r = ref.replace(" ", "")
        d = dec.replace(" ", "")
        res.append((lev(r, d) / max(len(r), 1), dec, ref,
                    np.median([e["wpm"] for e in mine]) if mine else 0.0,
                    np.median([e["snr_db"] for e in mine]) if mine else 0.0))
    stray = [e for e in evs if id(e) not in used]
    return res, stray


def tones(rng, k, lo=350, hi=2700, sep=160):
    while True:
        t = sorted(rng.uniform(lo, hi) for _ in range(k))
        if all(b - a >= sep for a, b in zip(t, t[1:])):
            return t


# ---------------------------------------------------------------- tests
def test_matrix(args):
    wpms = [12, 20, 28, 35, 40]
    snrs = [10, 0, -5, -10]
    dur = 40 if args.quick else 75
    print(f"\n== CER matrix: 3 simultaneous signals per cell (wpm x0.9/1.0/1.1), "
          f"jitter +-15%, {dur} s, AGC, no QSB ==")
    print("   CER (spaces ignored), mean of the 3 signals [min..max]")
    print("WPM \\ SNR(3k) " + "".join(f"{s:>17d} dB" for s in snrs))
    table = {}
    tot_cpu = tot_audio = 0.0
    for wpm in wpms:
        row = f"{wpm:>13d} "
        for snr in snrs:
            seed = 1000 + wpm * 10 + snr
            rng = random.Random(seed)
            ts = tones(rng, 3)
            sigs = [(ts[0], wpm * 0.9), (ts[1], wpm * 1.0), (ts[2], wpm * 1.1)]
            pcm, refs = scenario(dur, sigs, snr, seed)
            evs, cpu, _, _ = run_decoder(pcm)
            tot_cpu += cpu
            tot_audio += dur
            res, stray = score(evs, refs)
            cers = [min(r[0], 1.0) for r in res]
            table[(wpm, snr)] = cers
            row += f"   {np.mean(cers)*100:5.1f}% [{min(cers)*100:4.0f}..{max(cers)*100:4.0f}]"
            if args.verbose:
                for (c, dec, ref, ew, es), (tone, w, _) in zip(res, refs):
                    print(f"     {tone:6.0f}Hz {w:4.1f}wpm est {ew:4.1f} snr_est {es:+5.1f} "
                          f"CER {c*100:5.1f}%\n       REF {ref[:150]}\n       DEC {dec[:150]}")
                if stray:
                    print("     stray:", [(e['audio_hz'], e['text']) for e in stray][:10])
        print(row, flush=True)
    print(f"   (CPU over matrix: {tot_cpu/tot_audio*100:.2f}% of one core)")
    return table


def test_qsb(args):
    dur = 40 if args.quick else 75
    print(f"\n== QSB: 3 signals (16/25/33 wpm), 10 dB fades 0.05-0.25 Hz, jitter 15%, {dur} s ==")
    for snr in [10, 0, -5]:
        seed = 777 + snr
        rng = random.Random(seed)
        ts = tones(rng, 3)
        pcm, refs = scenario(dur, list(zip(ts, [16, 25, 33])), snr, seed, qsb_db=10.0)
        evs, cpu, _, _ = run_decoder(pcm)
        res, stray = score(evs, refs)
        s = "  ".join(f"{r[0]*100:5.1f}%@{refs[i][1]:.0f}wpm" for i, r in enumerate(res))
        print(f"  SNR(peak) {snr:+3d} dB: CER {s}   stray chars {sum(len(e['text']) for e in stray)}")
        if args.verbose:
            for c, dec, ref, ew, es in res:
                print(f"     REF {ref[:120]}\n     DEC {dec[:120]}")


def make_qso(n, tone_a, tone_b, wpm_a, wpm_b, rng, jitter=0.15):
    """Two stations alternating on (nearly) the same frequency."""
    key = np.zeros(n)
    tone = np.zeros(n)
    ca, cb = callsign(rng), callsign(rng)
    t, sent, turn = 0.5, [], 0
    while True:
        if turn % 2 == 0:
            msg, wpm, f = rng.choice([f"CQ TEST {ca}", f"{cb} 5NN {rng.randint(1, 40)}",
                                      f"TU {ca}"]), wpm_a, tone_a
        else:
            msg, wpm, f = rng.choice([cb, f"R 5NN {rng.randint(1, 40)}", f"{cb} {cb}"]), wpm_b, tone_b
        tl = key_timeline(msg, wpm, rng, jitter)
        dur = sum(d for _, d in tl)
        if (t + dur + 0.3) * FS >= n:
            break
        for on, d in tl:
            i0, i1 = int(t * FS), int((t + d) * FS)
            if on:
                key[i0:i1] = 1.0
            tone[i0:i1] = f
            t += d
        sent.append(msg)
        turn += 1
        t += rng.uniform(0.3, 1.0)
    tone[tone == 0] = tone_a
    edge = np.hanning(int(0.005 * FS))
    key = np.convolve(key, edge / edge.sum(), mode="same")
    return key * np.cos(2 * np.pi * np.cumsum(tone) / FS), " ".join(sent)


def test_qso(args):
    dur = 40 if args.quick else 90
    print(f"\n== Run station + callers alternating on one frequency, {dur} s ==")
    for (fa, fb, wa, wb, snr) in [(700, 708, 32, 32, 5), (700, 705, 35, 24, 5), (1200, 1208, 28, 18, 0),
                                  (900, 980, 34, 22, 5), (1500, 1620, 30, 18, 0)]:
        rng = random.Random(fa + wb)
        n = int(dur * FS)
        s, ref = make_qso(n, fa, fb, wa, wb, rng)
        x = s + np.random.default_rng(fb).normal(0, noise_sigma(snr), n)
        evs, cpu, _, _ = run_decoder(channel(x, rng))
        if fb - fa < 30:      # one stream, both stations
            res, stray = score(evs, [((fa + fb) / 2, wa, ref)], tol=45)
        else:                 # two separate trackers: merge their text in time order
            mine = [e for e in evs if abs(e["audio_hz"] - fa) < 30 or abs(e["audio_hz"] - fb) < 30]
            res, stray = score([dict(e, audio_hz=fa) for e in mine], [(fa, wa, ref)], tol=1)
        c, dec = res[0][0], res[0][1]
        print(f"  {fa}/{fb} Hz {wa}/{wb} wpm SNR {snr:+d} dB: CER {c*100:5.1f}%")
        if args.verbose:
            print(f"     REF {ref[:150]}\n     DEC {dec[:150]}")


def test_multi(args):
    dur = 40 if args.quick else 90
    print(f"\n== Busy channel: 4 and 8 simultaneous signals, jitter 20%, {dur} s ==")
    out = []
    for k, snr, wl in [(4, 0, [14, 22, 30, 38]), (4, 10, [18, 26, 32, 36]),
                       (8, 5, [13, 17, 21, 25, 28, 31, 34, 38])]:
        seed = 4242 + k + snr
        rng = random.Random(seed)
        ts = tones(rng, k, sep=150 if k <= 4 else 140, lo=350, hi=2750)
        sigs = list(zip(ts, wl))
        pcm, refs = scenario(dur, sigs, snr, seed, jitter=0.20, qsb_db=6.0)
        evs, cpu, w, maxtr = run_decoder(pcm)
        res, stray = score(evs, refs)
        print(f"  {k} signals, SNR {snr:+d} dB, 6 dB QSB:  CPU {cpu/dur*100:.2f}% of a core, "
              f"max trackers {maxtr}")
        for (c, dec, ref, ew, es), (tone, wpm, _) in zip(res, refs):
            print(f"     {tone:6.0f} Hz {wpm:4.1f} wpm (est {ew:4.1f}, snr est {es:+5.1f} dB) "
                  f"CER {c*100:5.1f}%")
            if args.verbose:
                print(f"       REF {ref[:120]}\n       DEC {dec[:120]}")
        print(f"     stray text: {sum(len(e['text']) for e in stray)} chars")
        out.append(cpu / dur)
    return out


def _false_run(name, pcm):
    evs, cpu, w, maxtr = run_decoder(pcm)
    nch = sum(len(e["text"]) for e in evs)
    dur = len(pcm) / FS
    sample = " | ".join(f"{e['audio_hz']:.0f}:{e['text']}" for e in evs[:8])
    print(f"  {name:<44s} {dur:5.0f} s  events {len(evs):3d}  chars {nch:4d}  "
          f"CPU {cpu/dur*100:5.2f}%  max trackers {maxtr}" + (f"\n      {sample}" if evs else ""))
    return cpu / dur


def voice_like(n, rng, nrng, harmonic, vib=1.0, f0=120.0):
    """Syllable-modulated speech-like signal."""
    tt = np.arange(n) / FS
    env = np.zeros(n)
    t = 0.0
    while t < n / FS:
        if rng.random() < 0.15:
            t += rng.uniform(0.3, 1.5)             # breath pause
        d = rng.uniform(0.08, 0.35)
        i0, i1 = int(t * FS), int((t + d) * FS)
        env[i0:i1] = rng.uniform(0.3, 1.0)
        t += d + rng.uniform(0.03, 0.2)
    sm = np.hanning(int(0.02 * FS))
    env = np.convolve(env, sm / sm.sum(), mode="same")
    if harmonic:
        f = f0 + vib * (25 * np.sin(2 * np.pi * 0.7 * tt) + 15 * np.sin(2 * np.pi * 2.3 * tt))
        ph = 2 * np.pi * np.cumsum(f) / FS
        src = sum(np.cos(k * ph) / k ** 0.5 for k in range(1, 22))
        x = np.zeros(n)
        for fc, bw in [(700, 150), (1200, 200), (2500, 300)]:
            sos = butter(2, [fc - bw, fc + bw], btype="band", fs=FS, output="sos")
            x += sosfilt(sos, src)
    else:
        sos = butter(4, [300, 2500], btype="band", fs=FS, output="sos")
        x = sosfiltfilt(sos, nrng.normal(0, 1, n))
    return x / (np.std(x) + 1e-12) * env


def test_false(args):
    print("\n== False text (no CW present) ==")
    nrng = np.random.default_rng(5)
    rng = random.Random(5)
    cpu = {}
    dn = 120 if args.quick else 600
    n = int(dn * FS)
    pcm = channel(nrng.normal(0, 1, n), rng)
    cpu["idle"] = _false_run(f"white noise (AGC)", pcm)
    n = int(120 * FS)
    tt = np.arange(n) / FS
    for snr in [20, 0]:
        x = np.cos(2 * np.pi * 1000 * tt) + nrng.normal(0, noise_sigma(snr), n)
        _false_run(f"steady carrier 1000 Hz, SNR {snr} dB", channel(x, rng))
    x = np.cos(2 * np.pi * 700 * tt) * qsb(n, rng, 15, 0.3) + nrng.normal(0, noise_sigma(5), n)
    _false_run("carrier 700 Hz with 15 dB QSB 0.3 Hz", channel(x, rng))
    for snr_v in [20, 5]:
        v = voice_like(n, rng, nrng, False)
        x = v + nrng.normal(0, np.std(v) / 10 ** (snr_v / 20), n)
        _false_run(f"voice-like filtered noise, {snr_v} dB", channel(x, rng))
    for f0, vib in [(120, 1.0), (180, 1.0), (100, 0.5), (140, 0.15), (210, 0.05)]:
        for snr_v in [20, 5]:
            v = voice_like(n, rng, nrng, True, vib, f0)
            x = v + nrng.normal(0, np.std(v) / 10 ** (snr_v / 20), n)
            _false_run(f"voice-like harmonic f0={f0} vibrato x{vib}, {snr_v} dB",
                       channel(x, rng))
    # rtty-ish FSK and a slowly on/off keyed carrier (tuning)
    for baud, snr_r in [(45.45, 10), (45.45, 0), (50, 10), (75, 10)]:
        for rep in range(2):
            sp = int(FS / baud)
            keyed = np.repeat(nrng.integers(0, 2, n // sp + 1), sp)[:n]
            f = np.where(keyed, 1085, 915)
            x = np.cos(2 * np.pi * np.cumsum(f) / FS) + nrng.normal(0, noise_sigma(snr_r), n)
            _false_run(f"RTTY-like FSK {baud} Bd 170 Hz shift, {snr_r} dB #{rep+1}",
                       channel(x, rng))
    k = np.zeros(n)
    t = 0.0
    while t < 120:
        d = rng.uniform(0.5, 4.0)
        k[int(t * FS):int((t + d) * FS)] = 1
        t += d + rng.uniform(0.5, 3.0)
    x = k * np.cos(2 * np.pi * 1500 * tt) + nrng.normal(0, noise_sigma(5), n)
    _false_run("carrier keyed on/off at random 0.5-4 s", channel(x, rng))
    return cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--only", default="false,matrix,qsb,qso,multi")
    args = ap.parse_args()
    only = set(args.only.split(","))
    t0 = time.time()
    cpu_busy = cpu_idle = None
    if "false" in only:
        cpu_idle = test_false(args)["idle"]
    if "matrix" in only:
        test_matrix(args)
    if "qsb" in only:
        test_qsb(args)
    if "qso" in only:
        test_qso(args)
    if "multi" in only:
        cpu_busy = test_multi(args)
    print("\n== CPU (process time per second of audio, one core) ==")
    if cpu_idle is not None:
        print(f"  idle (noise only):      {cpu_idle*1000:6.2f} ms/s  ({cpu_idle*100:.2f}%)")
    if cpu_busy is not None:
        print(f"  busy 4 signals:         {cpu_busy[0]*1000:6.2f} ms/s  ({cpu_busy[0]*100:.2f}%)")
        print(f"  busy 8 signals:         {cpu_busy[-1]*1000:6.2f} ms/s  ({cpu_busy[-1]*100:.2f}%)")
    print(f"\n(total wall time {time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
