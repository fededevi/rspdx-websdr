#!/usr/bin/python3
"""FT8/FT4 watcher test-suite.

Messages are encoded to tone sequences with ft8_lib's gen_ft8 (<repo>/local/bin),
synthesised here as GFSK directly at 8 kHz at chosen audio frequencies and
time offsets, mixed with white Gaussian noise, band-limited to 200..2900 Hz
like the receiver's SSB filter, scaled like an AGC to int16 and fed to
FT8Watcher in 800-sample blocks with UTC-aligned wall-clock timestamps that
carry 50..80 ms of random latency.

SNR definition (WSJT-X convention): signal power / noise power in 2500 Hz.

    /usr/bin/python3 tools/digimodes/ft8/test/run_tests.py           # full suite
    /usr/bin/python3 tools/digimodes/ft8/test/run_tests.py --quick

Prints results only; temporary files go to /tmp.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import os
import random
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.signal import firwin, lfilter
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import decoder as D  # noqa: E402

FS = D.FS
GEN = D._find("FT8_GEN_FT8", "gen_ft8")
PARAM = {  # symbol period, BT, tone spacing (= 1/period)
    "FT8": (0.16, 2.0),
    "FT4": (0.048, 1.0),
}
_BPF = firwin(255, [200, 2900], pass_zero=False, fs=FS)
_tone_cache = {}


def tones(msg, mode):
    key = (msg, mode)
    if key not in _tone_cache:
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            args = [GEN, msg, os.path.join(d, "x.wav"), "1000"] + (["-ft4"] if mode == "FT4" else [])
            out = subprocess.run(args, capture_output=True, text=True).stdout
        m = re.search(r"FSK tones: (\d+)", out)
        if not m:
            raise ValueError(f"gen_ft8 cannot encode {msg!r}: {out}")
        _tone_cache[key] = np.array([int(c) for c in m.group(1)])
    return _tone_cache[key]


def gfsk(sym, f0, mode, fs=FS):
    """Phase-continuous GFSK, same shaping as WSJT-X / ft8_lib."""
    T, bt = PARAM[mode]
    nsps = int(round(T * fs))
    k = np.pi * np.sqrt(2 / np.log(2))
    t = np.arange(3 * nsps) / nsps - 1.5
    pulse = (erf(k * bt * (t + 0.5)) - erf(k * bt * (t - 0.5))) / 2
    n = len(sym) * nsps
    dphi = np.zeros(n + 2 * nsps)
    pk = 2 * np.pi / nsps
    for i, s in enumerate(sym):
        dphi[i * nsps:i * nsps + 3 * nsps] += pk * s * pulse
    dphi[:2 * nsps] += pk * sym[0] * pulse[nsps:]
    dphi[n:n + 2 * nsps] += pk * sym[-1] * pulse[:2 * nsps]
    dphi += 2 * np.pi * f0 / fs
    phi = np.cumsum(dphi[nsps:nsps + n]) - dphi[nsps]
    x = np.sin(phi)
    nr = nsps // 8
    env = (1 - np.cos(2 * np.pi * np.arange(nr) / (2 * nr))) / 2
    x[:nr] *= env
    x[-nr:] *= env[::-1]
    return x


# ------------------------------------------------------------------ messages
L1 = "ABCDEFGHIKLMNOPRSTUVWYZ"


def rcall(rng):
    pre = rng.choice(["DL", "K", "W", "G", "F", "I", "EA", "JA", "VK", "OH", "SP", "UA", "PY", "N", "OK", "HB9", "ON", "LZ"])
    if not pre[-1].isdigit():
        pre += str(rng.randint(0, 9))
    return pre + "".join(rng.choice(L1) for _ in range(rng.randint(2, 3)))


def rgrid(rng):
    return (rng.choice("ABCDEFGHIJKLMNOPQR") + rng.choice("ABCDEFGHIJKLMNOPQR")
            + str(rng.randint(0, 9)) + str(rng.randint(0, 9)))


def rmsg(rng):
    a, b = rcall(rng), rcall(rng)
    k = rng.randint(0, 5)
    if k == 0:
        return f"CQ {a} {rgrid(rng)}"
    if k == 1:
        return f"{a} {b} {rgrid(rng)}"
    if k == 2:
        return f"{a} {b} {rng.randint(-24, 10):+03d}"
    if k == 3:
        return f"{a} {b} R{rng.randint(-24, 10):+03d}"
    return f"{a} {b} " + rng.choice(["RR73", "73", "RRR"])


def plan_slot(rng, mode, nsig, snr):
    """nsig signals, frequencies >= 60 Hz (FT8) / 100 Hz (FT4) apart in 300..2700 Hz."""
    bw = 60 if mode == "FT8" else 100
    span = 50 if mode == "FT8" else 85
    grid = np.arange(300, 2700 - span, bw)
    fs = sorted(rng.sample(list(grid), nsig))
    sigs = []
    for f in fs:
        sigs.append({"text": rmsg(rng), "f": float(f + rng.uniform(0, bw - span)),
                     "dt": rng.uniform(-0.6, 1.6) if mode == "FT8" else rng.uniform(-0.4, 0.8),
                     "snr": snr if not isinstance(snr, tuple) else rng.uniform(*snr)})
    return sigs


def render(mode, sigs, rng_np, nsec):
    """Audio float64 for nsec seconds at 8 kHz starting at a slot boundary, noise sigma 1."""
    P = D.MODES[mode]
    n = int(nsec * FS)
    x = rng_np.standard_normal(n + 254)
    for s in sigs:
        slot = s.get("slot", 0)
        a = np.sqrt(2 * 10 ** (s["snr"] / 10) * 2500 / (FS / 2))
        y = a * gfsk(tones(s["text"], mode), s["f"], mode)
        i0 = int(round((slot * P + 0.5 + s["dt"]) * FS)) + 254
        x[i0:i0 + y.size] += y[:max(0, x.size - i0)]
    return lfilter(_BPF, 1, x)[254:]


def agc_int16(x):
    return np.clip(x * (3000 / np.std(x)), -32767, 32767).astype(np.int16)


# ------------------------------------------------------------------ driver
def stream(w, pcm, t0, rng, drop=None, wait_each_slot=True):
    """Feed pcm (int16, starts at UTC t0) in 800-sample blocks with jittered latency.
    drop: list of (start_s, end_s) relative to t0 that are not delivered at all."""
    events, feed_t = [], []
    for i in range(0, pcm.size, 800):
        t = t0 + i / FS
        if drop and any(a <= i / FS < b for a, b in drop):
            continue
        ts = t + 0.05 + rng.uniform(0, 0.03)
        c = time.perf_counter()
        events += w.feed(pcm[i:i + 800], ts)
        feed_t.append(time.perf_counter() - c)
        if wait_each_slot and w.busy():
            while w.busy():
                time.sleep(0.01)
    events += w.feed(np.zeros(0, np.int16), t0 + pcm.size / FS + 1.0)  # nothing new; just in case
    events += w.flush()
    return events, feed_t


def score(sigs, events, mode, t0):
    P = D.MODES[mode]
    sent = {}
    for s in sigs:
        sent[(round(t0 + s.get("slot", 0) * P, 2), s["text"])] = s
    ok, bad_meta, false, ap = 0, 0, [], 0
    got = set()
    for e in events:
        if e["mode"] != mode:
            continue
        key = (round(e["slot_ts"], 2), e["text"])
        s = sent.get(key)
        if s is None:
            false.append(e)
            continue
        if key in got:
            continue
        got.add(key)
        ok += 1
        ap += bool(e.get("ap"))
        if abs(e["audio_hz"] - s["f"]) > 8 or abs(e["dt"] - s["dt"] - 0.065) > 0.25:
            bad_meta += 1
    return ok, len(sent), false, bad_meta, ap


def utc_base(P):
    return np.floor(time.time() / P) * P - 3600 * P   # an aligned UTC time (past), realistic magnitude


def run_snr(mode, backend, snrs, slots, nsig, seed=1, label=""):
    rows = []
    for snr in snrs:
        rng = random.Random(seed * 1000 + abs(int(snr * 10)))
        rng_np = np.random.default_rng(seed * 1000 + abs(int(snr * 10)))
        P = D.MODES[mode]
        sigs = []
        for k in range(slots):
            for s in plan_slot(rng, mode, nsig, snr):
                s["slot"] = k + 1           # slot 0 is a partial lead-in, ignored
                sigs.append(s)
        lead = 4.3
        x = render(mode, sigs, rng_np, (slots + 1) * P + 0.5)
        pcm = agc_int16(x)[int((P - lead) * FS):]     # start mid-slot 0
        t0 = utc_base(P)
        w = D.FT8Watcher(label, modes=(mode,), backend=backend, max_pending=100)
        ev, ft = stream(w, pcm, t0 - lead, rng)
        ok, n, false, meta, ap = score(sigs, ev, mode, t0 - P)
        st = w.stats
        rows.append((snr, ok, n, len(false), meta, ap, st["cpu_s"] / max(1, st["slots"]),
                     st["wall_s"] / max(1, st["slots"]), 1e3 * max(ft), 1e6 * np.mean(ft)))
        for e in false[:3]:
            print(f"      false: {e}")
        w.close()
    return rows


def print_rows(title, rows):
    print(f"\n{title}")
    print("   SNR dB | decoded/sent |   %  | false | bad f/dt | via AP | CPU s/slot | wall s/slot | feed max ms / mean us")
    for snr, ok, n, nf, meta, ap, cpu, wall, fmax, fmean in rows:
        print(f"   {snr:6.1f} | {ok:5d}/{n:<6d} | {100 * ok / n:4.0f} | {nf:5d} | {meta:8d} | {ap:6d} | "
              f"{cpu:10.2f} | {wall:11.2f} | {fmax:6.1f} / {fmean:5.0f}")


def run_noise(mode, backend, seconds, ap=True):
    rng = random.Random(7)
    rng_np = np.random.default_rng(7 + len(mode))
    P = D.MODES[mode]
    pcm = agc_int16(render(mode, [], rng_np, seconds + P))
    t0 = utc_base(P)
    w = D.FT8Watcher("noise", modes=(mode,), backend=backend, ap=ap, max_pending=100)
    ev, _ = stream(w, pcm, t0, rng)
    st = w.stats
    w.close()
    return ev, st


def run_gap(backend):
    """Strong signals in slots 1..4; audio lost: 3 s inside slot 1 (before the
    signals, which start at 0.5+dt), all of slot 2, the end of slot 3 + start of slot 4.
    Checks slot_ts / dt alignment survives and nothing is decoded in slot 2."""
    mode, P = "FT8", 15.0
    rng = random.Random(3)
    rng_np = np.random.default_rng(3)
    sigs = []
    for k in (1, 2, 3, 4, 5):
        for s in plan_slot(rng, mode, 4, -8):
            s["slot"] = k
            s["dt"] = rng.uniform(0.0, 0.5)
            sigs.append(s)
    x = agc_int16(render(mode, sigs, rng_np, 6 * P + 0.5))
    t0 = utc_base(P)
    # slot1: drop 0.2..0.5 s? (keep message) -> drop 14.0..14.6 s (after message end, ~13.6 s)
    # slot2: dropped completely; slot3: end dropped 13.9..15; slot4: drop first 0.3 s
    drop = [(P + 14.0, P + 14.6), (2 * P, 3 * P), (3 * P + 13.9, 4 * P + 0.3)]
    w = D.FT8Watcher("gap", modes=(mode,), backend=backend, max_pending=100)
    ev, _ = stream(w, x, t0, rng, drop=drop)
    w.close()
    expect = [s for s in sigs if s["slot"] != 2]
    ok, n, false, meta, _ = score(expect, ev, mode, t0)
    slot2 = [e for e in ev if abs(e["slot_ts"] - (t0 + 2 * P)) < 0.1]
    return ok, n, false, meta, slot2, w.stats


def run_crowded(backend, nsig=30, snr=(-20, 5)):
    rng = random.Random(11)
    rng_np = np.random.default_rng(11)
    mode, P = "FT8", 15.0
    sigs = []
    for k in (1, 2):
        # crowded band: signals may overlap in frequency (random 200..2800 Hz)
        for _ in range(nsig):
            sigs.append({"text": rmsg(rng), "f": rng.uniform(250, 2750), "dt": rng.uniform(-0.5, 1.2),
                         "snr": rng.uniform(*snr), "slot": k})
    x = agc_int16(render(mode, sigs, rng_np, 3 * P + 0.5))
    t0 = utc_base(P)
    w = D.FT8Watcher("crowd", modes=(mode,), backend=backend, max_pending=100)
    ev, _ = stream(w, x[int(10 * FS):], t0 + 10, rng)
    ok, n, false, meta, ap = score(sigs, ev, mode, t0)
    st = w.stats
    w.close()
    hi = [s for s in sigs if s["snr"] >= -15]
    got = {(round(e["slot_ts"], 2), e["text"]) for e in ev}
    ok_hi = sum((round(t0 + s["slot"] * P, 2), s["text"]) in got for s in hi)
    return ok, n, len(false), ok_hi, len(hi), st


def run_realtime(backend, seconds=33):
    """Real-time paced stream with live time.time() stamps, FT8+FT4 watcher, no waiting:
    events must come back from later feed() calls while feed() stays fast."""
    rng = random.Random(5)
    rng_np = np.random.default_rng(5)
    now = time.time()
    k0 = int(np.ceil(now / 15.0))           # first FT8 slot boundary ahead
    lead = k0 * 15.0 - now
    sigs = []
    for k in (1, 2):
        for s in plan_slot(rng, "FT8", 6, -15):
            s["slot"] = k
            sigs.append(s)
    x = agc_int16(render("FT8", sigs, rng_np, 3 * 15.0 + 0.5))
    pcm = x[int((15.0 - lead) * FS):int((15.0 - lead + seconds + lead) * FS)]
    w = D.FT8Watcher("rt", modes=("FT8", "FT4"), backend=backend)
    ev, ft, t_start = [], [], time.time()
    for i in range(0, pcm.size, 800):
        target = t_start + (i + 800) / FS
        while time.time() < target:
            time.sleep(0.005)
        c = time.perf_counter()
        ev += [dict(e, t_got=time.time()) for e in w.feed(pcm[i:i + 800], time.time() - 0.1)]
        ft.append(time.perf_counter() - c)
    ok, n, false, meta, _ = score([s for s in sigs if (s["slot"] + 1) * 15 <= seconds + lead], ev,
                                  "FT8", (k0 - 1) * 15.0)
    lat = [e["t_got"] - e["slot_ts"] - D.MODES[e["mode"]] for e in ev]
    st = w.stats
    w.close()
    return ok, n, len(false), meta, max(ft) * 1e3, (max(lat) if lat else float("nan")), st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--backend", default="all", help="jt9 | ft8_lib | all")
    a = ap.parse_args()
    print(f"jt9: {D.JT9}\ndecode_ft8: {D.DECODE_FT8}\ngen_ft8: {GEN}\ncpus: {os.cpu_count()}")
    backends = [b for b in ("jt9", "ft8_lib") if a.backend in ("all", b)
                and (D.JT9 if b == "jt9" else D.DECODE_FT8)]
    slots = 2 if a.quick else 5
    snrs8 = [-10, -15, -18, -20, -21, -22, -24] if not a.quick else [-10, -18, -21]
    snrs4 = [-10, -14, -16, -17, -18] if not a.quick else [-10, -16]
    noise_s = 30 if a.quick else 120
    for b in backends:
        print(f"\n================ backend {b} ================")
        print_rows(f"FT8, {slots} slots x 8 signals per SNR (300..2700 Hz, dt -0.6..+1.6 s)",
                   run_snr("FT8", b, snrs8, slots, 8))
        print_rows(f"FT4, {2 * slots} slots x 8 signals per SNR (dt -0.4..+0.8 s)",
                   run_snr("FT4", b, snrs4, 2 * slots, 8))
        for mode in ("FT8", "FT4"):
            ev, st = run_noise(mode, b, noise_s)
            napf = sum(bool(e.get("ap")) for e in ev)
            print(f"\n{mode} pure noise {noise_s} s: {st['slots']} slots decoded, {len(ev)} false decodes "
                  f"({napf} AP-assisted), CPU {st['cpu_s'] / max(1, st['slots']):.2f} s/slot")
            for e in ev:
                print(f"      false: {e}")
        ok, n, false, meta, slot2, st = run_gap(b)
        print(f"\nGap test (lost audio incl. one full slot): {ok}/{n} decoded, {len(false)} false, "
              f"{meta} with wrong f/dt, {len(slot2)} decodes in the lost slot, "
              f"resyncs={st['resyncs']} skipped={st['skipped']}")
        if not a.quick and b == backends[0]:
            ok, n, nf, meta, fmax, lat, st = run_realtime(b)
            print(f"\nReal-time paced (live time.time() stamps, FT8+FT4 watcher): {ok}/{n} FT8 decoded "
                  f"from feed() returns, {nf} false, {meta} wrong f/dt, feed max {fmax:.1f} ms, "
                  f"events at most {lat:.2f} s after slot end, slots={st['slots']} dropped={st['dropped']}")
        if not a.quick:
            ok, n, nf, okh, nh, st = run_crowded(b)
            print(f"\nCrowded band: 2 slots x 30 overlapping signals, SNR uniform -20..+5 dB: "
                  f"{ok}/{n} decoded ({okh}/{nh} of those >= -15 dB), {nf} false, "
                  f"CPU {st['cpu_s'] / max(1, st['slots']):.2f} s/slot, wall {st['wall_s'] / max(1, st['slots']):.2f} s/slot")


if __name__ == "__main__":
    main()
