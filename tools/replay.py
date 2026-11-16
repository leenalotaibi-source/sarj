"""Offline replay of MySystem on our own synthetic calls + an APPROXIMATE score.

This is NOT the official scorer (use `python -m ttbench.play` for that). It
mirrors SCORING.md as described in the brief, so we can measure on our own
synthetic / SADA-based calls that the kit does not have.

    python tools/replay.py synth_out            # all calls in a folder
    TT_CFG="stop_after_ms=450" python tools/replay.py synth_out --fails 10
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from my_system import MySystem  # noqa: E402

FRAME_S = 0.02


def stop_credit(t_stop, ev):
    if t_stop is None:
        return 0.0
    lat = t_stop - ev["evident_s"]
    if lat <= 0.3:
        return 1.0
    if lat <= 1.5:
        return 1.0 - 0.6 * (lat - 0.3) / 1.2
    return 0.0


def run_call(wav_path, label_path):
    x, sr = sf.read(wav_path, dtype="int16", always_2d=True)
    with open(label_path, encoding="utf-8") as f:
        lab = json.load(f)
    n = int(sr * FRAME_S)
    sysm = MySystem()
    utts = lab["agent_utts"]
    starts = {int(round(u["start_s"] / FRAME_S)): u for u in utts}
    ends = {int(round(u["end_s"] / FRAME_S)): u for u in utts}
    actions = []
    worst = 0.0
    cur = None
    for i in range(len(x) // n):
        if i in ends and cur is not None:
            sysm.on_agent_end()
            cur = None
        if i in starts:
            u = starts[i]
            words = [{"word": w.get("word", ""), "start": w["start"], "end": w["end"]}
                     for w in u.get("words", [])]
            sysm.on_agent_start({"text": u.get("text", ""), "words": words})
            cur = u
        fr = x[i * n:(i + 1) * n]
        t0 = time.perf_counter()
        a = sysm.on_frame(fr[:, 0], fr[:, 1])
        worst = max(worst, time.perf_counter() - t0)
        if a:
            actions.append(((i + 1) * FRAME_S, a, cur["utt"] if cur else None))
    return lab, actions, worst


def score(results):
    per_cat = collections.defaultdict(list)
    fails = []
    for name, lab, actions in results:
        used = set()
        for ev in lab["events"]:
            stops = [(k, t) for k, (t, a, u) in enumerate(actions)
                     if a == "STOP" and u == ev["utt"] and ev["onset_s"] - 0.1 <= t <= ev["end_s"] + 0.5]
            if ev["expect"] == "STOP":
                t_stop = stops[0][1] if stops else None
                c = stop_credit(t_stop, ev)
                used.update(k for k, _ in stops)
            else:
                if not stops:
                    c = 1.0
                else:
                    k, t = stops[0]
                    used.update(k for k, _ in stops)
                    res = [t2 for t2, a2, _ in actions[k + 1:k + 3] if a2 == "RESUME"]
                    c = 0.5 if res and res[0] - t <= 0.5 else 0.0
            per_cat[(ev["expect"], ev["cat"])].append(c)
            if c < 1.0:
                fails.append((c, name, ev, [a for a in actions if a[2] == ev["utt"]]))
        stray = [a for k, a in enumerate(actions) if a[1] == "STOP" and k not in used]
        n_quiet = max(1, len(lab["agent_utts"]) - len(lab["events"]))
        per_cat[("KEEP", "stray")].extend([0.0] * len(stray) + [1.0] * max(0, n_quiet - len(stray)))
        for a in stray:
            fails.append((0.0, name, {"cat": "stray", "expect": "KEEP", "t": a[0]}, [a]))
    R = [np.mean(v) for (e, _), v in per_cat.items() if e == "STOP"]
    S = [np.mean(v) for (e, _), v in per_cat.items() if e == "KEEP"]
    R = float(np.mean(R)) if R else 0.0
    S = float(np.mean(S)) if S else 0.0
    F = 2 * R * S / (R + S) if R + S else 0.0
    return per_cat, R, S, F, sorted(fails, key=lambda f: f[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--fails", type=int, default=5)
    args = ap.parse_args()
    results, worst = [], 0.0
    for wav in sorted(glob.glob(os.path.join(args.folder, "*.wav"))):
        lab, actions, w = run_call(wav, wav[:-4] + ".json")
        worst = max(worst, w)
        results.append((os.path.basename(wav), lab, actions))
    per_cat, R, S, F, fails = score(results)
    print(f"calls={len(results)}  worst frame={worst*1000:.1f} ms")
    for (e, c), v in sorted(per_cat.items()):
        print(f"  {e:4s} {c:6s} n={len(v):4d}  credit={np.mean(v):.3f}")
    print(f"R={R:.3f}  S={S:.3f}  F={F:.3f}   (approx; official scorer is ttbench)")
    for c, name, ev, acts in fails[: args.fails]:
        print(f"  FAIL {c:.2f} {name} {ev}  actions={acts}")


if __name__ == "__main__":
    main()
