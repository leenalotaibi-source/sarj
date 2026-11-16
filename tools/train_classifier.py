"""Train the backchannel-vs-interruption prefix classifier.

Replays labelled calls through the *runtime* Decider, collects the prefix
feature vector of every caller speech run it detects while the agent talks,
labels each run from the call labels (STOP event -> 1, KEEP event or
unlabelled run -> 0) and fits a logistic regression. Cross-validation is
grouped by source clip so the same voice/clip never sits in train and test.

    python tools/train_classifier.py synth_out [more_dirs...] --prefix-ms 300
    # then: TT_CFG="use_clf=1" python tools/replay.py heldout_synth

Label format = tools/make_synth_calls.py output (call_XXXX.wav + .json).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tt.config import Config  # noqa: E402
from tt.decider import Decider  # noqa: E402
from tt.prefix import FEATURE_NAMES  # noqa: E402

FRAME_S = 0.02


def collect(wav, prefix_ms):
    x, sr = sf.read(wav, dtype="int16", always_2d=True)
    with open(wav[:-4] + ".json", encoding="utf-8") as f:
        lab = json.load(f)
    n = int(sr * FRAME_S)
    dec = Decider(Config())
    dec.prefix_sink, dec.prefix_ms = [], prefix_ms
    starts = {int(round(u["start_s"] / FRAME_S)): u for u in lab["agent_utts"]}
    ends = {int(round(u["end_s"] / FRAME_S)) for u in lab["agent_utts"]}
    for i in range(len(x) // n):
        if i in ends:
            dec.agent_end()
        if i in starts:
            dec.agent_start({"text": starts[i].get("text", ""), "words": starts[i].get("words", [])})
        fr = x[i * n:(i + 1) * n]
        dec.frame(fr[:, 0], fr[:, 1])
    rows = []
    name = os.path.basename(wav)
    for _, run_start_ms, vec in dec.prefix_sink:
        ts = run_start_ms / 1000.0 - FRAME_S
        ev = next((e for e in lab["events"] if e["onset_s"] - 0.15 <= ts <= e["end_s"]), None)
        if ev is None:
            rows.append((vec, 0, "stray", f"{name}:stray"))
        else:
            rows.append((vec, int(ev["expect"] == "STOP"), ev["cat"], ev.get("src", name)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--prefix-ms", type=float, default=300.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "bc_clf.json"))
    ap.add_argument("--C", type=float, default=0.5)
    args = ap.parse_args()
    if args.prefix_ms % 20:
        raise SystemExit("--prefix-ms must be a multiple of 20")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    rows = []
    for d in args.dirs:
        for wav in sorted(glob.glob(os.path.join(d, "*.wav"))):
            rows += collect(wav, args.prefix_ms)
    X = np.stack([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    cats = np.array([r[2] for r in rows])
    groups = np.array([r[3] for r in rows])
    print(f"runs={len(y)}  positives={y.sum()}  by category={dict(collections.Counter(cats))}")
    if len(set(y)) < 2:
        raise SystemExit("need both STOP and KEEP runs")

    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    oof = np.zeros(len(y))
    k = min(5, len(set(groups)))
    for tr, te in GroupKFold(n_splits=k).split(Z, y, groups):
        m = LogisticRegression(C=args.C, class_weight="balanced", max_iter=1000).fit(Z[tr], y[tr])
        oof[te] = m.predict_proba(Z[te])[:, 1]
    print(f"grouped {k}-fold AUC = {roc_auc_score(y, oof):.3f}")
    for c in sorted(set(cats)):
        sel = cats == c
        print(f"  {c:6s} n={sel.sum():4d}  mean p(interrupt)={oof[sel].mean():.2f}")

    m = LogisticRegression(C=args.C, class_weight="balanced", max_iter=1000).fit(Z, y)
    for nme, w in sorted(zip(FEATURE_NAMES, m.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {nme:22s} {w:+.2f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"features": FEATURE_NAMES, "prefix_ms": args.prefix_ms,
                   "mean": mu.tolist(), "std": sd.tolist(),
                   "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
                   "trained_on": args.dirs, "n": int(len(y))}, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
