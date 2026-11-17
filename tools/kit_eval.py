"""Fast offline evaluation with the OFFICIAL kit scorer (no WebSocket, no real time).

Feeds the exact harness stream (ttbench.harness.build_stream) to MySystem
in-process, writes a harness-style log, and scores it with ttbench.score.
Decision times are stamped at the end of the frame that produced them, i.e.
zero processing latency: use `python -m ttbench.play --calls all` (real time)
for numbers that go into the write-up.

Run from the kit root (where ttbench/ and data/ are):

    python tools/kit_eval.py                          # all 24 dev calls
    TT_CFG="stop_after_ms=600" python tools/kit_eval.py --quiet
    python tools/kit_eval.py --sweep stop_after_ms=400,520,650 --sweep hangover_ms=100,140
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ttbench.dataset import lay_out_stream, load_split  # noqa: E402
from ttbench.harness import FRAME_MS, FRAME_SAMPLES, build_stream  # noqa: E402
from ttbench.score import load_labels, score  # noqa: E402

import my_system  # noqa: E402


def run_once(scenarios, labels, manifest, cfg_spec: str = ""):
    os.environ["TT_CFG"] = cfg_spec
    stream, controls = build_stream(scenarios)
    sysm = my_system.MySystem()
    log = []
    n = len(stream) // FRAME_SAMPLES
    for i in range(n):
        for m in controls.get(i, []):
            t = m["type"]
            if t == "scenario_start":
                sysm.on_scenario_start(m.get("context", {}))
            elif t == "agent_utterance_start":
                sysm.on_agent_start(m.get("text", ""), m.get("duration_ms", 0))
            elif t == "agent_utterance_end":
                sysm.on_agent_end()
        fr = stream[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
        d = sysm.on_frame(fr[:, 0], fr[:, 1])
        if d:
            msg = {"type": "decision", "decision": d} if isinstance(d, str) else {"type": "decision", **d}
            log.append({"dir": "rx", "stream_ms": (i + 1) * FRAME_MS, "msg": msg})
    meta = {"scenarios": [{"scenario_id": s.scenario_id, "start_ms": s.start_ms, "duration_ms": s.duration_ms}
                          for s in scenarios]}
    return score(labels, meta, log, manifest.get("thresholds")), log


def summary(res) -> str:
    c = res["core"]
    cats = " ".join(f"{k}:{'-' if v['R'] is None else f'{v[chr(82)]:.2f}'}/{v['S']:.2f}"
                    for k, v in sorted(res["categories"].items()))
    return f"Core={c['core_score']:.1f} R={c['R']:.3f} S={c['S']:.3f} | R/S {cats}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dev")
    ap.add_argument("--calls", default="")
    ap.add_argument("--sweep", action="append", default=[], help="key=v1,v2,...")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    only = [c for c in args.calls.split(",") if c] or None
    labels, manifest = load_labels(Path(args.data))
    scen = lay_out_stream(load_split(Path(args.data), only))
    base = os.environ.get("TT_CFG", "")

    grid = [[]]
    for sw in args.sweep:
        k, vals = sw.split("=", 1)
        grid = [g + [f"{k}={v}"] for g in grid for v in vals.split(",")]
    best = None
    for g in grid:
        spec = ",".join(filter(None, [base] + g))
        res, log = run_once(scen, labels, manifest, spec)
        line = f"{spec or '(defaults)':60s} {summary(res)}"
        print(line, flush=True)
        if best is None or res["core"]["core_score"] > best[0]:
            best = (res["core"]["core_score"], spec, res)
    if len(grid) > 1:
        print(f"BEST {best[0]:.1f}  {best[1]}")
    res = best[2]
    if not args.quiet:
        for e in res["events"]:
            if e.get("credit", 1) < 1:
                print(f"  {e['event_id']:10s} {e['kind']:13s} exp={e['expected']:5s} onset={e['onset_ms']:>6} "
                      f"credit={e['credit']:.2f}  {e.get('detail', '')}")
        for sr in res["scenarios"]:
            for t in sr["spurious_stops"]:
                print(f"  {sr['scenario_id']:10s} spurious stop at {t:.0f} ms")
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
