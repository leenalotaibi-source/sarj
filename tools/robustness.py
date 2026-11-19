"""Robustness check: the 24 dev calls with the CALLER line degraded, official scorer, offline.

Run from the kit root:   python /path/to/repo/tools/robustness.py [--json out.json]

Conditions (caller channel only; the agent channel is the system's clean reference and stays as is):
    as is · 6 dB quieter · 6 dB louder · narrowband phone (300-3400 Hz, 8 kHz mu-law)
    · pink/brown noise at 20 and 10 dB SNR · babble (3 other callers, far field) at 15 dB
    · phone + quieter + noise 20 dB
Both the kit baseline (B1, any sound) and our system (with the words layer when the model is present).
SNR is measured against the caller's own speech (active frames), not the whole channel.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from augment import add_at_snr, coloured_noise, far_field, phone_channel  # noqa: E402
from ttbench.baselines import AnySound  # noqa: E402
from ttbench.dataset import lay_out_stream, load_split  # noqa: E402
from ttbench.harness import FRAME_MS, FRAME_SAMPLES, build_stream  # noqa: E402
from ttbench.score import load_labels, score  # noqa: E402

import my_system  # noqa: E402
from tt.words import WordsWorker  # noqa: E402

SR = 16000


def speech_db(x: np.ndarray) -> float:
    """Level of the active part (frames within 25 dB of the loudest)."""
    f = x[: len(x) // 320 * 320].reshape(-1, 320).astype(np.float64)
    e = 10 * np.log10(np.mean(f ** 2, axis=1) + 1e-12)
    act = e[e > e.max() - 25]
    return float(10 * np.log10(np.mean(10 ** (act / 10))))


def degrade(caller: np.ndarray, cond: str, rng: np.random.Generator, others: list[np.ndarray]) -> np.ndarray:
    x = caller.astype(np.float32) / 32768.0
    ref = speech_db(x)
    if "quiet" in cond:
        x = x * 10 ** (-6 / 20)
        ref -= 6
    if "loud" in cond:
        x = np.clip(x * 10 ** (6 / 20), -1, 1)
        ref += 6
    if "phone" in cond:
        x = phone_channel(x, SR, rng)[: len(x)]
        x = np.pad(x, (0, max(0, len(caller) - len(x))))
    if "noise20" in cond or "noise10" in cond:
        snr = 20 if "noise20" in cond else 10
        x = add_at_snr(x, coloured_noise(len(x), rng), snr, ref_db=ref)
    if "babble" in cond:
        b = np.zeros(len(x), dtype=np.float32)
        for o in others:
            o = o.astype(np.float32) / 32768.0
            s = rng.integers(0, max(1, len(o) - len(x))) if len(o) > len(x) else 0
            seg = np.resize(o[s:], len(x))
            b += far_field(seg, SR, rng)[: len(x)]
        x = add_at_snr(x, b, 15, ref_db=ref)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def run_system(make, scen, stream, controls, labels, manifest, policy=False):
    sysm = make()
    log = []
    for i in range(len(stream) // FRAME_SAMPLES):
        for m in controls.get(i, []):
            if policy:
                for d in sysm.on_control(m):
                    log.append({"dir": "rx", "stream_ms": (i + 1) * FRAME_MS, "msg": {"type": "decision", "decision": d}})
                continue
            t = m["type"]
            if t == "scenario_start":
                sysm.on_scenario_start(m.get("context", {}))
            elif t == "agent_utterance_start":
                sysm.on_agent_start(m.get("text", ""), m.get("duration_ms", 0))
            elif t == "agent_utterance_end":
                sysm.on_agent_end()
        fr = stream[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
        d = sysm.on_frame(fr[:, 0], fr[:, 1])
        for x in (d if isinstance(d, list) else [d]):
            if x:
                log.append({"dir": "rx", "stream_ms": (i + 1) * FRAME_MS, "msg": {"type": "decision", "decision": x}})
    meta = {"scenarios": [{"scenario_id": s.scenario_id, "start_ms": s.start_ms, "duration_ms": s.duration_ms}
                          for s in scen]}
    return score(labels, meta, log, manifest.get("thresholds"))


def ours():
    m = my_system.MySystem()
    if m.words is not None:
        m.words = WordsWorker(m.words.backend, sync=True)
        m.dec.words = m.words
    return m


CONDS = [("As is", ""), ("6 dB quieter", "quiet"), ("6 dB louder", "loud"), ("Narrowband phone", "phone"),
         ("Noise, SNR 20 dB", "noise20"), ("Noise, SNR 10 dB", "noise10"), ("Babble, 15 dB", "babble"),
         ("Phone + quieter + noise 20 dB", "phone quiet noise20")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--only", default="", help="comma list of condition labels")
    a = ap.parse_args()
    labels, manifest = load_labels(Path("data/dev"))
    scen = lay_out_stream(load_split(Path("data/dev")))
    stream0, controls = build_stream(scen)
    callers = [stream0[s.start_ms * 16: (s.start_ms + s.duration_ms) * 16, 1].copy() for s in scen]
    out = []
    for name, cond in CONDS:
        if a.only and name not in a.only.split(","):
            continue
        rng = np.random.default_rng(7)
        stream = stream0.copy()
        if cond:
            for k, s in enumerate(scen):
                others = [callers[(k + j) % len(scen)] for j in (5, 11, 17)]
                seg = slice(s.start_ms * 16, s.start_ms * 16 + len(callers[k]))
                stream[seg, 1] = degrade(callers[k], cond, rng, others)
        rb = run_system(AnySound, scen, stream, controls, labels, manifest, policy=True)
        ro = run_system(ours, scen, stream, controls, labels, manifest)
        row = [name, round(rb["core"]["core_score"], 1), round(ro["core"]["core_score"], 1),
               round(ro["core"]["R"], 3), round(ro["core"]["S"], 3)]
        out.append(row)
        print(f"{name:32s} baseline {row[1]:5.1f}   ours {row[2]:5.1f} (R {row[3]:.2f} S {row[4]:.2f})", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
