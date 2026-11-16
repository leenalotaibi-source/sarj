"""Assemble synthetic stereo calls (ch0 = agent, ch1 = caller mic) with labels.

Clip layout (any TTS / SADA cuts / consented recordings, any sample rate):

    clips/
      agent/*.wav            agent utterances; optional sidecar <name>.json:
                             {"text": "...", "words": [{"word": "..", "start": s, "end": s}]}
      caller/<CAT>/*.wav     caller events, one folder per category (GI, BC, BG, NV, ...)
                             optional sidecar <name>.json: {"evident_s": 0.8}
      caller/turn/*.wav      (optional) caller's normal turns between agent turns
      noise/*.wav            (optional) real noise; otherwise generated

Each category is STOP or KEEP (see DEFAULT_EXPECT / --expect CAT=STOP).
Output per call: call_XXXX.wav (stereo, --sr) + call_XXXX.json (labels).

    python tools/make_synth_calls.py --clips clips --out synth_out --n 200 --seed 1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import augment as A  # noqa: E402

DEFAULT_EXPECT = {"GI": "STOP", "BC": "KEEP", "BG": "KEEP", "NV": "KEEP"}
# How each category is mixed in: level relative to the caller's speaking level
LEVEL_DB = {"GI": (0, 4), "BC": (-5, 4), "BG": (-18, 5), "NV": (-2, 5)}
FAR_FIELD = {"BG"}


def load(path: str, sr: int) -> np.ndarray:
    x, s = sf.read(path, dtype="float32", always_2d=True)
    return A.resample(x.mean(axis=1), s, sr)


def sidecar(path: str) -> dict:
    j = os.path.splitext(path)[0] + ".json"
    if os.path.exists(j):
        with open(j, encoding="utf-8") as f:
            return json.load(f)
    return {}


class Library:
    def __init__(self, root: str, sr: int):
        self.sr = sr
        self.agent = sorted(glob.glob(os.path.join(root, "agent", "*.wav")))
        self.cats = {}
        for d in sorted(glob.glob(os.path.join(root, "caller", "*"))):
            files = sorted(glob.glob(os.path.join(d, "*.wav")))
            if files:
                self.cats[os.path.basename(d)] = files
        self.turns = self.cats.pop("turn", [])
        self.noise = sorted(glob.glob(os.path.join(root, "noise", "*.wav")))
        self._cache = {}
        if not self.agent or not self.cats:
            raise SystemExit("need clips/agent/*.wav and at least one clips/caller/<CAT>/")

    def get(self, path):
        if path not in self._cache:
            self._cache[path] = load(path, self.sr)
        return self._cache[path]


def phrase_ends(meta: dict) -> list:
    ends = []
    words = meta.get("words") or []
    for i, w in enumerate(words):
        txt = str(w.get("word", w.get("text", ""))).strip()
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (float(nxt["start"]) - float(w["end"])) if nxt else 0
        if txt[-1:] in ".!?؟،," or gap > 0.25:
            ends.append(float(w["end"]))
    return ends


def make_call(lib: Library, rng: np.random.Generator, args, expect: dict) -> tuple:
    sr = lib.sr
    n_utts = int(rng.integers(args.min_utts, args.max_utts + 1))
    caller_db = rng.uniform(-32, -20)
    agent_parts, caller_parts, utts, events = [], [], [], []
    t = 0

    def pad(n):
        agent_parts.append(np.zeros(n, np.float32))
        caller_parts.append(np.zeros(n, np.float32))

    pad(int(sr * rng.uniform(0.3, 0.8)))
    t = sum(len(p) for p in agent_parts)
    for u in range(n_utts):
        # caller turn before the agent speaks (gives a near-field level reference)
        if lib.turns and (u == 0 or rng.random() < 0.6):
            ct = A.set_level(lib.get(lib.turns[rng.integers(len(lib.turns))]), caller_db)
            agent_parts.append(np.zeros(len(ct), np.float32))
            caller_parts.append(ct)
            t += len(ct)
            g = int(sr * rng.uniform(0.2, 0.7))
            pad(g)
            t += g

        apath = lib.agent[rng.integers(len(lib.agent))]
        meta = sidecar(apath)
        agent = A.set_level(lib.get(apath), rng.uniform(-24, -18)).copy()
        caller = np.zeros(len(agent), np.float32)
        ev = None
        if rng.random() < args.event_prob:
            cat = list(lib.cats)[rng.integers(len(lib.cats))]
            cpath = lib.cats[cat][rng.integers(len(lib.cats[cat]))]
            clip = lib.get(cpath)
            cmeta = sidecar(cpath)
            if cat in FAR_FIELD:
                clip = A.far_field(clip, sr, rng)
            mu, sd = LEVEL_DB.get(cat, (0, 4))
            clip = A.set_level(clip, caller_db + rng.normal(mu, sd))
            ends = [e for e in phrase_ends(meta) if e * sr < len(agent) - sr * 0.3]
            if cat == "BC" and ends and rng.random() < 0.7:
                onset = int(sr * (rng.choice(ends) + rng.uniform(0.0, 0.35)))
            else:
                hi = max(int(0.4 * sr) + 1, len(agent) - int(0.4 * sr))
                onset = int(rng.integers(int(0.4 * sr), hi))
            exp = expect.get(cat, "KEEP")
            evident = float(cmeta.get("evident_s", min(len(clip) / sr, args.default_evident_s)))
            if exp == "STOP":  # a real agent would stop after a while
                cut = onset + int(sr * (evident + rng.uniform(0.3, 1.2)))
                if cut < len(agent):
                    fade = min(int(0.05 * sr), len(agent) - cut)
                    agent[cut:cut + fade] *= np.linspace(1, 0, fade)
                    agent = agent[:cut + fade]
            end = onset + len(clip)
            if end > len(caller):
                caller = np.concatenate([caller, np.zeros(end - len(caller), np.float32)])
            caller[onset:end] += clip
            ev = {"cat": cat, "expect": exp, "utt": u,
                  "onset_s": (t + onset) / sr, "end_s": (t + end) / sr,
                  "evident_s": (t + onset) / sr + evident, "src": os.path.basename(cpath)}
            events.append(ev)
        agent_len = len(agent)  # the agent utterance ends here (maybe cut by a STOP)
        L = max(len(agent), len(caller))
        agent = np.pad(agent, (0, L - len(agent)))
        caller = np.pad(caller, (0, L - len(caller)))
        utts.append({"utt": u, "start_s": t / sr, "end_s": (t + agent_len) / sr,
                     "text": meta.get("text", ""), "words": meta.get("words", []),
                     "src": os.path.basename(apath)})
        agent_parts.append(agent)
        caller_parts.append(caller)
        t += len(agent)
        g = int(sr * rng.uniform(0.3, 1.0))
        pad(g)
        t += g

    agent = np.concatenate(agent_parts)
    caller = np.concatenate(caller_parts)
    aug = {}
    if rng.random() < args.echo_prob:
        aug["echo_gain_db"] = float(rng.uniform(-35, -8))
        aug["echo_delay_ms"] = float(rng.uniform(20, 250))
        caller = caller + A.echo_path(agent, sr, rng, aug["echo_delay_ms"], aug["echo_gain_db"])
    if lib.noise and rng.random() < 0.5:
        noise = lib.get(lib.noise[rng.integers(len(lib.noise))])
        aug["noise"] = "file"
    else:
        noise = A.coloured_noise(len(caller), rng)
        aug["noise"] = "synthetic"
    aug["snr_db"] = float(rng.uniform(8, 35))
    caller = A.add_at_snr(caller, noise, aug["snr_db"], ref_db=caller_db)
    if rng.random() < args.phone_frac:
        caller = A.phone_channel(caller, sr, rng)
        aug["phone"] = True
    caller = caller * 10 ** (rng.uniform(-4, 4) / 20)
    caller = np.pad(caller, (0, max(0, len(agent) - len(caller))))[: len(agent)]
    stereo = np.stack([agent, caller], axis=1)
    peak = np.abs(stereo).max()
    if peak > 0.99:
        stereo *= 0.99 / peak
    labels = {"sr": sr, "agent_utts": utts, "events": events, "aug": aug}
    return stereo.astype(np.float32), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", default="synth_out")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-utts", type=int, default=3)
    ap.add_argument("--max-utts", type=int, default=6)
    ap.add_argument("--event-prob", type=float, default=0.8)
    ap.add_argument("--echo-prob", type=float, default=0.7)
    ap.add_argument("--phone-frac", type=float, default=0.5)
    ap.add_argument("--default-evident-s", type=float, default=0.6)
    ap.add_argument("--expect", action="append", default=[], help="CAT=STOP|KEEP")
    args = ap.parse_args()
    expect = dict(DEFAULT_EXPECT)
    for e in args.expect:
        k, v = e.split("=")
        expect[k] = v.upper()
    rng = np.random.default_rng(args.seed)
    lib = Library(args.clips, args.sr)
    os.makedirs(args.out, exist_ok=True)
    for i in range(args.n):
        stereo, labels = make_call(lib, rng, args, expect)
        base = os.path.join(args.out, f"call_{i:04d}")
        sf.write(base + ".wav", stereo, args.sr, subtype="PCM_16")
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=1)
    print(f"wrote {args.n} calls to {args.out}")


if __name__ == "__main__":
    main()
