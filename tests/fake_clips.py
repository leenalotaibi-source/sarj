"""Generate crude speech-like clips so the pipeline can be tested without data.

Only for plumbing/regression tests: real tuning must use the dev set and
real/TTS Arabic speech.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy import signal

SR = 16000


def syllables(rng, n_syl, f0, sr=SR):
    out = []
    for _ in range(n_syl):
        d = rng.uniform(0.12, 0.25)
        t = np.arange(int(sr * d)) / sr
        f = f0 * (1 + 0.15 * np.sin(2 * np.pi * rng.uniform(1, 4) * t)) * rng.uniform(0.85, 1.15)
        ph = 2 * np.pi * np.cumsum(f) / sr
        x = sum(np.sin(k * ph) / k for k in range(1, 25))
        for fc in (rng.uniform(300, 900), rng.uniform(900, 2400)):
            b, a = signal.iirpeak(fc, 4, fs=sr)
            x = x + 2 * signal.lfilter(b, a, x)
        env = np.sin(np.pi * np.linspace(0, 1, len(t))) ** 0.6
        out.append(x * env)
        out.append(np.zeros(int(sr * rng.uniform(0.01, 0.06))))
    y = np.concatenate(out)
    return (0.1 * y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def phrase_speech(rng, n_phrases, f0):
    parts, words, t = [], [], 0.0
    for p in range(n_phrases):
        for w in range(int(rng.integers(3, 7))):
            x = syllables(rng, int(rng.integers(1, 4)), f0)
            words.append({"word": "كلمة" + ("." if w == 0 and p > 0 else ""), "start": t, "end": t + len(x) / SR})
            parts.append(x)
            t += len(x) / SR
        gap = np.zeros(int(SR * rng.uniform(0.2, 0.45)), np.float32)
        parts.append(gap)
        words[-1]["word"] += "."
        t += len(gap) / SR
    return np.concatenate(parts), words


def main(root):
    rng = np.random.default_rng(0)
    for d in ("agent", "caller/GI", "caller/BC", "caller/BG", "caller/NV", "caller/turn"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    for i in range(8):
        x, words = phrase_speech(rng, int(rng.integers(2, 5)), rng.uniform(170, 220))
        sf.write(f"{root}/agent/a{i}.wav", x, SR)
        json.dump({"text": " ".join(w["word"] for w in words), "words": words},
                  open(f"{root}/agent/a{i}.json", "w", encoding="utf-8"), ensure_ascii=False)
    for i in range(10):
        f0 = rng.uniform(95, 140)
        sf.write(f"{root}/caller/GI/g{i}.wav", syllables(rng, int(rng.integers(6, 14)), f0), SR)
        sf.write(f"{root}/caller/BC/b{i}.wav", syllables(rng, int(rng.integers(1, 3)), f0), SR)
        sf.write(f"{root}/caller/BG/k{i}.wav", syllables(rng, int(rng.integers(8, 20)), rng.uniform(150, 250)), SR)
        sf.write(f"{root}/caller/turn/t{i}.wav", syllables(rng, int(rng.integers(5, 12)), f0), SR)
        n = int(SR * rng.uniform(0.15, 0.35))
        cough = rng.standard_normal(n) * np.exp(-np.arange(n) / (0.06 * SR))
        sf.write(f"{root}/caller/NV/c{i}.wav", (0.1 * cough).astype(np.float32), SR)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fake_clips")
