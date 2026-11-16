"""Audio augmentation for synthetic calls / classifier clips.

Goal: make TTS or studio clips look like the hidden set: narrowband phone
audio, line echo of the agent, room/background noise, level jitter.
All functions take/return float32 mono arrays in [-1, 1].
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def rms_db(x: np.ndarray) -> float:
    return 10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-12)


def set_level(x: np.ndarray, target_db: float) -> np.ndarray:
    return (x * 10 ** ((target_db - rms_db(x)) / 20)).astype(np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = np.gcd(sr_in, sr_out)
    return signal.resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def mulaw(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """G.711-style 8-bit mu-law encode + decode (quantisation noise)."""
    x = np.clip(x, -1, 1)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1) / 2 * mu) / mu * 2 - 1
    return (np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu).astype(np.float32)


def phone_channel(x: np.ndarray, sr: int, rng: np.random.Generator,
                  out_sr: int | None = None, packet_loss: float = 0.003) -> np.ndarray:
    """Band-limit to 300-3400 Hz, go through 8 kHz mu-law, random frame drops."""
    lo = rng.uniform(200, 350)
    hi = rng.uniform(3200, 3500)
    sos = signal.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    y = signal.sosfilt(sos, x)
    y = resample(y, sr, 8000)
    y = mulaw(y)
    if packet_loss > 0:
        n = 160  # 20 ms at 8 kHz
        for i in range(0, len(y) - n, n):
            if rng.random() < packet_loss:
                y[i:i + n] = 0.0
    return resample(y, 8000, out_sr or sr)


def echo_path(agent: np.ndarray, sr: int, rng: np.random.Generator,
              delay_ms: float | None = None, gain_db: float | None = None) -> np.ndarray:
    """Agent audio as it leaks into the caller mic / line (delay, coloured, decaying)."""
    delay_ms = rng.uniform(20, 250) if delay_ms is None else delay_ms
    gain_db = rng.uniform(-35, -8) if gain_db is None else gain_db
    ir_len = int(sr * rng.uniform(0.02, 0.12))
    t = np.arange(ir_len) / sr
    ir = rng.standard_normal(ir_len) * np.exp(-t / rng.uniform(0.01, 0.04))
    ir[0] += 1.0
    ir /= np.sqrt(np.sum(ir ** 2))
    y = signal.fftconvolve(agent, ir)[: len(agent)]
    sos = signal.butter(2, rng.uniform(2500, 3800), btype="low", fs=sr, output="sos")
    y = signal.sosfilt(sos, y)
    if rng.random() < 0.3:  # mild loudspeaker non-linearity
        y = np.tanh(3 * y) / 3
    d = int(sr * delay_ms / 1000)
    y = np.concatenate([np.zeros(d), y])[: len(agent)]
    return (y * 10 ** (gain_db / 20)).astype(np.float32)


def coloured_noise(n: int, rng: np.random.Generator, kind: str | None = None) -> np.ndarray:
    kind = kind or rng.choice(["white", "pink", "brown", "hum"])
    if kind == "white":
        x = rng.standard_normal(n)
    elif kind in ("pink", "brown"):
        f = np.fft.rfftfreq(n)
        f[0] = f[1]
        spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f)))
        spec /= f ** (0.5 if kind == "pink" else 1.0)
        x = np.fft.irfft(spec, n)
    else:  # mains hum + harmonics (fans, AC)
        t = np.arange(n) / 16000
        f0 = rng.choice([50, 60])
        x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
        x = x + 0.1 * rng.standard_normal(n)
    return (x / (np.std(x) + 1e-9)).astype(np.float32)


def add_at_snr(x: np.ndarray, noise: np.ndarray, snr_db: float, ref_db: float | None = None) -> np.ndarray:
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    noise = noise[: len(x)]
    ref = rms_db(x) if ref_db is None else ref_db
    return (x + set_level(noise, ref - snr_db)).astype(np.float32)


def far_field(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Make a clip sound like a distant talker (background speech)."""
    ir_len = int(sr * 0.4)
    t = np.arange(ir_len) / sr
    ir = rng.standard_normal(ir_len) * np.exp(-t / rng.uniform(0.08, 0.2))
    ir[0] = 0.3
    y = signal.fftconvolve(x, ir)[: len(x)]
    sos = signal.butter(2, rng.uniform(1500, 3000), btype="low", fs=sr, output="sos")
    return signal.sosfilt(sos, y).astype(np.float32)
