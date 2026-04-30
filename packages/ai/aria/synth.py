"""Synthetic audio generator for respiratory-distress acoustics.

Real choking sounds are hard to source ethically. This module synthesizes
plausible distress-signal audio by combining narrow-band noise bursts,
sustained breath-like noise, and irregular onset patterns characteristic of
choking, coughing, wheezing, and gasping.

The synthesis is intentionally diverse — random envelopes, jitter, pitch
shifts — so the trained model learns acoustic *patterns* rather than
specific waveforms. Combined with real-world negative examples (silence,
speech, ambient kitchen noise), this is sufficient training data for the
demo classifier.

Distress classes generated:
    - cough      : sharp explosive onset, short decay, mid-frequency burst
    - wheeze     : sustained narrow-band noise, 200-2000 Hz peak
    - gasp       : ascending pitch breath inhale + sharp cutoff
    - sustained  : repeated cough-cough-cough cluster (the choking pattern)

Negative classes:
    - silence     : low-amplitude noise floor
    - speech_like : modulated formants
    - kitchen     : broadband ambient noise (utensils, water)

These are the rough categories of ESC-50 we'd target if downloading; we
synthesize stand-ins that are statistically distinguishable.
"""

from __future__ import annotations

import math
import numpy as np

from aria.audio import SAMPLE_RATE, WINDOW_SECONDS


def _shape_envelope(n: int, attack: float, sustain: float, decay: float) -> np.ndarray:
    """ADSR-ish envelope with arbitrary timing."""
    a = max(1, int(n * attack))
    s = max(1, int(n * sustain))
    d = max(1, n - a - s)
    env = np.concatenate([
        np.linspace(0, 1, a, dtype=np.float32),
        np.ones(s, dtype=np.float32),
        np.linspace(1, 0, d, dtype=np.float32),
    ])
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    return env[:n]


def _bandpass_noise(rng: np.random.Generator, n: int, low: float, high: float) -> np.ndarray:
    """Band-limited white noise via FFT masking."""
    raw = rng.standard_normal(n).astype(np.float32)
    spec = np.fft.rfft(raw)
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    mask = ((freqs >= low) & (freqs <= high)).astype(np.float32)
    spec *= mask
    return np.fft.irfft(spec, n).astype(np.float32)


def synth_cough(rng: np.random.Generator) -> np.ndarray:
    n = int(WINDOW_SECONDS * SAMPLE_RATE)
    out = np.zeros(n, dtype=np.float32)
    n_cough = rng.integers(1, 3)
    for _ in range(n_cough):
        burst_n = int(rng.uniform(0.1, 0.25) * SAMPLE_RATE)
        start = int(rng.uniform(0, n - burst_n))
        center = float(rng.uniform(700, 1800))
        bandwidth = float(rng.uniform(400, 900))
        burst = _bandpass_noise(rng, burst_n, max(80, center - bandwidth), center + bandwidth)
        env = _shape_envelope(burst_n, 0.05, 0.15, 0.80)
        out[start : start + burst_n] += burst * env * float(rng.uniform(0.5, 0.95))
    return out


def synth_wheeze(rng: np.random.Generator) -> np.ndarray:
    n = int(WINDOW_SECONDS * SAMPLE_RATE)
    center = float(rng.uniform(400, 1800))
    bandwidth = float(rng.uniform(80, 250))
    base = _bandpass_noise(rng, n, max(80, center - bandwidth), center + bandwidth)
    # Slow amplitude modulation 4-7 Hz like breath cycle
    mod_hz = float(rng.uniform(3.5, 7.5))
    t = np.arange(n) / SAMPLE_RATE
    am = 0.5 + 0.5 * np.sin(2 * math.pi * mod_hz * t + rng.uniform(0, 6.28))
    env = _shape_envelope(n, 0.05, 0.85, 0.10)
    return (base * am * env * float(rng.uniform(0.4, 0.85))).astype(np.float32)


def synth_gasp(rng: np.random.Generator) -> np.ndarray:
    n = int(WINDOW_SECONDS * SAMPLE_RATE)
    burst_n = int(rng.uniform(0.25, 0.55) * SAMPLE_RATE)
    base = _bandpass_noise(rng, burst_n, 200, 2500)
    # Pitch sweep up — characteristic of an inhale gasp
    t = np.arange(burst_n) / SAMPLE_RATE
    sweep = np.sin(2 * math.pi * (200 + 1800 * t / max(t[-1], 1e-6)) * t)
    env = _shape_envelope(burst_n, 0.4, 0.4, 0.2)
    burst = (0.7 * base + 0.3 * sweep) * env * float(rng.uniform(0.5, 0.9))
    out = np.zeros(n, dtype=np.float32)
    start = int(rng.uniform(0, n - burst_n))
    out[start : start + burst_n] = burst
    return out


def synth_sustained(rng: np.random.Generator) -> np.ndarray:
    """Repeated short bursts — the choking-cluster pattern."""
    n = int(WINDOW_SECONDS * SAMPLE_RATE)
    out = np.zeros(n, dtype=np.float32)
    cursor = 0
    while cursor < n:
        gap = int(rng.uniform(0.05, 0.18) * SAMPLE_RATE)
        cough_n = int(rng.uniform(0.06, 0.15) * SAMPLE_RATE)
        cursor += gap
        if cursor + cough_n > n:
            break
        center = float(rng.uniform(800, 1600))
        burst = _bandpass_noise(rng, cough_n, max(80, center - 500), center + 500)
        env = _shape_envelope(cough_n, 0.05, 0.1, 0.85)
        out[cursor : cursor + cough_n] += burst * env * float(rng.uniform(0.6, 0.95))
        cursor += cough_n
    return out


def synth_silence(rng: np.random.Generator) -> np.ndarray:
    n = int(WINDOW_SECONDS * SAMPLE_RATE)
    return (rng.standard_normal(n) * float(rng.uniform(0.001, 0.01))).astype(np.float32)


def synth_speech_like(rng: np.random.Generator) -> np.ndarray:
    """Crude formant-based speech mimic. Not actual speech, but
    spectrally similar enough to be a good negative."""
    n = int(WINDOW_SECONDS * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    # 3 wandering formants
    out = np.zeros(n, dtype=np.float32)
    for f0 in (rng.uniform(220, 380), rng.uniform(900, 1400), rng.uniform(2200, 2900)):
        wobble = 30 * np.sin(2 * math.pi * rng.uniform(2, 5) * t)
        out += np.sin(2 * math.pi * (f0 + wobble) * t).astype(np.float32) * float(rng.uniform(0.05, 0.15))
    # Voiced/unvoiced amplitude modulation
    mod = (np.sin(2 * math.pi * rng.uniform(3, 6) * t) > 0).astype(np.float32)
    return (out * mod * float(rng.uniform(0.4, 0.8))).astype(np.float32)


def synth_kitchen(rng: np.random.Generator) -> np.ndarray:
    """Broadband ambient noise + occasional clatter pulses."""
    n = int(WINDOW_SECONDS * SAMPLE_RATE)
    out = (rng.standard_normal(n) * 0.03).astype(np.float32)
    n_clatter = rng.integers(0, 3)
    for _ in range(n_clatter):
        start = int(rng.uniform(0, n - 800))
        out[start : start + 800] += rng.standard_normal(800).astype(np.float32) * 0.4
    return out


# Class registry. Distress classes -> label 1; benign -> label 0.
SYNTHESIZERS = {
    1: [synth_cough, synth_wheeze, synth_gasp, synth_sustained],
    0: [synth_silence, synth_speech_like, synth_kitchen],
}


def generate_dataset(
    n_per_class: int = 800, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (waveforms (N, n_samples), labels (N,))."""
    rng = np.random.default_rng(seed)
    waves: list[np.ndarray] = []
    labels: list[int] = []
    for label, synths in SYNTHESIZERS.items():
        for _ in range(n_per_class):
            synth = synths[rng.integers(len(synths))]
            w = synth(rng)
            # Random tiny gain jitter for augmentation
            w = w * float(rng.uniform(0.7, 1.0))
            waves.append(w)
            labels.append(label)
    return np.stack(waves, axis=0), np.array(labels, dtype=np.int64)
