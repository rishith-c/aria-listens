"""Audio preprocessing — mel-spectrogram pipeline.

This pipeline must EXACTLY match the JavaScript-side mel-spectrogram
computation in apps/web/src/lib/audio.ts. The browser computes the same
input that the model was trained on, otherwise inference is meaningless.

Constants are exported so the JS side can reference them via codegen.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio.transforms as T

SAMPLE_RATE = 16_000        # 16 kHz mono — covers respiratory acoustics
WINDOW_SECONDS = 1.0        # 1 second analysis window
HOP_SECONDS = 0.1           # 10 Hz prediction rate
N_FFT = 512
HOP_LENGTH = 256            # ~16 ms at 16k → 62 frames per second window
MEL_BINS = 64
F_MIN = 80.0                # below 80Hz is mostly room rumble
F_MAX = 7000.0              # above this rarely carries respiratory info


_mel: T.MelSpectrogram | None = None
_amp_to_db: T.AmplitudeToDB | None = None


def _ensure_pipeline() -> None:
    global _mel, _amp_to_db
    if _mel is None:
        _mel = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=MEL_BINS,
            f_min=F_MIN,
            f_max=F_MAX,
            power=2.0,
        )
        _amp_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)


def compute_mel_spectrogram(waveform: np.ndarray | torch.Tensor) -> torch.Tensor:
    """waveform: shape (n,) float32 audio at SAMPLE_RATE.
    Returns: tensor (MEL_BINS, T) log-mel spectrogram in dB.
    """
    _ensure_pipeline()
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform.astype(np.float32))
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    spec = _mel(waveform)
    db = _amp_to_db(spec)
    # Per-clip min-max normalization to [-1, 1] — matches the JS code.
    db_min = db.amin(dim=(-2, -1), keepdim=True)
    db_max = db.amax(dim=(-2, -1), keepdim=True)
    norm = 2 * (db - db_min) / (db_max - db_min + 1e-6) - 1
    return norm.squeeze(0)
