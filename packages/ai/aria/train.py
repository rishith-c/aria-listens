"""Train Aria. Produces aria.onnx in apps/web/public.

End-to-end run on CPU in ~30-90 seconds:
    python -m aria.train

Uses synthetic dataset by default — see aria.synth. To use a real public
dataset (ESC-50 / FSD50K), point AUDIO_DATA_DIR at extracted WAV files and
the loader will combine them with synth examples.

Output:
  - apps/web/public/aria.onnx       (model weights for browser inference)
  - apps/web/public/aria.meta.json  (preprocessing constants for the JS side)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from aria.audio import (
    F_MAX,
    F_MIN,
    HOP_LENGTH,
    MEL_BINS,
    N_FFT,
    SAMPLE_RATE,
    WINDOW_SECONDS,
    compute_mel_spectrogram,
)
from aria.model import AriaModel, example_input
from aria.synth import generate_dataset


def featurize(waves: np.ndarray) -> torch.Tensor:
    """Convert (N, n_samples) waveforms to (N, 1, MEL_BINS, T) spectrograms."""
    out = []
    for w in waves:
        m = compute_mel_spectrogram(w)
        out.append(m.unsqueeze(0))  # (1, MEL, T)
    return torch.stack(out, dim=0)


def train(out_dir: Path, n_epochs: int = 12, batch_size: int = 64) -> None:
    print("== Aria training ==")
    print(f"   sample_rate={SAMPLE_RATE}  window={WINDOW_SECONDS}s  mel_bins={MEL_BINS}")

    print("[1/4] Generating synthetic dataset...")
    waves, labels = generate_dataset(n_per_class=800, seed=42)
    print(f"   waves shape = {waves.shape}, labels shape = {labels.shape}")

    print("[2/4] Computing mel-spectrograms...")
    X = featurize(waves)
    y = torch.from_numpy(labels)
    print(f"   X = {X.shape}  y = {y.shape}")

    # Train / val split
    perm = torch.randperm(X.size(0), generator=torch.Generator().manual_seed(0))
    X = X[perm]
    y = y[perm]
    n_val = X.size(0) // 5
    X_train, X_val = X[n_val:], X[:n_val]
    y_train, y_val = y[n_val:], y[:n_val]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    print("[3/4] Training CNN...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AriaModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
        acc = correct / len(val_loader.dataset)
        sched.step()
        print(f"   epoch {epoch+1:02d}/{n_epochs}   train_loss={train_loss:.4f}   val_acc={acc:.3%}")

    # Force eval-mode and CPU before export
    model.eval()
    model = model.cpu()

    print("[4/4] Exporting to ONNX...")
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "aria.onnx"
    torch.onnx.export(
        model,
        example_input(),
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"   wrote {onnx_path}  ({onnx_path.stat().st_size // 1024} KB)")

    # Write metadata for the JS preprocessing pipeline
    meta = {
        "sample_rate": SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "mel_bins": MEL_BINS,
        "f_min": F_MIN,
        "f_max": F_MAX,
        "input_shape": [1, 1, MEL_BINS, 63],
        "classes": ["normal", "distress"],
    }
    meta_path = out_dir / "aria.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"   wrote {meta_path}")
    print("DONE.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "apps" / "web" / "public",
        help="Output directory for aria.onnx + aria.meta.json",
    )
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    train(args.out, n_epochs=args.epochs)


if __name__ == "__main__":
    main()
