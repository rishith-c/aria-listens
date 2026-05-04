# Aria

Real-time choking detection that runs entirely in the browser. Aria listens to a microphone, detects respiratory distress in under one second, and immediately shows the room what to do.

## What it does

Aria is a small audio classifier (about 120,000 parameters, roughly 28 KB as an ONNX model) that takes the last one second of microphone input, converts it to a 64-band log-mel-spectrogram, and outputs a probability that the speaker is in respiratory distress. It runs at 10 Hz in the browser using ONNX Runtime Web on WASM. No audio ever leaves the device.

When the probability crosses the alarm threshold and stays there for about one second, Aria triggers a full-screen emergency cascade: a loud siren, device vibration, a four-step illustrated Heimlich maneuver card, a 30-second countdown before dialing 911, and a pre-drafted SMS to emergency contacts.

What makes this more than a threshold detector is the personal calibration step. The user records 15 seconds of normal eating sounds. Aria collects the model's per-frame distress probabilities during that window, fits a Gaussian (mean and standard deviation), and stores it. After calibration, the alarm requires the probability to deviate at least 2.5 standard deviations above that personal baseline before firing. This dramatically reduces false alarms, especially for elderly users whose normal eating sounds might otherwise trigger the model.

## Why it exists

Choking is the fourth leading cause of accidental death in the United States. It is the top killer of children under 4 and adults over 75. The survival rate is about 95% if a bystander acts within the first 60 seconds, but drops to near zero after four minutes of brain hypoxia. Most bystanders freeze. Aria is designed to be the bystander that does not freeze.

## How the AI pipeline works

### Synthetic data generation

Real choking audio is difficult to source ethically. The `aria.synth` module generates plausible distress audio from acoustic primitives: band-passed noise bursts, ADSR envelopes, breath-rate amplitude modulation, and formant chains. It produces four distress classes (cough, wheeze, gasp, sustained cluster) and three benign classes (silence, speech-like ambient, kitchen noise). The synthesis is intentionally randomized with varied envelopes, jitter, and pitch so the model learns acoustic patterns rather than specific waveforms.

### Training (about 30 seconds on CPU)

The model is a 4-block 2D CNN with BatchNorm, ReLU, and MaxPool, ending in AdaptiveAvgPool and a two-class head (normal vs. distress). It trains with AdamW and a cosine learning rate schedule. Training generates the dataset on the fly, trains the model, and exports it to ONNX.

```bash
cd packages/ai
pip install -e ".[data]"
python -m aria.train
# Produces: apps/web/public/aria.onnx (~28 KB)
# Produces: apps/web/public/aria.meta.json
```

### In-browser inference (1 to 3 ms per frame)

The web app loads the ONNX model on page load using ONNX Runtime Web (WASM backend). A `MicListener` class captures microphone audio at 48 kHz, resamples to 16 kHz, maintains a sliding one-second buffer, computes the mel-spectrogram in JavaScript (a port of the Python preprocessing), and feeds it to the model at about 10 Hz. End-to-end latency from acoustic event to UI alarm is under 100 ms.

### Personal calibration

The calibration panel records 15 seconds of normal sounds, collects the model's per-frame output probabilities, and computes a baseline distribution. The zustand state store then uses this baseline to apply a personalized threshold on top of the global one. This is real on-device personalization of a trained model, not just a slider.

## Quickstart

### Prerequisites

- Python 3.11 or newer (for model training)
- Node 20 or newer, pnpm (for the web app)
- A browser with microphone access (Chrome, Firefox, or Safari)

### 1. Train the model

```bash
cd packages/ai
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[data]"
python -m aria.train
```

This takes about 30 seconds on a CPU. It generates synthetic audio, trains the CNN, and exports the ONNX model directly into the web app's public directory.

### 2. Run the web app

```bash
cd apps/web
pnpm install
pnpm dev
# Open http://localhost:3000
```

Click "Start Listening" and grant microphone permission. The waveform and probability graph will appear in real time.

## Project structure

```
aria-listens/
  apps/
    web/                          Next.js 15 + React 19 + Turbopack
      public/
        aria.onnx                 Trained model (~28 KB)
        aria.meta.json            Preprocessing constants
        ort/                      ONNX Runtime WASM binaries
      src/
        components/
          alarm-overlay.tsx       Full-screen emergency cascade
          heimlich.tsx            4-step illustrated Heimlich card
          waveform.tsx            Live probability and waveform canvas
          calibrate-panel.tsx     Personal baseline UI
        lib/
          audio.ts               WebAudio mel-spectrogram pipeline
          inference.ts           ONNX Runtime Web wrapper
          listener.ts            Mic capture + 10 Hz inference tick
          store.ts               Zustand state + alarm decision logic
  packages/
    ai/                           PyTorch training pipeline
      aria/
        audio.py                 torchaudio mel-spec (JS counterpart)
        model.py                 AriaModel: 4-block 2D CNN
        synth.py                 Synthetic respiratory audio generator
        train.py                 Full pipeline: synth, train, ONNX export
```

## Screenshots / Demo

<!-- Add screenshot: the main listening screen showing the live waveform, probability graph, and green "normal" status -->

<!-- Add screenshot: the alarm state with full-screen red overlay, Heimlich diagram, 911 countdown timer, and SMS draft -->

<!-- Add screenshot: the calibration panel during the 15-second recording, showing probability samples being collected -->

<!-- Add screenshot: the calibration result showing the fitted baseline mean and standard deviation -->

## Honest engineering caveats

This is a hackathon prototype, not a medical device.

- The training data is entirely synthetic. It teaches acoustic patterns, but a production system would use ESC-50, FSD50K, AudioSet, and IRB-approved real choking recordings.
- The alarm threshold and HMM-lite filter are hand-tuned for demo conditions. A real product would learn the operating point from per-population precision/recall curves.
- The ONNX model is small enough to run on a smart-speaker DSP. A production deployment would push inference to dedicated hardware rather than a browser tab.
- The 911 dialing is a `tel:` link. A real product would use platform-native emergency dispatch APIs with PSAP geolocation.

None of these change what the demo shows. All of them are tractable engineering work for a production version.

## Stack

- **Model training:** PyTorch 2.5, torchaudio, ONNX export
- **In-browser AI:** ONNX Runtime Web (WASM)
- **Audio pipeline:** WebAudio API + custom mel-spectrogram (JS port of Python code)
- **UI:** Next.js 15, React 19, Turbopack, shadcn/ui
- **State:** zustand
- **Motion:** framer-motion
- **Emergency UX:** tel:/sms: links, Vibration API, WebAudio siren

## License

MIT
