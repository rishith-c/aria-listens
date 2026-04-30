/**
 * MicListener — wraps WebAudio + sliding-window buffer + 10Hz inference.
 *
 * Consumers subscribe to `onResult` and receive a probability stream. The
 * listener also exposes the latest waveform window and the latest
 * spectrogram for visualization without redundant work.
 */

import {
  HOP_LENGTH,
  N_FRAMES,
  SAMPLE_RATE,
  WINDOW_SECONDS,
  computeMelSpectrogram,
  resample,
} from "./audio";
import { classify, ensureSession } from "./inference";

export interface ListenerSnapshot {
  /** Probability of distress in [0, 1] */
  probability: number;
  /** Inference latency ms */
  latencyMs: number;
  /** Wall-clock timestamp ms */
  t: number;
  /** RMS energy of the current window (for vis) */
  rms: number;
  /** Latest waveform window (1 second @ 16k) */
  waveform: Float32Array;
  /** Latest mel-spectrogram (MEL_BINS * N_FRAMES) */
  spectrogram: Float32Array;
}

type Subscriber = (s: ListenerSnapshot) => void;

const TARGET_LENGTH = SAMPLE_RATE * WINDOW_SECONDS; // 16000 samples
const HOP_MS = (HOP_LENGTH / SAMPLE_RATE) * 1000 * 6; // ~96ms — about 10Hz

export class MicListener {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private worklet: AudioWorkletNode | null = null;
  private buffer = new Float32Array(TARGET_LENGTH);
  private subscribers = new Set<Subscriber>();
  private intervalId: number | null = null;
  private inputRate = SAMPLE_RATE;
  private warmingUp = false;
  private running = false;

  async start(): Promise<void> {
    if (this.running) return;
    this.running = true;
    await ensureSession();
    this.warmingUp = false;

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
      video: false,
    });
    this.ctx = new AudioContext({ sampleRate: 48000 });
    this.inputRate = this.ctx.sampleRate;

    // Use a ScriptProcessor for max compatibility — modest, but fine here
    const source = this.ctx.createMediaStreamSource(this.stream);
    const node = this.ctx.createScriptProcessor(2048, 1, 1);
    node.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      const resampled = resample(input, this.inputRate, SAMPLE_RATE);
      // Slide buffer
      const remain = TARGET_LENGTH - resampled.length;
      if (remain > 0) {
        this.buffer.copyWithin(0, resampled.length);
        this.buffer.set(resampled, remain);
      } else {
        this.buffer.set(resampled.subarray(resampled.length - TARGET_LENGTH));
      }
    };
    source.connect(node);
    node.connect(this.ctx.destination);
    // Mute audio output (don't echo mic to speakers)
    const gain = this.ctx.createGain();
    gain.gain.value = 0;
    node.connect(gain).connect(this.ctx.destination);

    this.intervalId = window.setInterval(() => {
      void this.tick();
    }, HOP_MS);
  }

  async stop(): Promise<void> {
    this.running = false;
    if (this.intervalId !== null) {
      window.clearInterval(this.intervalId);
      this.intervalId = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    if (this.ctx) {
      await this.ctx.close();
      this.ctx = null;
    }
  }

  subscribe(fn: Subscriber): () => void {
    this.subscribers.add(fn);
    return () => this.subscribers.delete(fn);
  }

  private async tick(): Promise<void> {
    if (this.warmingUp) return;
    this.warmingUp = true;
    try {
      const wave = this.buffer.slice();
      const spec = computeMelSpectrogram(wave);
      const result = await classify(spec);
      let rms = 0;
      for (let i = 0; i < wave.length; i++) rms += wave[i] * wave[i];
      rms = Math.sqrt(rms / wave.length);
      const snap: ListenerSnapshot = {
        probability: result.probability,
        latencyMs: result.latency,
        t: performance.now(),
        rms,
        waveform: wave,
        spectrogram: spec,
      };
      this.subscribers.forEach((fn) => fn(snap));
    } finally {
      this.warmingUp = false;
    }
  }
}
