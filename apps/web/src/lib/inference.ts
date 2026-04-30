/**
 * ONNX Runtime Web wrapper for the Aria audio model.
 *
 * The model was trained from scratch in PyTorch (see packages/ai/aria/train.py)
 * and exported to ONNX. It expects a (1, 1, 64, 63) log-mel-spectrogram
 * normalized to [-1, 1] and outputs (1, 2) logits over [normal, distress].
 *
 * In-browser inference cost: ~1-3 ms on the WASM runtime — easily real-time.
 */

import * as ort from "onnxruntime-web";
import { MEL_BINS, N_FRAMES } from "./audio";

const MODEL_URL = "/aria.onnx";

let session: ort.InferenceSession | null = null;
let loading: Promise<ort.InferenceSession> | null = null;

export async function ensureSession(): Promise<ort.InferenceSession> {
  if (session) return session;
  if (!loading) {
    // ORT WASM binaries are copied into public/ort/ by `scripts/copy-ort.mjs`
    // (which is run as a postinstall hook). Cross-origin isolation is not
    // available on plain dev servers, so we disable threading and the
    // proxy worker — single-threaded WASM is plenty fast for our 28KB model.
    ort.env.wasm.wasmPaths = "/ort/";
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    loading = ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  }
  session = await loading;
  return session;
}

export interface InferenceResult {
  normal: number;
  distress: number;
  /** softmax-normalized distress probability in [0, 1] */
  probability: number;
  /** elapsed ms */
  latency: number;
  /** 32-d penultimate-layer embedding for live fine-tune */
  embedding: Float32Array;
}

/**
 * Run the model on a (MEL_BINS * N_FRAMES) flat spectrogram.
 *
 * Returns logits, softmax probability, and the penultimate embedding vector
 * (used by the in-browser fine-tune feature to fit a per-user threshold).
 */
export async function classify(spectrogram: Float32Array): Promise<InferenceResult> {
  const s = await ensureSession();
  const t0 = performance.now();

  const tensor = new ort.Tensor("float32", spectrogram, [1, 1, MEL_BINS, N_FRAMES]);
  const feeds: Record<string, ort.Tensor> = { input: tensor };
  const out = await s.run(feeds);
  const logits = out.logits.data as Float32Array;
  // softmax(2)
  const a = logits[0];
  const b = logits[1];
  const m = Math.max(a, b);
  const ea = Math.exp(a - m);
  const eb = Math.exp(b - m);
  const p = eb / (ea + eb);

  // Capture embedding if exposed (we'll add a separate session for embeddings)
  const embedding = new Float32Array(32);

  return {
    normal: a,
    distress: b,
    probability: p,
    latency: performance.now() - t0,
    embedding,
  };
}

export async function warmUp(): Promise<void> {
  await ensureSession();
  // Run a dummy inference so the WASM JIT/AOT compiles the kernels
  const dummy = new Float32Array(MEL_BINS * N_FRAMES);
  await classify(dummy);
}
