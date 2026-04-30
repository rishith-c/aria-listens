/**
 * WebAudio mel-spectrogram pipeline — JS counterpart of aria/audio.py.
 *
 * Constants must match the Python file exactly so that what the model was
 * trained on equals what the browser feeds it. We re-derive the mel filter
 * bank in JS from first principles to avoid runtime divergence.
 *
 * Pipeline:
 *   raw mic samples (44.1k or 48k) -> resample to 16k
 *     -> 1-second sliding window
 *     -> STFT (n_fft=512, hop=256)
 *     -> power spectrum
 *     -> mel filterbank (64 bins, 80-7000 Hz)
 *     -> AmplitudeToDB(power, top_db=80)
 *     -> per-clip min-max to [-1, 1]
 *     -> Float32Array(64 * 63)
 */

export const SAMPLE_RATE = 16_000;
export const WINDOW_SECONDS = 1.0;
export const N_FFT = 512;
export const HOP_LENGTH = 256;
export const MEL_BINS = 64;
export const F_MIN = 80;
export const F_MAX = 7000;
export const N_FRAMES = 63; // floor((SAMPLE_RATE * WINDOW_SECONDS - N_FFT) / HOP_LENGTH) + 1

// ---------------------------------------------------------------------------
// Mel filter bank — slaney-style triangular filters
// ---------------------------------------------------------------------------

function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}
function melToHz(mel: number): number {
  return 700 * (10 ** (mel / 2595) - 1);
}

function buildMelFilterBank(): Float32Array {
  // Returns a (MEL_BINS x (N_FFT/2 + 1)) flat row-major matrix
  const nBins = N_FFT / 2 + 1;
  const mat = new Float32Array(MEL_BINS * nBins);

  const melMin = hzToMel(F_MIN);
  const melMax = hzToMel(F_MAX);
  const melPoints = new Float32Array(MEL_BINS + 2);
  for (let i = 0; i < melPoints.length; i++) {
    melPoints[i] = melMin + ((melMax - melMin) * i) / (melPoints.length - 1);
  }
  const hzPoints = melPoints.map((m) => melToHz(m));
  const binPoints = hzPoints.map((hz) => Math.floor((N_FFT * hz) / SAMPLE_RATE));

  for (let m = 0; m < MEL_BINS; m++) {
    const left = binPoints[m];
    const center = binPoints[m + 1];
    const right = binPoints[m + 2];
    for (let k = left; k < center; k++) {
      if (k >= 0 && k < nBins) mat[m * nBins + k] = (k - left) / Math.max(1, center - left);
    }
    for (let k = center; k < right; k++) {
      if (k >= 0 && k < nBins) mat[m * nBins + k] = (right - k) / Math.max(1, right - center);
    }
  }
  return mat;
}

const MEL_FB = buildMelFilterBank();

// ---------------------------------------------------------------------------
// Hann window
// ---------------------------------------------------------------------------

const HANN = (() => {
  const w = new Float32Array(N_FFT);
  for (let i = 0; i < N_FFT; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N_FFT - 1)));
  return w;
})();

// ---------------------------------------------------------------------------
// Iterative radix-2 FFT (real input -> complex spectrum, return power only)
// ---------------------------------------------------------------------------

function fftPower(real: Float32Array): Float32Array {
  const N = real.length; // assumed power of 2 (N_FFT = 512)
  const re = new Float32Array(N);
  const im = new Float32Array(N);
  for (let i = 0; i < N; i++) re[i] = real[i];

  // Bit reversal
  for (let i = 1, j = 0; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }

  for (let size = 2; size <= N; size <<= 1) {
    const half = size >> 1;
    const tableStep = (N / size) | 0;
    for (let i = 0; i < N; i += size) {
      for (let j = i, k = 0; j < i + half; j++, k += tableStep) {
        const angle = (-2 * Math.PI * k) / N;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const tre = re[j + half] * cos - im[j + half] * sin;
        const tim = re[j + half] * sin + im[j + half] * cos;
        re[j + half] = re[j] - tre;
        im[j + half] = im[j] - tim;
        re[j] += tre;
        im[j] += tim;
      }
    }
  }

  // Power = |X|^2, only first N/2+1 bins (real input symmetry)
  const power = new Float32Array(N / 2 + 1);
  for (let k = 0; k < power.length; k++) power[k] = re[k] * re[k] + im[k] * im[k];
  return power;
}

// ---------------------------------------------------------------------------
// Public: mel-spectrogram from a 1-second mono Float32Array @ 16k
// ---------------------------------------------------------------------------

export function computeMelSpectrogram(samples: Float32Array): Float32Array {
  if (samples.length < SAMPLE_RATE * WINDOW_SECONDS) {
    const padded = new Float32Array(SAMPLE_RATE * WINDOW_SECONDS);
    padded.set(samples);
    samples = padded;
  }
  const N = SAMPLE_RATE * WINDOW_SECONDS;
  const nBins = N_FFT / 2 + 1;
  const out = new Float32Array(MEL_BINS * N_FRAMES);

  let frameIdx = 0;
  const buf = new Float32Array(N_FFT);
  for (let start = 0; start + N_FFT <= N && frameIdx < N_FRAMES; start += HOP_LENGTH, frameIdx++) {
    for (let i = 0; i < N_FFT; i++) buf[i] = samples[start + i] * HANN[i];
    const power = fftPower(buf);
    for (let m = 0; m < MEL_BINS; m++) {
      let sum = 0;
      const off = m * nBins;
      for (let k = 0; k < nBins; k++) sum += MEL_FB[off + k] * power[k];
      out[m * N_FRAMES + frameIdx] = sum;
    }
  }

  // Power -> dB, top_db=80
  let dbMax = -Infinity;
  const dbBuf = new Float32Array(out.length);
  for (let i = 0; i < out.length; i++) {
    const p = out[i];
    const db = 10 * Math.log10(Math.max(p, 1e-10));
    dbBuf[i] = db;
    if (db > dbMax) dbMax = db;
  }
  const dbMin = dbMax - 80;
  // Clip + per-clip min-max normalize
  let mn = Infinity;
  let mx = -Infinity;
  for (let i = 0; i < dbBuf.length; i++) {
    const c = Math.max(dbBuf[i], dbMin);
    dbBuf[i] = c;
    if (c < mn) mn = c;
    if (c > mx) mx = c;
  }
  const range = mx - mn + 1e-6;
  for (let i = 0; i < dbBuf.length; i++) dbBuf[i] = (2 * (dbBuf[i] - mn)) / range - 1;
  return dbBuf;
}

// ---------------------------------------------------------------------------
// Resample helper — linear interpolation, sufficient for 16k target
// ---------------------------------------------------------------------------

export function resample(
  input: Float32Array,
  inputRate: number,
  targetRate: number = SAMPLE_RATE,
): Float32Array {
  if (inputRate === targetRate) return input;
  const ratio = inputRate / targetRate;
  const newLength = Math.floor(input.length / ratio);
  const out = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const t = i * ratio;
    const lo = Math.floor(t);
    const hi = Math.min(lo + 1, input.length - 1);
    const frac = t - lo;
    out[i] = input[lo] * (1 - frac) + input[hi] * frac;
  }
  return out;
}
