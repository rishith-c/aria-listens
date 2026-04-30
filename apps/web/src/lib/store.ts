"use client";

import { create } from "zustand";

export type AlarmStatus = "idle" | "watching" | "warning" | "alarm" | "cancelled";

interface AriaState {
  status: AlarmStatus;
  probability: number;
  latencyMs: number;
  rms: number;
  /** Rolling history of {t, p, rms}; oldest first. */
  history: { t: number; p: number; rms: number }[];
  /** Per-user calibration (set by the live fine-tune flow). */
  baselineMean: number | null;
  baselineStd: number | null;

  setStatus: (s: AlarmStatus) => void;
  push: (t: number, p: number, rms: number, latency: number) => void;
  setBaseline: (mean: number, std: number) => void;
  clearBaseline: () => void;
  reset: () => void;
}

const HISTORY_MAX = 200; // ~20 seconds at 10Hz

export const useAriaStore = create<AriaState>((set, get) => ({
  status: "idle",
  probability: 0,
  latencyMs: 0,
  rms: 0,
  history: [],
  baselineMean: null,
  baselineStd: null,

  setStatus: (s) => set({ status: s }),

  push: (t, p, rms, latencyMs) => {
    const { history, status, baselineMean, baselineStd } = get();
    const next = [...history, { t, p, rms }];
    if (next.length > HISTORY_MAX) next.shift();

    // Decision logic:
    //   - WARNING when probability > 0.7 sustained for 0.5 sec
    //   - ALARM   when probability > 0.85 sustained for 1 sec
    //   - Skip if user has cancelled
    let newStatus = status;
    if (status !== "cancelled" && status !== "alarm") {
      const recent = next.slice(-10); // ~1 sec
      const sustainedHi = recent.filter((r) => r.p > 0.85).length >= 6;
      const sustainedMid = recent.slice(-5).filter((r) => r.p > 0.7).length >= 3;

      // Personalized threshold: if calibrated, also require deviation from baseline
      let personalOk = true;
      if (baselineMean !== null && baselineStd !== null) {
        const devs = recent.map((r) => (r.p - baselineMean) / Math.max(baselineStd, 0.01));
        const sustainedDev = devs.filter((d) => d > 2.5).length >= 4;
        personalOk = sustainedDev;
      }

      if (sustainedHi && personalOk) newStatus = "alarm";
      else if (sustainedMid) newStatus = "warning";
      else if (status === "warning") newStatus = "watching";
    }

    set({
      probability: p,
      latencyMs,
      rms,
      history: next,
      status: newStatus,
    });
  },

  setBaseline: (mean, std) => set({ baselineMean: mean, baselineStd: std }),
  clearBaseline: () => set({ baselineMean: null, baselineStd: null }),

  reset: () =>
    set({
      status: "idle",
      probability: 0,
      latencyMs: 0,
      rms: 0,
      history: [],
    }),
}));
