"use client";

import { useEffect, useRef } from "react";
import { useAriaStore } from "@/lib/store";

/**
 * Confidence-over-time strip plus an instantaneous waveform.
 *
 * The strip on top is the model's distress probability over the last ~20s
 * (10 Hz history). The bottom is the raw waveform of the latest 1-second
 * window. Both render via canvas for low-overhead 60fps redraws.
 */
export function Waveform() {
  const probCanvas = useRef<HTMLCanvasElement>(null);
  const waveCanvas = useRef<HTMLCanvasElement>(null);
  const history = useAriaStore((s) => s.history);
  const status = useAriaStore((s) => s.status);

  useEffect(() => {
    const c = probCanvas.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth * dpr;
    const H = c.clientHeight * dpr;
    if (c.width !== W) c.width = W;
    if (c.height !== H) c.height = H;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, W, H);

    // Gridlines
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    [0.25, 0.5, 0.75].forEach((p) => {
      ctx.beginPath();
      ctx.moveTo(0, H * (1 - p));
      ctx.lineTo(W, H * (1 - p));
      ctx.stroke();
    });

    // Threshold line (warning)
    ctx.strokeStyle = "rgba(255, 200, 80, 0.35)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, H * (1 - 0.7));
    ctx.lineTo(W, H * (1 - 0.7));
    ctx.stroke();

    // Threshold line (alarm)
    ctx.strokeStyle = "rgba(255, 80, 80, 0.5)";
    ctx.beginPath();
    ctx.moveTo(0, H * (1 - 0.85));
    ctx.lineTo(W, H * (1 - 0.85));
    ctx.stroke();
    ctx.setLineDash([]);

    if (history.length < 2) return;

    // Fill under the curve
    const xStep = W / Math.max(history.length - 1, 1);
    ctx.fillStyle = "rgba(125, 235, 245, 0.10)";
    ctx.beginPath();
    ctx.moveTo(0, H);
    history.forEach((p, i) => ctx.lineTo(i * xStep, H * (1 - p.p)));
    ctx.lineTo(W, H);
    ctx.closePath();
    ctx.fill();

    // Stroke
    ctx.strokeStyle = "rgba(125, 235, 245, 0.95)";
    ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath();
    history.forEach((p, i) => {
      const x = i * xStep;
      const y = H * (1 - p.p);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }, [history, status]);

  // Latest waveform render — pulled directly off the most recent snapshot.
  // We don't keep waveform in zustand (heavy); instead we read from a window
  // global that the listener writes after each tick.
  useEffect(() => {
    const c = waveCanvas.current;
    if (!c) return;
    let raf = 0;
    const dpr = window.devicePixelRatio || 1;
    const draw = () => {
      const W = c.clientWidth * dpr;
      const H = c.clientHeight * dpr;
      if (c.width !== W) c.width = W;
      if (c.height !== H) c.height = H;
      const ctx = c.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, W, H);
      const wave = (window as unknown as { __ariaWave?: Float32Array }).__ariaWave;
      if (wave) {
        ctx.strokeStyle = "rgba(255,255,255,0.55)";
        ctx.lineWidth = 1 * dpr;
        ctx.beginPath();
        const step = wave.length / W;
        for (let x = 0; x < W; x++) {
          const idx = Math.floor(x * step);
          const v = wave[idx] || 0;
          const y = H / 2 + v * H * 0.45;
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div className="space-y-3">
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <div className="mono micro text-foreground/55">
            distress probability · 20s
          </div>
          <div className="mono micro text-foreground/40">10 Hz</div>
        </div>
        <canvas ref={probCanvas} className="h-24 w-full rounded-sm bg-black/40" />
      </div>
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <div className="mono micro text-foreground/55">live waveform · 1s</div>
          <div className="mono micro text-foreground/40">16 kHz</div>
        </div>
        <canvas ref={waveCanvas} className="h-16 w-full rounded-sm bg-black/40" />
      </div>
    </div>
  );
}
