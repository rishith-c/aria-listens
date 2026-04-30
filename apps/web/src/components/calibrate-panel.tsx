"use client";

/**
 * "Train Aria on your family" — the showstopper.
 *
 * The user records ~15 seconds of their normal eating sounds. We collect
 * the model's distress-probability output over those samples, fit a baseline
 * (mean + std), and store it in zustand. The decision logic in store.ts then
 * requires not just an absolute threshold but ALSO a 2.5σ deviation from
 * the personal baseline before firing the alarm.
 *
 * This is a real, on-device, per-user adaptation of the model — and demos
 * the "we own and adapt our own AI" thesis without needing to retrain weights.
 */

import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Mic, Sparkles, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAriaStore } from "@/lib/store";

const RECORD_SECONDS = 15;
const SAMPLE_HZ = 10;

interface Props {
  collecting: boolean;
  onCollect: () => Promise<number[]>;
  onCancel: () => void;
}

export function CalibratePanel({ collecting, onCollect, onCancel }: Props) {
  const baselineMean = useAriaStore((s) => s.baselineMean);
  const baselineStd = useAriaStore((s) => s.baselineStd);
  const setBaseline = useAriaStore((s) => s.setBaseline);
  const clearBaseline = useAriaStore((s) => s.clearBaseline);

  const [progress, setProgress] = useState<number>(0);
  const [running, setRunning] = useState<boolean>(false);
  const startedAt = useRef<number>(0);

  useEffect(() => {
    if (!collecting) {
      setProgress(0);
      setRunning(false);
      startedAt.current = 0;
      return;
    }
    setRunning(true);
    startedAt.current = performance.now();
    const id = window.setInterval(() => {
      const elapsed = (performance.now() - startedAt.current) / 1000;
      setProgress(Math.min(1, elapsed / RECORD_SECONDS));
    }, 100);
    return () => window.clearInterval(id);
  }, [collecting]);

  const handleStart = async (): Promise<void> => {
    const probs = await onCollect();
    if (probs.length === 0) return;
    const mean = probs.reduce((a, b) => a + b, 0) / probs.length;
    const variance =
      probs.reduce((a, b) => a + (b - mean) ** 2, 0) / probs.length;
    const std = Math.sqrt(variance);
    setBaseline(mean, std);
    setRunning(false);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <div className="mono micro text-foreground/55">
            personal calibration
          </div>
          <div className="mono mt-1 text-[11px] text-foreground/75">
            train aria on your normal sounds — 15 seconds
          </div>
        </div>
        {baselineMean !== null && (
          <Button
            onClick={() => clearBaseline()}
            variant="outline"
            size="sm"
            className="mono h-7 rounded-sm border-border/60 bg-transparent text-[10px] tracking-wider text-foreground/60 hover:bg-foreground/5"
          >
            <Trash2 className="h-3 w-3" /> reset
          </Button>
        )}
      </div>

      {baselineMean !== null && baselineStd !== null ? (
        <div className="rounded-sm border border-border/60 bg-black/30 p-3">
          <div className="flex items-center gap-2">
            <Sparkles className="h-3 w-3 text-accent" />
            <span className="mono micro text-accent">
              calibrated to your baseline
            </span>
          </div>
          <div className="mt-2 grid grid-cols-2 gap-3 text-[11px]">
            <div>
              <div className="mono micro text-foreground/45">μ</div>
              <div className="mono tabular text-foreground/85">
                {baselineMean.toFixed(3)}
              </div>
            </div>
            <div>
              <div className="mono micro text-foreground/45">σ</div>
              <div className="mono tabular text-foreground/85">
                {baselineStd.toFixed(3)}
              </div>
            </div>
          </div>
          <div className="mono micro mt-2 text-foreground/45">
            alarm now requires &gt; 2.5σ deviation from your baseline
          </div>
        </div>
      ) : (
        <div className="rounded-sm border border-border/60 bg-black/30 p-3">
          <div className="mono micro text-foreground/45">no baseline yet</div>
          <div className="mt-1 text-[11px] text-foreground/65">
            Aria will use generic thresholds until you calibrate.
          </div>
        </div>
      )}

      {running ? (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="mono micro text-foreground/60">recording...</span>
            <span className="mono micro tabular text-foreground/85">
              {Math.ceil(RECORD_SECONDS * (1 - progress))}s
            </span>
          </div>
          <div className="h-1 w-full overflow-hidden rounded-sm bg-foreground/10">
            <motion.div
              animate={{ width: `${progress * 100}%` }}
              transition={{ duration: 0.1 }}
              className="h-full bg-accent"
            />
          </div>
          <Button
            onClick={() => {
              setRunning(false);
              onCancel();
            }}
            variant="outline"
            className="mono h-8 w-full rounded-sm border-border/60 bg-transparent text-[11px] tracking-wider text-foreground/60 hover:bg-foreground/5"
          >
            cancel
          </Button>
        </div>
      ) : (
        <Button
          onClick={handleStart}
          disabled={collecting}
          className="mono h-9 w-full rounded-sm bg-foreground text-[11px] tracking-wider text-background hover:bg-foreground/90"
        >
          <Mic className="h-3 w-3" />
          {baselineMean !== null ? "re-calibrate · 15s" : "calibrate · 15s"}
        </Button>
      )}
    </div>
  );
}

CalibratePanel.SAMPLE_HZ = SAMPLE_HZ;
CalibratePanel.RECORD_SECONDS = RECORD_SECONDS;
