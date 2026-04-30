"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  Ear,
  Mic,
  Play,
  Square,
  ZapOff,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

import { AlarmOverlay } from "@/components/alarm-overlay";
import { Waveform } from "@/components/waveform";
import { CalibratePanel } from "@/components/calibrate-panel";

import { useAriaStore } from "@/lib/store";
import { MicListener } from "@/lib/listener";
import { warmUp } from "@/lib/inference";

const fadeUp = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] as const },
};

export default function AriaPage() {
  const [listening, setListening] = useState<boolean>(false);
  const [modelReady, setModelReady] = useState<boolean>(false);
  const [calibrating, setCalibrating] = useState<boolean>(false);

  const listenerRef = useRef<MicListener | null>(null);
  const calibrationProbsRef = useRef<number[]>([]);

  const status = useAriaStore((s) => s.status);
  const probability = useAriaStore((s) => s.probability);
  const latencyMs = useAriaStore((s) => s.latencyMs);
  const rms = useAriaStore((s) => s.rms);
  const baselineMean = useAriaStore((s) => s.baselineMean);
  const setStatus = useAriaStore((s) => s.setStatus);
  const push = useAriaStore((s) => s.push);
  const reset = useAriaStore((s) => s.reset);

  // Warm the ONNX session up front so the first inference doesn't stall
  useEffect(() => {
    warmUp().then(() => setModelReady(true)).catch(() => setModelReady(false));
  }, []);

  const startListening = async (): Promise<void> => {
    if (listening) return;
    const listener = new MicListener();
    listenerRef.current = listener;
    listener.subscribe((s) => {
      // Make latest waveform available to the canvas without zustand churn
      (window as unknown as { __ariaWave?: Float32Array }).__ariaWave = s.waveform;
      push(s.t, s.probability, s.rms, s.latencyMs);
      if (calibrating) calibrationProbsRef.current.push(s.probability);
    });
    await listener.start();
    setListening(true);
    setStatus("watching");
  };

  const stopListening = async (): Promise<void> => {
    if (!listening) return;
    await listenerRef.current?.stop();
    listenerRef.current = null;
    setListening(false);
    setStatus("idle");
  };

  const collectCalibration = async (): Promise<number[]> => {
    if (!listening) await startListening();
    setCalibrating(true);
    calibrationProbsRef.current = [];
    await new Promise<void>((resolve) =>
      window.setTimeout(resolve, CalibratePanel.RECORD_SECONDS * 1000),
    );
    setCalibrating(false);
    return calibrationProbsRef.current.slice();
  };

  const cancelCalibration = (): void => {
    setCalibrating(false);
    calibrationProbsRef.current = [];
  };

  const triggerDemo = async (): Promise<void> => {
    // Synthesize a brief simulated detection event for offline-friendly demos
    if (!listening) await startListening();
    const now = performance.now();
    for (let i = 0; i < 12; i++) push(now + i * 100, 0.95, 0.4, 2);
  };

  const statusColor =
    status === "alarm"
      ? "bg-destructive shadow-[0_0_18px_hsl(var(--destructive))]"
      : status === "warning"
        ? "bg-[hsl(var(--warn-amber))] shadow-[0_0_14px_hsl(var(--warn-amber))]"
        : status === "watching"
          ? "bg-accent shadow-[0_0_10px_hsl(var(--accent))]"
          : "bg-foreground/30";

  return (
    <TooltipProvider delayDuration={200}>
      <main className="relative h-screen w-screen overflow-hidden bg-background">
        <div className="grid-bg pointer-events-none absolute inset-0 z-0" />

        {/* Hero / top bar */}
        <motion.header
          {...fadeUp}
          className="pointer-events-none absolute inset-x-0 top-0 z-20 flex items-center justify-between px-8 py-6"
        >
          <div className="flex items-center gap-3">
            <span className={`inline-flex h-1.5 w-1.5 rounded-full ${statusColor}`} />
            <span className="mono micro tabular text-foreground/70">
              aria / detector-001
            </span>
          </div>
          <div className="mono flex items-center gap-6 text-[11px] text-foreground/60 tabular">
            <span>
              <span className="micro mr-2">model</span>
              {modelReady ? "ready" : "loading..."}
            </span>
            <span>
              <span className="micro mr-2">latency</span>
              {latencyMs.toFixed(1)} ms
            </span>
            <span>
              <span className="micro mr-2">rms</span>
              {rms.toFixed(3)}
            </span>
          </div>
        </motion.header>

        {/* Centered editorial hero */}
        <motion.section
          {...fadeUp}
          transition={{ ...fadeUp.transition, delay: 0.08 }}
          className="pointer-events-none absolute left-1/2 top-[14%] z-20 -translate-x-1/2 text-center"
        >
          <div className="mono micro mb-4 flex items-center justify-center gap-2 text-foreground/55">
            <span className="h-px w-8 bg-foreground/20" />
            ai that listens for the sound of life
            <span className="h-px w-8 bg-foreground/20" />
          </div>
          <h1 className="display text-7xl leading-none tracking-tight md:text-8xl">
            Aria
          </h1>
          <p className="mx-auto mt-6 max-w-md text-balance text-[13px] leading-relaxed text-foreground/65">
            5,000 Americans choke to death each year. The first 60 seconds
            decide everything. Aria catches respiratory distress in less than
            one — and tells the room what to do.
          </p>
        </motion.section>

        {/* Center stage — the live signal */}
        <motion.section
          {...fadeUp}
          transition={{ ...fadeUp.transition, delay: 0.18 }}
          className="pointer-events-auto absolute left-1/2 top-[42%] z-20 w-[42rem] -translate-x-1/2"
        >
          <div className="surface rounded-md p-5">
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Ear className="h-3 w-3 text-foreground/60" />
                <span className="mono micro text-foreground/75">
                  live signal
                </span>
              </div>
              <ProbDisplay probability={probability} status={status} />
            </div>
            <Waveform />
          </div>
        </motion.section>

        {/* Lower-left: capture controls */}
        <motion.aside
          {...fadeUp}
          transition={{ ...fadeUp.transition, delay: 0.26 }}
          className="pointer-events-auto absolute bottom-10 left-8 z-20 w-[20rem] space-y-4"
        >
          <div className="surface rounded-md">
            <div className="flex items-center justify-between border-b border-border/60 px-4 py-3">
              <div className="flex items-center gap-2">
                <Mic className="h-3 w-3 text-foreground/60" />
                <span className="mono micro text-foreground/75">
                  capture · 16 khz mono
                </span>
              </div>
              <span className={`mono micro ${listening ? "text-accent" : "text-foreground/40"}`}>
                {listening ? "live" : "off"}
              </span>
            </div>
            <div className="space-y-3 p-4">
              <div className="grid grid-cols-2 gap-2">
                <Button
                  onClick={listening ? stopListening : startListening}
                  disabled={!modelReady}
                  className={`mono h-9 rounded-sm text-[11px] tracking-wider ${
                    listening
                      ? "bg-foreground text-background hover:bg-foreground/90"
                      : "bg-accent text-accent-foreground hover:bg-accent/90"
                  }`}
                >
                  {listening ? <Square className="h-3 w-3" /> : <Play className="h-3 w-3" />}
                  {listening ? "stop" : "start listening"}
                </Button>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      onClick={triggerDemo}
                      variant="outline"
                      className="mono h-9 rounded-sm border-border/60 bg-transparent text-[11px] tracking-wider text-foreground/70 hover:bg-foreground/5"
                    >
                      <ZapOff className="h-3 w-3" /> simulate
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="top" className="mono text-[10px]">
                    Inject a fake high-confidence event to test the alarm UI
                    without making a real choking sound
                  </TooltipContent>
                </Tooltip>
              </div>

              {!modelReady && (
                <div className="space-y-1">
                  <div className="mono micro text-foreground/55">
                    loading model
                  </div>
                  <Skeleton className="h-2 rounded-none bg-foreground/10" />
                </div>
              )}
            </div>
            <Separator className="bg-border/40" />
            <div className="grid grid-cols-3 divide-x divide-border/40">
              <Stat label="status" value={status} />
              <Stat label="prob" value={probability.toFixed(2)} />
              <Stat label="rms" value={rms.toFixed(3)} />
            </div>
          </div>

          <div className="surface rounded-md p-4">
            <CalibratePanel
              collecting={calibrating}
              onCollect={collectCalibration}
              onCancel={cancelCalibration}
            />
          </div>
        </motion.aside>

        {/* Lower-right: model card */}
        <motion.aside
          {...fadeUp}
          transition={{ ...fadeUp.transition, delay: 0.34 }}
          className="pointer-events-auto absolute bottom-10 right-8 z-20 w-[22rem]"
        >
          <div className="surface rounded-md">
            <div className="flex items-center justify-between border-b border-border/60 px-4 py-3">
              <div className="flex items-center gap-2">
                <Activity className="h-3 w-3 text-foreground/60" />
                <span className="mono micro text-foreground/75">
                  model card
                </span>
              </div>
              <span className="mono micro text-foreground/40">aria · v0.1</span>
            </div>
            <div className="space-y-3 p-4 text-[11px]">
              <ModelRow k="architecture" v="4-block 2D-CNN, 120k params" />
              <ModelRow k="input" v="64-mel × 63 frames @ 16kHz" />
              <ModelRow k="output" v="2-class · normal / distress" />
              <ModelRow k="trained on" v="synthetic respiratory + ambient" />
              <ModelRow k="runtime" v="onnx-runtime-web · wasm" />
              <ModelRow
                k="personal threshold"
                v={
                  baselineMean !== null
                    ? `μ=${baselineMean.toFixed(2)} · 2.5σ guard`
                    : "generic thresholds (uncalibrated)"
                }
              />
            </div>
          </div>
        </motion.aside>

        {/* Centered status banner if alarming via simulate */}
        {status === "warning" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="pointer-events-none absolute left-1/2 top-[64%] z-30 -translate-x-1/2"
          >
            <div className="flex items-center gap-2 rounded-sm border border-[hsl(var(--warn-amber))]/60 bg-[hsl(var(--warn-amber))]/10 px-4 py-2 backdrop-blur-md">
              <AlertTriangle className="h-4 w-4 text-[hsl(var(--warn-amber))]" />
              <span className="mono text-[11px] tracking-wider text-foreground">
                early warning · monitoring closely
              </span>
            </div>
          </motion.div>
        )}

        <motion.footer
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6, duration: 0.8 }}
          className="pointer-events-none absolute inset-x-0 bottom-3 z-20 text-center"
        >
          <p className="mono micro text-foreground/35">
            custom-trained pytorch model · 100% on-device inference · no audio
            ever leaves your device
          </p>
        </motion.footer>

        {/* Full-screen alarm cascade */}
        <AlarmOverlay />
      </main>
    </TooltipProvider>
  );
}

interface ProbDisplayProps {
  probability: number;
  status: string;
}
function ProbDisplay({ probability, status }: ProbDisplayProps) {
  const color =
    status === "alarm"
      ? "text-destructive"
      : status === "warning"
        ? "text-[hsl(var(--warn-amber))]"
        : status === "watching"
          ? "text-accent"
          : "text-foreground/40";
  return (
    <div className="flex items-baseline gap-2">
      <span className="mono micro text-foreground/45">distress</span>
      <span className={`display tabular text-3xl leading-none ${color}`}>
        {(probability * 100).toFixed(0)}%
      </span>
    </div>
  );
}

interface StatProps {
  label: string;
  value: string | number;
}
function Stat({ label, value }: StatProps) {
  return (
    <div className="px-3 py-3">
      <div className="mono micro mb-1 text-foreground/45">{label}</div>
      <div className="mono tabular text-[12px] text-foreground/85">{value}</div>
    </div>
  );
}

interface ModelRowProps {
  k: string;
  v: string;
}
function ModelRow({ k, v }: ModelRowProps) {
  return (
    <div className="flex items-baseline gap-3 border-b border-border/30 pb-2 last:border-b-0 last:pb-0">
      <span className="mono micro w-28 shrink-0 text-foreground/45">{k}</span>
      <span className="text-foreground/80">{v}</span>
    </div>
  );
}
