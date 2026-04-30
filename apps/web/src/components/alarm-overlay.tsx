"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertOctagon, PhoneCall, MessageSquare, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Heimlich } from "./heimlich";
import { useAriaStore } from "@/lib/store";

const COUNTDOWN_SECONDS = 30;

export function AlarmOverlay() {
  const status = useAriaStore((s) => s.status);
  const setStatus = useAriaStore((s) => s.setStatus);
  const [secondsLeft, setSecondsLeft] = useState<number>(COUNTDOWN_SECONDS);
  const [calling, setCalling] = useState<boolean>(false);

  useEffect(() => {
    if (status !== "alarm") {
      setSecondsLeft(COUNTDOWN_SECONDS);
      setCalling(false);
      return;
    }
    const t = window.setInterval(() => {
      setSecondsLeft((v) => {
        if (v <= 1) {
          window.clearInterval(t);
          setCalling(true);
          return 0;
        }
        return v - 1;
      });
    }, 1000);
    return () => window.clearInterval(t);
  }, [status]);

  // Trigger device vibration in waves while alarming
  useEffect(() => {
    if (status !== "alarm" || typeof navigator === "undefined") return;
    if (!("vibrate" in navigator)) return;
    const id = window.setInterval(() => {
      navigator.vibrate?.([400, 100, 400, 100, 400]);
    }, 1500);
    return () => window.clearInterval(id);
  }, [status]);

  // Audible alarm via WebAudio (don't depend on a file)
  useEffect(() => {
    if (status !== "alarm") return;
    const ctx = new AudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain).connect(ctx.destination);
    gain.gain.value = 0.0001;
    osc.frequency.value = 880;
    osc.type = "square";
    osc.start();

    let on = true;
    const id = window.setInterval(() => {
      on = !on;
      gain.gain.setTargetAtTime(on ? 0.18 : 0.0001, ctx.currentTime, 0.01);
      osc.frequency.value = on ? 1240 : 880;
    }, 220);
    return () => {
      window.clearInterval(id);
      try {
        osc.stop();
      } catch {
        // already stopped
      }
      ctx.close();
    };
  }, [status]);

  const dial911 = (): void => {
    window.location.href = "tel:911";
  };

  const sendSms = (): void => {
    const body = encodeURIComponent(
      "Aria detected a possible choking event at home. Calling 911 now. Please come immediately if nearby.",
    );
    window.location.href = `sms:?&body=${body}`;
  };

  const cancel = (): void => {
    setStatus("cancelled");
  };

  if (status !== "alarm") return null;

  return (
    <AnimatePresence>
      <motion.div
        key="alarm"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.2 }}
        className="fixed inset-0 z-[100] flex flex-col"
        role="alertdialog"
        aria-modal
      >
        {/* Strobe red backdrop */}
        <div className="absolute inset-0 alarm-pulse" aria-hidden />

        <div className="relative z-10 flex flex-col flex-1 px-8 py-6 text-foreground">
          {/* Top bar */}
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <AlertOctagon className="h-9 w-9" />
              <div>
                <div className="display text-3xl leading-none">CHOKING DETECTED</div>
                <div className="mono micro mt-1 text-foreground/75">
                  aria has identified respiratory distress · act now
                </div>
              </div>
            </div>
            <Button
              onClick={cancel}
              variant="outline"
              className="mono h-9 rounded-sm border-foreground/40 bg-black/30 text-[11px] tracking-wider text-foreground hover:bg-black/50"
            >
              <X className="h-3 w-3" /> false alarm
            </Button>
          </div>

          {/* Main content: countdown + Heimlich */}
          <div className="mt-6 grid flex-1 grid-cols-1 gap-4 lg:grid-cols-[1fr,auto]">
            <div className="flex flex-col">
              <div className="rounded-md border-2 border-foreground/30 bg-black/40 p-5 backdrop-blur-md">
                <div className="mono micro text-foreground/70">heimlich · do these steps now</div>
                <div className="mt-4">
                  <Heimlich />
                </div>
              </div>
            </div>

            {/* Right rail: countdown + actions */}
            <div className="flex w-full flex-col gap-3 lg:w-[20rem]">
              <div className="rounded-md border-2 border-foreground/30 bg-black/40 p-5 text-center backdrop-blur-md">
                <div className="mono micro text-foreground/70">911 in</div>
                <div className="display mt-2 text-7xl leading-none tabular text-foreground">
                  {String(secondsLeft).padStart(2, "0")}
                </div>
                <div className="mt-2 text-[11px] text-foreground/75">
                  auto-dialing emergency services
                </div>
                {calling && (
                  <div className="mono mt-3 text-[11px] uppercase tracking-wider text-foreground">
                    calling now
                  </div>
                )}
              </div>

              <Button
                onClick={dial911}
                className="mono h-12 rounded-sm bg-foreground text-[13px] tracking-wider text-background hover:bg-foreground/90"
              >
                <PhoneCall className="h-4 w-4" /> CALL 911 NOW
              </Button>

              <Button
                onClick={sendSms}
                variant="outline"
                className="mono h-10 rounded-sm border-foreground/40 bg-black/30 text-[12px] tracking-wider text-foreground hover:bg-black/50"
              >
                <MessageSquare className="h-4 w-4" /> text emergency contact
              </Button>

              <div className="mono mt-2 text-[10px] leading-relaxed text-foreground/70">
                Stay calm. Give 5 firm thrusts. If the airway clears, monitor
                breathing and call 911 even after relief.
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
