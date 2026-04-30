"use client";

/**
 * Heimlich maneuver instructions — illustrated, large-format, designed to be
 * readable from across a room while a bystander panics. No nuance: 4 steps,
 * each illustrated, all visible without scrolling on common screens.
 */

interface StepProps {
  index: string;
  label: string;
  desc: string;
}

const STEPS: StepProps[] = [
  {
    index: "01",
    label: "STAND BEHIND",
    desc: "Stand behind the person. Wrap your arms around their waist. Lean them forward.",
  },
  {
    index: "02",
    label: "MAKE A FIST",
    desc: "Make a fist with one hand. Place the thumb side just above the navel.",
  },
  {
    index: "03",
    label: "GRASP & THRUST",
    desc: "Grasp your fist with the other hand. Thrust inward and upward — hard.",
  },
  {
    index: "04",
    label: "REPEAT × 5",
    desc: "Five quick thrusts. Then check the airway. Continue until the object is dislodged.",
  },
];

export function Heimlich() {
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      {STEPS.map((s) => (
        <div
          key={s.index}
          className="relative rounded-md border-2 border-foreground/30 bg-black/40 p-4 backdrop-blur-md"
        >
          <div className="mono micro mb-3 text-foreground/60">{s.index}</div>
          <div className="display mb-2 text-2xl leading-none text-foreground">
            {s.label}
          </div>
          <div className="text-[12px] leading-snug text-foreground/85">
            {s.desc}
          </div>
        </div>
      ))}
    </div>
  );
}
