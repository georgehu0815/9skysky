"use client";

// In-app operating guide for Duck Viewer. The policy notes are intentionally
// specific to the nine vendored Pollen policies: this is a verification guide,
// not a promise that every robotd command sequence is reproduced by the lab.

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { Frame, LabClient } from "@/lib/lab";
import { pushToast } from "./Toasts";

const mono = "ui-monospace, SFMono-Regular, Menlo, monospace";

export const GUIDE_TABS = ["Quick start", "Policies", "Verify", "Controls"] as const;
type GuideTab = (typeof GUIDE_TABS)[number];
type Support = "full" | "partial" | "smoke-test only";

export interface ShippedPolicyGuide {
  name: string;
  purpose: string;
  semantics: string;
  support: Support;
  steps: readonly string[];
  expected: string;
  verify: readonly string[];
  limitation: string;
}

/** One card per ONNX file in microduck/policies. Keep this explicit: adding a
 * policy to the shipped bundle should force a deliberate guide update. */
export const SHIPPED_POLICY_GUIDES: readonly ShippedPolicyGuide[] = [
  {
    name: "alpha_walking",
    purpose: "Velocity-command walking / velstand gait.",
    semantics:
      "Uses the shared 61-value observation and 14 joint actions. Duck Lab's automatic runway asks vx = 0.9 m/s for 27 s, then zero for 3 s; the 30 s episode repeats.",
    support: "full",
    steps: [
      "Spawn it, or drag its policy chip onto an existing duck.",
      "Select the duck so its roster row and amber floor ring agree.",
      "Press R to restart every duck at the beginning of the same runway cycle.",
      "Watch the HUD m/s pair (achieved / asked) and falls through at least 30 s.",
    ],
    expected:
      "It should start from rest, walk forward during the 0.9 m/s request, then settle during the 3 s zero-command tail.",
    verify: [
      "Achieved speed remains meaningful beside the 0.9 m/s request.",
      "No fall is hidden by an unsynchronised start.",
      "Direction and recovery remain visually stable for a full cycle.",
    ],
    limitation:
      "This is deterministic local MuJoCo inspection, not the official GPU sim2real stack or hardware validation.",
  },
  {
    name: "alpha_stand",
    purpose: "Standing balance with trained head and body-pose control.",
    semantics:
      "robotd can drive head/body command fields. The Viewer sends zero head and body commands while still sharing the runway twist, so only the zero-pose slice is represented faithfully.",
    support: "partial",
    steps: [
      "Spawn it and select its roster row.",
      "Use Zero command for 6 seconds below to remove the shared runway twist temporarily.",
      "Press R, then inspect trunk, feet, head, and recovery from the reset pose.",
    ],
    expected: "At zero command it should seek a stable standing/body-neutral pose.",
    verify: [
      "It loads and produces a controlled, non-limp body response.",
      "Balance can be inspected without treating walking reward as its score.",
    ],
    limitation:
      "The Viewer cannot sweep or verify the policy's commanded head/body-pose range.",
  },
  {
    name: "alpha_sitstand",
    purpose: "Sit-to-stand posture transitions.",
    semantics:
      "The posture flag is encoded in twist vx: 1 means sit; 0 means rise/stand. The current 0.9-then-0 runway roughly exercises sit then rise, but it is not robotd’s exact sit/rise scheduler.",
    support: "partial",
    steps: [
      "Spawn it and press R to align the 30 s command cycle.",
      "Observe the long nonzero segment as the sit request.",
      "Observe the final 3 s zero segment for the rise/stand response.",
    ],
    expected: "A visible posture change toward sitting, followed by a rise/stand response when vx returns to zero.",
    verify: [
      "The two command phases produce distinct postures.",
      "The duck remains controlled across the transition.",
    ],
    limitation:
      "0.9 is only an approximate nonzero posture flag here, and the Viewer's 27 s / 3 s timing does not reproduce robotd’s transition timing.",
  },
  {
    name: "alpha_ground_pick",
    purpose: "Phase-driven ground-pick motion.",
    semantics:
      "robotd supplies twist = [cos(2πp), sin(2πp), 0], advances p over a nominal 4 s cycle, and hands back at phase 0.7.",
    support: "smoke-test only",
    steps: [
      "Spawn it to confirm the ONNX loads into a Viewer duck.",
      "Press R and check that the stream, roster, and body outputs remain alive.",
      "Do not grade the resulting movement as a ground pick.",
    ],
    expected: "Only a loaded policy and finite body response are expected in this Viewer path.",
    verify: [
      "The policy appears in the roster and produces renderable output.",
      "Treat any apparent trick motion as incidental without the phase command.",
    ],
    limitation:
      "The Viewer does not generate the cosine/sine phase command, so it cannot validate ground-pick behavior.",
  },
  {
    name: "ball_kick_left",
    purpose: "One-shot kick with the left leg.",
    semantics:
      "Expected twist is zero. robotd runs the kick network for a 0.5 s window, then hands control back to the normal policy scheduler.",
    support: "partial",
    steps: [
      "Spawn it, select it, and apply Zero command for 6 seconds.",
      "Press R and watch the left leg closely from a clear side/front angle.",
      "Capture a short video if frame-by-frame review is needed.",
    ],
    expected: "A brief left-leg kick-like response may be inspected near the start.",
    verify: ["Confirm the left leg is the active leg.", "Judge only the initial response, not a repeated raw-policy loop."],
    limitation:
      "A raw Viewer duck has no automatic 0.5 s robotd handoff, so repetition and recovery are not end-to-end kick validation.",
  },
  {
    name: "ball_kick_right",
    purpose: "One-shot kick with the right leg.",
    semantics:
      "Expected twist is zero. robotd runs the kick network for a 0.5 s window, then hands control back to the normal policy scheduler.",
    support: "partial",
    steps: [
      "Spawn it, select it, and apply Zero command for 6 seconds.",
      "Press R and watch the right leg closely from a clear side/front angle.",
      "Capture a short video if frame-by-frame review is needed.",
    ],
    expected: "A brief right-leg kick-like response may be inspected near the start.",
    verify: ["Confirm the right leg is the active leg.", "Judge only the initial response, not a repeated raw-policy loop."],
    limitation:
      "A raw Viewer duck has no automatic 0.5 s robotd handoff, so repetition and recovery are not end-to-end kick validation.",
  },
  {
    name: "roller",
    purpose: "Roller-mode locomotion.",
    semantics:
      "In robotd roller mode this replaces the walk network and uses the roller tuning preset, including action scale 0.8.",
    support: "partial",
    steps: [
      "Spawn it and select it to confirm policy loading.",
      "Press R and inspect the commanded body/joint response through one cycle.",
      "Record what the body does, but do not score wheel travel or traction.",
    ],
    expected: "A coherent roller-trained body response can be visually inspected.",
    verify: ["The interface loads and outputs remain controlled.", "Label captures as Viewer body-response inspection."],
    limitation:
      "The current lab scene and physics are not roller-mode hardware validation; wheel locomotion claims are unsupported.",
  },
  {
    name: "roller_crouch",
    purpose: "Roller-mode crouch occupying robotd’s ground-pick slot.",
    semantics:
      "Robot configuration identifies a nominal 5 s phase cycle and action scale 0.8 for the roller crouch.",
    support: "smoke-test only",
    steps: [
      "Spawn it to confirm loading and 61→14 interface compatibility.",
      "Press R and inspect only for finite, renderable joint/body output.",
      "Do not infer a valid crouch cycle from the runway command.",
    ],
    expected: "Only policy loading and a body response are expected.",
    verify: ["The duck remains in the roster and stream.", "No wheel or phase behavior is claimed."],
    limitation:
      "The Viewer neither validates roller physics nor supplies the phase drive; this is not crouch behavior validation.",
  },
  {
    name: "roulade",
    purpose: "One-shot forward roll.",
    semantics:
      "Expected command is zero. robotd runs a 1.0 s skill window, then hands off; a held request can chain another window.",
    support: "partial",
    steps: [
      "Spawn it, select it, and apply Zero command for 6 seconds.",
      "Press R and inspect the first second from a useful side angle.",
      "Record the attempt to inspect rotation, contact, and the state after the nominal window.",
    ],
    expected: "A forward-roll response may be visible during the initial one-second interval.",
    verify: ["Check forward rather than sideways/backward rotation.", "Separate the first window from later raw-policy motion."],
    limitation:
      "The raw Viewer lacks robotd’s timed handoff and request-based chaining, so full sequencing and recovery are not validated.",
  },
] as const;

export function countPolicyDucks(frame: Frame | null, name: string): number {
  const id = `pollen:${name}`;
  return frame?.ducks.filter((duck) => duck.policy === id).length ?? 0;
}

const sectionTitle: React.CSSProperties = {
  color: "#e8e6e1",
  fontSize: 14,
  fontWeight: 700,
  margin: "0 0 8px",
};

const cardStyle: React.CSSProperties = {
  background: "rgba(255,255,255,0.035)",
  border: "1px solid rgba(255,255,255,0.09)",
  borderRadius: 9,
  padding: "11px 12px",
};

const actionStyle: React.CSSProperties = {
  background: "#1c2230",
  border: "1px solid rgba(255,255,255,0.14)",
  borderRadius: 7,
  color: "#cfe4f5",
  cursor: "pointer",
  fontFamily: mono,
  fontSize: 11,
  padding: "5px 9px",
};

function Numbered({ items }: { items: readonly string[] }) {
  return (
    <ol style={{ margin: "5px 0 0", paddingLeft: 20 }}>
      {items.map((item) => <li key={item} style={{ marginBottom: 3 }}>{item}</li>)}
    </ol>
  );
}

function Checks({ items }: { items: readonly string[] }) {
  return (
    <ul style={{ listStyle: "none", margin: "5px 0 0", padding: 0 }}>
      {items.map((item) => <li key={item} style={{ marginBottom: 3 }}>□ {item}</li>)}
    </ul>
  );
}

function SupportBadge({ support }: { support: Support }) {
  const color = support === "full" ? "#7dd87d" : support === "partial" ? "#d8c97d" : "#e0a08f";
  return (
    <span style={{ color, border: `1px solid ${color}66`, borderRadius: 10, padding: "1px 7px", fontSize: 10, whiteSpace: "nowrap" }}>
      {support}
    </span>
  );
}

function QuickStart() {
  return (
    <div>
      <h2 style={sectionTitle}>Quick start</h2>
      <p style={{ marginTop: 0 }}>From the workspace root, run this command in a terminal. It is reference text, not a GUI action:</p>
      <div style={{ ...cardStyle, color: "#9fb4d8", userSelect: "text", marginBottom: 10 }}>./restart-lab.sh</div>
      <p>
        It starts Duck Lab at <span style={{ color: "#7db8d8" }}>http://127.0.0.1:8788</span> (frames on <span style={{ color: "#7db8d8" }}>ws://127.0.0.1:8788/ws</span>) and Duck Viewer at <span style={{ color: "#7db8d8" }}>http://127.0.0.1:63317</span>.
      </p>
      <Numbered items={[
        "Wait for the Duck Lab header to show a green live connection and a non-empty roster.",
        "Open 🧠 policies. Drag a policy onto a duck to assign it, or double-click a chip / use Spawn here to add a duck.",
        "Click the duck or its HUD row; confirm the amber floor ring and highlighted row.",
        "Press R to synchronise/reset every duck to episode step zero.",
        "Compare side by side for at least one full 30 s cycle.",
        "Use 📷 shot or select a duck and use 🎥 record to capture evidence.",
      ]} />
      <div style={{ ...cardStyle, marginTop: 12 }}>
        <strong style={{ color: "#d8c97d" }}>Read claims narrowly.</strong> A policy loading is a smoke test. A visible response without robotd’s command encoding or handoff is partial inspection. Only the walking runway is fully represented here.
      </div>
    </div>
  );
}

function Policies({ frame, online, clientRef }: { frame: Frame | null; online: boolean; clientRef: React.MutableRefObject<LabClient | null> }) {
  const spawn = (name: string) => {
    clientRef.current?.sendSpawnDuck(`pollen:${name}`);
    pushToast(`＋ spawning ${name}`);
  };
  return (
    <div>
      <h2 style={sectionTitle}>Shipped policies</h2>
      <p style={{ marginTop: 0 }}>All nine vendored policies are ONNX <strong style={{ color: "#dfe5ee" }}>61 inputs → 14 actions</strong>. Their names are stable robot roles copied from the upstream runtime, not training-run names.</p>
      <div style={{ display: "grid", gap: 10 }}>
        {SHIPPED_POLICY_GUIDES.map((policy) => {
          const count = countPolicyDucks(frame, policy.name);
          return (
            <article key={policy.name} style={cardStyle}>
              <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 7 }}>
                <strong style={{ color: "#dfe5ee", fontSize: 12 }}>{policy.name}</strong>
                <SupportBadge support={policy.support} />
                <span style={{ color: "#566072", fontSize: 10 }}>{count} in scene</span>
                <span style={{ flex: 1 }} />
                <button
                  type="button"
                  disabled={!online}
                  title={online ? `spawn pollen:${policy.name}` : "Duck Lab is offline"}
                  onClick={() => spawn(policy.name)}
                  style={{ ...actionStyle, opacity: online ? 1 : 0.42, cursor: online ? "pointer" : "default" }}
                >
                  ＋ Spawn
                </button>
              </div>
              <p style={{ color: "#c4cad4", margin: "7px 0 4px" }}><strong>Purpose:</strong> {policy.purpose}</p>
              <p style={{ margin: "4px 0" }}><strong style={{ color: "#aab3c0" }}>Input/runtime:</strong> {policy.semantics}</p>
              <div style={{ color: "#aab3c0", marginTop: 7 }}><strong>In the Viewer</strong><Numbered items={policy.steps} /></div>
              <p style={{ margin: "7px 0 4px" }}><strong style={{ color: "#aab3c0" }}>Expected:</strong> {policy.expected}</p>
              <div><strong style={{ color: "#aab3c0" }}>Verification checklist</strong><Checks items={policy.verify} /></div>
              <p style={{ color: "#e0a08f", margin: "7px 0 0" }}>⚠ {policy.limitation}</p>
            </article>
          );
        })}
      </div>
    </div>
  );
}

function Verify() {
  return (
    <div>
      <h2 style={sectionTitle}>Repeatable verification protocol</h2>
      <Numbered items={[
        "Spawn the candidate and a relevant reference side by side; use matching viewpoints and do not compare unsynchronised episodes.",
        "Select the candidate. Confirm its amber ring, highlighted HUD row, exact policy id/name, episode time, and fall count.",
        "Press R once to synchronise every duck at step zero.",
        "Observe at least one full 30 s cycle. Note falls, resets, contacts, posture, direction, and recovery—not only a pleasing instant.",
        "For locomotion only, compare HUD achieved / asked speed. Do not apply that speed test to a zero-command or phase-driven skill.",
        "Ignore walking reward r̄ for trick policies; the HUD deliberately dims it because it scores the walking recipe.",
        "Capture a PNG for pose evidence or a video for timing/contact evidence. Keep the tab visible while recording.",
        "State the verdict as smoke test, partial inspection, or behavior validation, using each policy card’s support badge.",
      ]} />
      <div style={{ ...cardStyle, marginTop: 12 }}>
        <strong style={{ color: "#dfe5ee" }}>Compatibility:</strong> loading confirms the shared ONNX shape contract (61 observation values to 14 actions) and that the local interface can run the model. It does not prove the commands match the skill’s training distribution.
      </div>
      <div style={{ ...cardStyle, marginTop: 10 }}>
        <strong style={{ color: "#dfe5ee" }}>Deterministic visual check:</strong> the exported ONNX path is deterministic for the same observation sequence, but resets, simulation state, camera angle, command encoding, physics, and robotd handoffs still affect what is seen. Preserve the reset point and capture the whole relevant interval.
      </div>
    </div>
  );
}

function Controls() {
  return (
    <div>
      <h2 style={sectionTitle}>Controls and panels</h2>
      <div style={{ display: "grid", gap: 9 }}>
        <section style={cardStyle}>
          <strong style={{ color: "#dfe5ee" }}>Camera and stage</strong>
          <p>Drag to orbit; wheel or two-finger vertical scroll zooms; two-finger horizontal swipe slides. A/D slide, W/S or ↑/↓ dolly, ←/→ orbit, Q/E rise/fall, Shift+R resets the view.</p>
          <p style={{ color: "#d8c97d", marginBottom: 0 }}>WASD moves the camera—not the duck. Policies drive ducks; there is no keyboard teleop.</p>
        </section>
        <section style={cardStyle}>
          <strong style={{ color: "#dfe5ee" }}>Selection and episode</strong>
          <p>Click a duck or HUD row to select it (amber ring). Click empty floor or press Esc to deselect. Delete/Backspace removes the selected duck. R restarts every duck’s simulation together; Shift+R is camera reset.</p>
        </section>
        <section style={cardStyle}>
          <strong style={{ color: "#dfe5ee" }}>🧠 Policies</strong>
          <p>Drag a chip onto a duck to assign it. Drag to empty floor, double-click the chip, or arm it then click empty floor to spawn. Click a chip to arm, then click a duck to assign. Drop a run onto 🎓 teach to load it. Press / to open/focus policy search; Esc cancels an armed or dragged chip.</p>
        </section>
        <section style={cardStyle}>
          <strong style={{ color: "#dfe5ee" }}>🎓 Teach</strong>
          <p>Describe a trick, inspect its reward recipe and live score, add helpers with ＋, then adjust unlocked term/stage weights after a run. Retrain starts fresh; fine-tune continues from the selected finished run. Reward charts are training evidence, not visual proof.</p>
        </section>
        <section style={cardStyle}>
          <strong style={{ color: "#dfe5ee" }}>🎬 Animate</strong>
          <p>Choose 🦴 joint or 🎮 rig mode, use sliders or drag duck parts/⇕ handle, add and retime key poses, scrub/play the timeline, save a clip, then train a policy to track it. Shift gives fine dragging.</p>
        </section>
        <section style={cardStyle}>
          <strong style={{ color: "#dfe5ee" }}>📷 / 🎥 Capture</strong>
          <p>📷 shot downloads the current view as PNG. Select a duck to reveal 🎥 record; the camera frames it and the lab produces MP4 + GIF downloads. Takes stop at 60 s; keep the tab visible so rendered frames are recorded.</p>
        </section>
      </div>
    </div>
  );
}

export function GuidePanel({ clientRef, connected, onClose }: { clientRef: React.MutableRefObject<LabClient | null>; connected: boolean; onClose: () => void }) {
  const [tab, setTab] = useState<GuideTab>("Quick start");
  const [frame, setFrame] = useState<Frame | null>(() => clientRef.current?.frame ?? null);
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const dialogRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    closeRef.current?.focus();
    return () => previous?.focus({ preventScroll: true });
  }, []);

  useEffect(() => {
    const id = window.setInterval(() => setFrame(clientRef.current?.frame ?? null), 500);
    return () => window.clearInterval(id);
  }, [clientRef]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = dialogRef.current?.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
      );
      if (!focusable?.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [onClose]);

  const online = connected && !!frame;
  const reset = () => {
    clientRef.current?.sendReset();
    pushToast("↺ sim restarted — every duck from zero");
  };
  const zero = () => {
    clientRef.current?.sendCmd([0, 0, 0]);
    pushToast("zero command shared by all ducks for 6 seconds");
  };

  return createPortal(
    <div
      data-policy-ui
      data-modal
      role="dialog"
      aria-modal="true"
      aria-labelledby="duck-guide-title"
      onClick={onClose}
      style={{ position: "fixed", inset: 0, zIndex: 1100, background: "rgba(0,0,0,0.62)", display: "flex", alignItems: "center", justifyContent: "center", padding: 12, boxSizing: "border-box", backdropFilter: "blur(2px)" }}
    >
      <div
        ref={dialogRef}
        onClick={(event) => event.stopPropagation()}
        style={{ width: 820, maxWidth: "calc(100vw - 24px)", maxHeight: "calc(100dvh - 24px)", background: "rgba(14,16,20,0.98)", border: "1px solid rgba(255,255,255,0.14)", borderRadius: 11, color: "#aab3c0", fontFamily: mono, fontSize: 11, lineHeight: 1.55, boxShadow: "0 16px 50px rgba(0,0,0,0.7)", display: "flex", flexDirection: "column", overflow: "hidden" }}
      >
        <header style={{ padding: "11px 13px 9px", borderBottom: "1px solid rgba(255,255,255,0.09)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <h1 id="duck-guide-title" style={{ margin: 0, color: "#e8e6e1", fontSize: 14 }}>? Duck Viewer guide</h1>
            <span style={{ flex: 1 }} />
            <span aria-live="polite" style={{ color: online ? "#7dd87d" : "#e07a5f", whiteSpace: "nowrap" }}>
              {online ? `● live · ${frame?.ducks.length ?? 0} ducks · ${frame?.mode ?? "auto"}` : `○ ${connected ? "waiting for frames" : "offline"}`}
            </span>
            <button ref={closeRef} type="button" onClick={onClose} aria-label="Close Duck Viewer guide" title="close guide (Esc)" style={{ ...actionStyle, padding: "2px 7px", color: "#aab3c0" }}>✕</button>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 9 }} role="tablist" aria-label="Guide sections">
            {GUIDE_TABS.map((item) => (
              <button
                key={item}
                type="button"
                role="tab"
                aria-selected={tab === item}
                onClick={() => setTab(item)}
                style={{ ...actionStyle, background: tab === item ? "#2a3548" : "#161b26", color: tab === item ? "#cfe4f5" : "#8b93a3", borderColor: tab === item ? "#7db8d8" : "rgba(255,255,255,0.10)" }}
              >
                {item}
              </button>
            ))}
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 7, marginTop: 9 }}>
            <button type="button" disabled={!online} onClick={reset} style={{ ...actionStyle, opacity: online ? 1 : 0.42, cursor: online ? "pointer" : "default" }}>↺ Restart all ducks</button>
            <button type="button" disabled={!online} onClick={zero} style={{ ...actionStyle, opacity: online ? 1 : 0.42, cursor: online ? "pointer" : "default" }}>0 Zero command for 6 seconds</button>
            <span style={{ color: "#566072", alignSelf: "center" }}>Command override is shared/global and temporary; auto resumes after 6 s.</span>
          </div>
        </header>
        <main role="tabpanel" tabIndex={0} style={{ padding: "13px", overflowY: "auto", overscrollBehavior: "contain" }}>
          {tab === "Quick start" && <QuickStart />}
          {tab === "Policies" && <Policies frame={frame} online={online} clientRef={clientRef} />}
          {tab === "Verify" && <Verify />}
          {tab === "Controls" && <Controls />}
        </main>
      </div>
    </div>,
    document.body
  );
}
