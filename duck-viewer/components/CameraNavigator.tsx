"use client";

import { useEffect, useState, type MutableRefObject } from "react";

import {
  cameraMotionStart,
  cameraMotionStop,
  requestCameraReset,
  type CameraMotion,
} from "@/lib/camera";
import { loadJSON, saveJSON } from "@/lib/persist";
import type { LabClient } from "@/lib/lab";
import { pushToast } from "./Toasts";
import styles from "./CameraNavigator.module.css";

interface CameraNavigatorProps {
  clientRef: MutableRefObject<LabClient | null>;
  connected: boolean;
  defaultOpen?: boolean;
}

interface MotionButtonProps {
  motion: CameraMotion;
  symbol: string;
  label: string;
}

function MotionButton({ motion, symbol, label }: MotionButtonProps) {
  const [active, setActive] = useState(false);

  const stop = () => {
    cameraMotionStop(motion);
    setActive(false);
  };

  useEffect(() => () => cameraMotionStop(motion), [motion]);

  return (
    <button
      type="button"
      className={styles.control}
      aria-label={label}
      aria-pressed={active}
      title={label}
      data-active={active}
      onPointerDown={(event) => {
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        cameraMotionStart(motion);
        setActive(true);
      }}
      onPointerUp={stop}
      onPointerCancel={stop}
      onLostPointerCapture={stop}
      onKeyDown={(event) => {
        if ((event.key === "Enter" || event.key === " ") && !active) {
          event.preventDefault();
          cameraMotionStart(motion);
          setActive(true);
        }
      }}
      onKeyUp={(event) => {
        if (event.key === "Enter" || event.key === " ") stop();
      }}
    >
      {symbol}
    </button>
  );
}

export function CameraNavigator({
  clientRef,
  connected,
  defaultOpen = true,
}: CameraNavigatorProps) {
  const [open, setOpen] = useState(() =>
    loadJSON("cameraNavigatorOpen", defaultOpen)
  );

  useEffect(() => saveJSON("cameraNavigatorOpen", open), [open]);

  if (!open) {
    return (
      <button
        type="button"
        className={styles.collapsed}
        onClick={() => setOpen(true)}
        aria-label="Open camera navigator"
        title="Open camera navigator"
      >
        ⌖
      </button>
    );
  }

  return (
    <section className={styles.navigator} aria-label="Camera navigator">
      <div className={styles.header}>
        <strong>Camera</strong>
        <button
          type="button"
          onClick={() => setOpen(false)}
          aria-label="Collapse camera navigator"
          title="Collapse camera navigator"
        >
          −
        </button>
      </div>
      <div className={styles.grid}>
        <MotionButton motion="orbitLeft" symbol="↶" label="Orbit camera left" />
        <MotionButton motion="up" symbol="↑" label="Raise camera" />
        <MotionButton motion="orbitRight" symbol="↷" label="Orbit camera right" />
        <MotionButton motion="truckLeft" symbol="←" label="Move camera left" />
        <button
          type="button"
          className={`${styles.control} ${styles.home}`}
          onClick={requestCameraReset}
          aria-label="Reset camera view"
          title="Reset camera view (Shift+R)"
        >
          ⌂
        </button>
        <MotionButton motion="truckRight" symbol="→" label="Move camera right" />
        <MotionButton motion="dollyOut" symbol="−" label="Zoom out" />
        <MotionButton motion="down" symbol="↓" label="Lower camera" />
        <MotionButton motion="dollyIn" symbol="+" label="Zoom in" />
      </div>
      <button
        type="button"
        className={`${styles.control} ${styles.simReset}`}
        disabled={!connected}
        onClick={() => {
          clientRef.current?.sendReset();
          pushToast("Simulation restarted from zero");
        }}
        aria-label="Restart every duck simulation"
        title="Restart every duck simulation (R)"
      >
        ↻
      </button>
    </section>
  );
}
