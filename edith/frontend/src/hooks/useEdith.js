import { useState, useCallback, useRef } from "react";
import { MODELS } from "../constants/models";

/**
 * useEdith
 * Central state machine for the EDITH system.
 * All state changes are driven by WebSocket events from the backend.
 *
 * systemState: "idle" | "booting" | "active" | "shutdown"
 */
export function useEdith() {
  const [systemState, setSystemState] = useState("idle");
  const [activeModel, setActiveModel] = useState(null);
  const [bootProgress, setBootProgress] = useState(0);
  const [stats, setStats] = useState({ confidence: 0, fps: 0, latency: 0 });

  // Animate boot progress bar while state is "booting"
  const progressTimer = useRef(null);

  const _startProgressAnimation = useCallback(() => {
    setBootProgress(0);
    let p = 0;
    clearInterval(progressTimer.current);
    progressTimer.current = setInterval(() => {
      p = Math.min(p + 2, 95); // creep toward 95 — backend drives 100
      setBootProgress(p);
      if (p >= 95) clearInterval(progressTimer.current);
    }, 60);
  }, []);

  const _stopProgressAnimation = useCallback((finalValue = 100) => {
    clearInterval(progressTimer.current);
    setBootProgress(finalValue);
  }, []);

  /**
   * handleServerEvent — called by App.jsx for every WS message except "log".
   * Handles: state_change, stats, frame (frame is handled in App.jsx directly).
   */
  const handleServerEvent = useCallback(
    (data) => {
      if (data.event === "state_change") {
        const { state, model } = data;
        setSystemState(state);

        if (state === "booting") {
          setActiveModel(model);
          _startProgressAnimation();
        } else if (state === "active") {
          _stopProgressAnimation(100);
          setActiveModel(model);
        } else if (state === "idle" || state === "shutdown") {
          _stopProgressAnimation(0);
          if (state === "idle") setActiveModel(null);
        }
      } else if (data.event === "stats") {
        setStats({
          confidence: data.confidence,
          fps: data.fps,
          latency: data.latency,
        });
      }
    },
    [_startProgressAnimation, _stopProgressAnimation],
  );

  return {
    systemState,
    activeModel,
    bootProgress,
    stats,
    handleServerEvent,
  };
}
