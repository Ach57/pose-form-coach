import { useState, useCallback } from "react";

/**
 * useVoice
 *
 * The backend owns the entire voice loop (wake word via SpeechRecognition +
 * TTS via edge-tts). It pushes `wake_state` WebSocket events that drive
 * wakeState in the parent (App.jsx).
 *
 * This hook only handles the quick-command button simulation:
 *   simulateVoice(cmd) — typewriter effect then fires onCommand(cmd).
 *
 * Props:
 *   onCommand(text) — sends the command string to the WebSocket
 *   wakeState       — "idle" | "awake" | "listening" from the parent
 */
export function useVoice(onCommand, wakeState = "idle") {
  const [transcript, setTranscript] = useState("");

  // ── Button simulation ─────────────────────────────────────────────────────
  const simulateVoice = useCallback(
    (cmd) => {
      if (wakeState !== "idle") return; // don't clobber live voice
      setTranscript("");

      cmd.split("").forEach((_, i) => {
        setTimeout(() => setTranscript(cmd.slice(0, i + 1)), i * 40);
      });

      setTimeout(() => {
        onCommand(cmd);
        setTimeout(() => setTranscript(""), 1500);
      }, cmd.length * 40 + 400);
    },
    [wakeState, onCommand],
  );

  return { transcript, simulateVoice };
}
