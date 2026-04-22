import { useState, useCallback, useRef } from "react";

/**
 * useVoice
 * Real microphone input via Web Speech API (SpeechRecognition).
 * Falls back gracefully if the browser doesn't support it.
 *
 * simulateVoice(cmd) — types out a command char-by-char and dispatches it.
 * Used by the simulation buttons in BottomPanel.
 *
 * startListening() — activates the real microphone for one utterance.
 */
export function useVoice(onCommand) {
  const [listening,  setListening]  = useState(false);
  const [transcript, setTranscript] = useState("");
  const recognizerRef = useRef(null);

  // ── Real mic (Web Speech API) ─────────────────────────────────────────────
  const startListening = useCallback(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.warn("[EDITH] SpeechRecognition not supported in this browser");
      return;
    }

    if (listening) return;

    const recognizer = new SpeechRecognition();
    recognizer.lang = "en-US";
    recognizer.interimResults = true;
    recognizer.maxAlternatives = 1;
    recognizerRef.current = recognizer;

    recognizer.onstart = () => setListening(true);

    recognizer.onresult = (e) => {
      const result = e.results[e.results.length - 1];
      const text   = result[0].transcript;
      setTranscript(text);
      if (result.isFinal) {
        setListening(false);
        onCommand(text.trim());
        setTimeout(() => setTranscript(""), 1500);
      }
    };

    recognizer.onerror = (e) => {
      console.error("[EDITH] SpeechRecognition error:", e.error);
      setListening(false);
    };

    recognizer.onend = () => setListening(false);

    recognizer.start();
  }, [listening, onCommand]);

  const stopListening = useCallback(() => {
    recognizerRef.current?.stop();
    setListening(false);
  }, []);

  // ── Simulation (button press) ─────────────────────────────────────────────
  const simulateVoice = useCallback(
    (cmd) => {
      setListening(true);
      setTranscript("");

      cmd.split("").forEach((_, i) => {
        setTimeout(() => setTranscript(cmd.slice(0, i + 1)), i * 40);
      });

      setTimeout(
        () => {
          setListening(false);
          onCommand(cmd);
          setTimeout(() => setTranscript(""), 1500);
        },
        cmd.length * 40 + 400,
      );
    },
    [onCommand],
  );

  return { listening, transcript, startListening, stopListening, simulateVoice };
}
