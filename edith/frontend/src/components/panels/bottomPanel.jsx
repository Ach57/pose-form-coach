import { VoiceWave } from "../ui/effects";
import { COMMANDS } from "../../constants";

const WAKE_LABELS = {
  idle:      "EDITH STANDBY — Say \"Edith\"",
  awake:     "EDITH AWAKE — Yes?",
  listening: "EDITH LISTENING — Speak your command...",
};

/**
 * BottomPanel
 * Voice visualizer with two-phase wake word state:
 *   idle      → dim, shows "Say 'Edith'"
 *   awake     → pulse, shows "Yes?"
 *   listening → active wave, shows "Speak your command..."
 *
 * Quick command buttons simulate the same flow without mic.
 */
export function BottomPanel({ wakeState = "idle", listening, transcript, onCommand }) {
  const label = WAKE_LABELS[wakeState] ?? WAKE_LABELS.idle;

  return (
    <div className="bottom-panel">
      <div className="voice-section">
        <div className={`voice-label wake-label--${wakeState}`}>{label}</div>
        <VoiceWave listening={listening} />
        <div className="transcript-display">
          {transcript || (wakeState === "listening" ? "..." : "")}
        </div>
      </div>

      <div className="cmd-section">
        <div className="cmd-section-label">QUICK COMMANDS</div>
        <div className="cmd-buttons">
          {COMMANDS.map((cmd) => (
            <button
              key={cmd}
              className="cmd-btn"
              onClick={() => onCommand(cmd)}
              disabled={listening}
            >
              {cmd}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
