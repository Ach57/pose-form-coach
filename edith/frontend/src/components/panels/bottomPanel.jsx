import { VoiceWave } from "../ui/effects";
import { COMMANDS } from "../../constants";

/**
 * BottomPanel
 * Voice visualizer, live transcript, mic button, and quick command buttons.
 */
export function BottomPanel({ listening, transcript, onCommand, onMicClick }) {
  return (
    <div className="bottom-panel">
      <div className="voice-section">
        <div className="voice-label">VOICE INPUT · EDITH LISTENING</div>
        <VoiceWave listening={listening} />
        <div className="transcript-display">
          {transcript || (listening ? "..." : "")}
        </div>
        {onMicClick && (
          <button
            className={`cmd-btn mic-btn${listening ? " mic-btn--active" : ""}`}
            onClick={listening ? undefined : onMicClick}
            disabled={listening}
            title="Speak a command"
          >
            {listening ? "● LISTENING" : "🎤 SPEAK"}
          </button>
        )}
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
