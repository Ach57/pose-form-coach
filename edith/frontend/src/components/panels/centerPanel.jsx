import { PulseRing } from "../ui/effects";
import { StatBar } from "../ui/atoms";
import { MODELS } from "../../constants/models";

/**
 * CenterPanel
 * The hero panel: pulsing orb, system state label, boot bar, live inference stats,
 * and the annotated camera feed streamed from the backend as base64 JPEG.
 */
export function CenterPanel({ systemState, activeModel, bootProgress, stats, frameSrc }) {
  const activeModelData = MODELS.find((m) => m.id === activeModel);
  const accentColor = activeModelData?.color || "#00f0ff";

  const STATE_LABEL = {
    idle: "OFFLINE",
    booting: "LOADING",
    active: activeModelData?.name,
    shutdown: "OFFLINE",
  };

  const STATE_SUBLABEL = {
    idle: "Awaiting command",
    booting: "Initializing inference engine",
    active: activeModelData?.label,
    shutdown: "Terminating processes",
  };

  return (
    <div className="panel center-panel">
      {/* Camera feed — shown while active and a frame has arrived */}
      {systemState === "active" && frameSrc ? (
        <img
          src={frameSrc}
          alt="Live inference feed"
          className="camera-feed"
          style={{
            width: "100%",
            borderRadius: "8px",
            border: `1px solid ${accentColor}44`,
            marginBottom: "12px",
          }}
        />
      ) : (
        <PulseRing active={systemState === "active"} color={accentColor} />
      )}

      <div className="model-name-display" style={{ color: accentColor }}>
        {STATE_LABEL[systemState]}
      </div>
      <div className="model-sublabel">{STATE_SUBLABEL[systemState]}</div>

      {systemState === "booting" && (
        <div className="boot-bar-container">
          <div className="boot-bar-track">
            <div
              className="boot-bar-fill"
              style={{ width: `${bootProgress}%` }}
            />
          </div>
          <div className="boot-pct">{bootProgress}%</div>
        </div>
      )}

      {systemState === "active" && (
        <div className="live-stats">
          <StatBar
            label="CONFIDENCE"
            value={stats.confidence}
            color={accentColor}
          />
          <StatBar
            label={`FPS  ${stats.fps}`}
            value={Math.min(stats.fps * 3, 100)}
            color="#44ff88"
          />
          <StatBar
            label={`LAT  ${stats.latency}ms`}
            value={Math.max(100 - stats.latency * 3, 20)}
            color="#ffb347"
          />
        </div>
      )}
    </div>
  );
}
