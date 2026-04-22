import { MODELS } from "../../constants";

const STATUS_TEXT = {
  idle: "STANDBY",
  booting: "INITIALIZING",
  shutdown: "SHUTTING DOWN",
};

/**
 * HudHeader
 * Top bar with EDITH title, subtitle, and live system status indicator.
 */
export function HudHeader({ systemState, activeModel }) {
  const activeModelData = MODELS.find((m) => m.id === activeModel);
  const accentColor = activeModelData?.color || "#00f0ff";

  const statusText =
    systemState === "active"
      ? `${activeModelData?.label?.toUpperCase()} ACTIVE`
      : STATUS_TEXT[systemState];

  return (
    <div className="hud-header">
      <div>
        <div
          className="edith-title"
          style={{
            color: accentColor,
            textShadow: `0 0 20px ${accentColor}, 0 0 40px ${accentColor}66`,
          }}
        >
          E.D.I.T.H
        </div>
        <div className="header-sub">
          Even Dead I'm The Hero · Pose Detection System
        </div>
      </div>

      <div className="sys-status">
        <div className="status-dot" style={{ background: accentColor }} />
        <span>{statusText}</span>
      </div>
    </div>
  );
}
