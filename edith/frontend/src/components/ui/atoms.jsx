import { LOG_COLORS } from "../../constants";

// ─── CornerBracket ────────────────────────────────────────────────────────────
// Decorative HUD corner markers. position: "tl" | "tr" | "bl" | "br"
export function CornerBracket({ position }) {
  const styles = {
    tl: { top: 0, left: 0, borderTop: "2px solid", borderLeft: "2px solid" },
    tr: { top: 0, right: 0, borderTop: "2px solid", borderRight: "2px solid" },
    bl: {
      bottom: 0,
      left: 0,
      borderBottom: "2px solid",
      borderLeft: "2px solid",
    },
    br: {
      bottom: 0,
      right: 0,
      borderBottom: "2px solid",
      borderRight: "2px solid",
    },
  };
  return <div className="corner-bracket" style={styles[position]} />;
}

// ─── StatBar ──────────────────────────────────────────────────────────────────
// Horizontal progress bar used for CPU/GPU/MEM and inference metrics.
export function StatBar({ label, value, color }) {
  return (
    <div className="stat-bar">
      <div className="stat-label">{label}</div>
      <div className="stat-track">
        <div
          className="stat-fill"
          style={{
            width: `${value}%`,
            background: `linear-gradient(90deg, ${color}88, ${color})`,
          }}
        />
      </div>
      <div className="stat-value">{value}%</div>
    </div>
  );
}

// ─── LogEntry ─────────────────────────────────────────────────────────────────
// Single row in the event log. entry: { id, time, text, type }
export function LogEntry({ entry }) {
  return (
    <div
      className="log-entry"
      style={{ color: LOG_COLORS[entry.type] || LOG_COLORS.info }}
    >
      <span className="log-time">[{entry.time}]</span>
      <span className="log-text">{entry.text}</span>
    </div>
  );
}
