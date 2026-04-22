import { LogEntry } from "../ui/atoms";

/**
 * RightPanel
 * Session telemetry grid + scrollable event log feed.
 */
export function RightPanel({ systemState, activeModel, stats, logs, logRef }) {
  const modelName = activeModel?.toUpperCase() || "—";

  const telemetry = [
    ["UPTIME", systemState === "active" ? "00:04:22" : "—"],
    [
      "FRAMES",
      systemState === "active" ? (stats.fps * 264).toLocaleString() : "—",
    ],
    ["MODEL", modelName],
    ["STATUS", systemState.toUpperCase()],
  ];

  return (
    <div className="panel right-panel">
      <div className="panel-title">SYSTEM LOG</div>

      <div className="stats-section">
        <div className="telemetry-label">SESSION TELEMETRY</div>
        <div className="telemetry-grid">
          {telemetry.map(([k, v]) => (
            <div key={k}>
              <div className="telemetry-key">{k}</div>
              <div className="telemetry-val">{v}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="log-section">
        <div className="panel-title">EVENTS</div>
        <div className="log-scroll" ref={logRef}>
          {logs.map((e) => (
            <LogEntry key={e.id} entry={e} />
          ))}
        </div>
      </div>
    </div>
  );
}
