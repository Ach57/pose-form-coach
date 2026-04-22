import { MODELS } from "../../constants";
import { ModelCard } from "./modelCard";
import { StatBar } from "../ui/atoms";

/**
 * LeftPanel
 * Lists inference modules (ModelCards) and system resource bars.
 */
export function LeftPanel({ systemState, activeModel }) {
  const isIdle = systemState === "idle";

  return (
    <div className="panel left-panel">
      <div className="panel-title">INFERENCE MODULES</div>

      {MODELS.map((m) => (
        <ModelCard
          key={m.id}
          model={m}
          isActive={activeModel === m.id}
          onActivate={() => {}}
        />
      ))}

      <div className="system-integrity">
        <div className="integrity-label">SYSTEM INTEGRITY</div>
        <StatBar
          label="CPU"
          value={systemState === "active" ? 62 : 14}
          color="#00f0ff"
        />
        <StatBar
          label="GPU"
          value={systemState === "active" ? 87 : 5}
          color="#ff6b35"
        />
        <StatBar
          label="MEM"
          value={systemState === "active" ? 45 : 22}
          color="#44ff88"
        />
      </div>
    </div>
  );
}
