import { CornerBracket } from "../ui/atoms";

/**
 * ModelCard
 * Clickable card in the left panel for each inference module.
 * Shows active/standby state with color accent and glow.
 */
export function ModelCard({ model, isActive, onActivate }) {
  return (
    <div
      className={`model-card ${isActive ? "active" : ""}`}
      style={{ "--model-color": model.color }}
      onClick={onActivate}
    >
      <CornerBracket position="tl" />
      <CornerBracket position="tr" />
      <CornerBracket position="bl" />
      <CornerBracket position="br" />

      <div className="model-id">MODEL {model.name}</div>
      <div className="model-label">{model.label}</div>
      <div className="model-status">{isActive ? "● ACTIVE" : "○ STANDBY"}</div>

      {isActive && (
        <div
          className="model-glow"
          style={{
            background: `radial-gradient(ellipse, ${model.color}22 0%, transparent 70%)`,
          }}
        />
      )}
    </div>
  );
}
