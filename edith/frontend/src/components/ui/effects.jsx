// ─── HexGrid ──────────────────────────────────────────────────────────────────
// Full-bleed SVG hex pattern rendered as background atmosphere.
function hexPoints(cx, cy, r) {
  return Array.from({ length: 6 }, (_, i) => {
    const a = (Math.PI / 3) * i - Math.PI / 6;
    return `${cx + r * Math.cos(a)},${cy + r * Math.sin(a)}`;
  }).join(" ");
}

export function HexGrid() {
  const hexes = [];
  const cols = 18,
    rows = 10,
    r = 28;
  const dx = r * Math.sqrt(3),
    dy = r * 1.5;

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const cx = col * dx + (row % 2) * (dx / 2) + 20;
      const cy = row * dy + 20;
      const opacity = Math.random() * 0.06 + 0.02;
      hexes.push(
        <polygon
          key={`${row}-${col}`}
          points={hexPoints(cx, cy, r - 2)}
          fill="none"
          stroke="#00f0ff"
          strokeWidth="0.5"
          opacity={opacity}
        />,
      );
    }
  }

  return (
    <svg
      className="hex-grid"
      viewBox="0 0 900 400"
      preserveAspectRatio="xMidYMid slice"
    >
      {hexes}
    </svg>
  );
}

// ─── PulseRing ────────────────────────────────────────────────────────────────
// Expanding concentric rings around the central orb when active.
export function PulseRing({ active, color }) {
  return (
    <div className="pulse-container">
      {active && (
        <>
          <div
            className="pulse-ring"
            style={{ borderColor: color, animationDelay: "0s" }}
          />
          <div
            className="pulse-ring"
            style={{ borderColor: color, animationDelay: "0.5s" }}
          />
          <div
            className="pulse-ring"
            style={{ borderColor: color, animationDelay: "1s" }}
          />
        </>
      )}
      <div
        className="core-orb"
        style={{
          background: active ? color : "#1a2a3a",
          boxShadow: active ? `0 0 30px ${color}, 0 0 60px ${color}44` : "none",
        }}
      />
    </div>
  );
}

// ─── ScanLine ─────────────────────────────────────────────────────────────────
// Animated horizontal light sweep across the entire HUD.
export function ScanLine() {
  return <div className="scan-line" />;
}

// ─── VoiceWave ────────────────────────────────────────────────────────────────
// Audio visualizer bar cluster. Animates when listening=true.
export function VoiceWave({ listening }) {
  return (
    <div className="voice-wave">
      {Array.from({ length: 24 }, (_, i) => (
        <div
          key={i}
          className={`wave-bar ${listening ? "active" : ""}`}
          style={{
            animationDelay: `${(i * 0.07) % 1.2}s`,
            animationDuration: `${0.6 + (i % 5) * 0.1}s`,
          }}
        />
      ))}
    </div>
  );
}
