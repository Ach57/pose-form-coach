import "./styles/hud.css";

import { useState } from "react";
import { useEdith } from "./hooks/useEdith";
import { useLogs } from "./hooks/useLogs";
import { useVoice } from "./hooks/useVoice";
import { useWebSocket } from "./hooks/useWebSocket";

import { HexGrid, ScanLine } from "./components/ui/effects";
import { HudHeader } from "./components/panels/HudHeader";
import { LeftPanel } from "./components/panels/LeftPanel";
import { CenterPanel } from "./components/panels/CenterPanel";
import { RightPanel } from "./components/panels/RightPanel";
import { BottomPanel } from "./components/panels/BottomPanel";

export default function App() {
  // ── Logs ───────────────────────────────────────────────────────────────────
  const { logs, logRef, addLog } = useLogs([
    { id: 0, time: "00:00:00", text: "EDITH SYSTEM INITIALIZED", type: "info" },
    {
      id: 1,
      time: "00:00:01",
      text: "Awaiting voice authentication...",
      type: "warn",
    },
  ]);

  // ── Core state machine (driven entirely by server events) ──────────────────
  const { systemState, activeModel, bootProgress, stats, handleServerEvent } =
    useEdith();

  // ── Live camera frame (base64 JPEG from backend) ───────────────────────────
  const [frameSrc, setFrameSrc] = useState(null);

  // ── Wake state (driven by backend voice loop via WS events) ───────────────
  const [wakeState, setWakeState] = useState("idle");
  const [voiceTranscript, setVoiceTranscript] = useState("");

  // ── WebSocket ──────────────────────────────────────────────────────────────
  const { sendCommand } = useWebSocket((data) => {
    if (data.event === "log") addLog(data.text, data.type);
    else if (data.event === "frame")
      setFrameSrc(`data:image/jpeg;base64,${data.data}`);
    else if (data.event === "wake_state") {
      setWakeState(data.state);
      if (data.transcript != null) setVoiceTranscript(data.transcript);
      if (data.state === "idle") setTimeout(() => setVoiceTranscript(""), 1500);
    }
    else handleServerEvent(data);
  });

  // ── Voice (button simulation only — backend owns the mic) ────────────────
  const { transcript: btnTranscript, simulateVoice } = useVoice(sendCommand, wakeState);

  // Prefer live voice transcript (backend), fall back to button typewriter
  const transcript = voiceTranscript || btnTranscript;

  return (
    <div className="hud-root">
      {/* Background atmosphere */}
      <HexGrid />
      <ScanLine />
      <div className="noise" />
      <div className="vignette" />

      {/* HUD layout */}
      <div className="hud-layout">
        <HudHeader systemState={systemState} activeModel={activeModel} />
        <LeftPanel systemState={systemState} activeModel={activeModel} />
        <CenterPanel
          systemState={systemState}
          activeModel={activeModel}
          bootProgress={bootProgress}
          stats={stats}
          frameSrc={frameSrc}
        />
        <RightPanel
          systemState={systemState}
          activeModel={activeModel}
          stats={stats}
          logs={logs}
          logRef={logRef}
        />
        <BottomPanel
          wakeState={wakeState}
          listening={wakeState !== "idle"}
          transcript={transcript}
          onCommand={simulateVoice}
        />
      </div>
    </div>
  );
}
