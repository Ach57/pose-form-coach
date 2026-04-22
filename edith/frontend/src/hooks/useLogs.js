import { useState, useRef, useEffect } from "react";

/**
 * useLogs
 * Manages the HUD event log: append entries, auto-scroll, cap history.
 */
export function useLogs(initialEntries = []) {
  const [logs, setLogs] = useState(initialEntries);
  const logRef = useRef(null);
  const counter = useRef(initialEntries.length);

  // Auto-scroll to bottom on new entry
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  const addLog = (text, type = "info") => {
    const now = new Date();
    const time = [now.getHours(), now.getMinutes(), now.getSeconds()]
      .map((n) => String(n).padStart(2, "0"))
      .join(":");
    setLogs((prev) => [
      ...prev.slice(-40),
      { id: counter.current++, time, text, type },
    ]);
  };

  return { logs, logRef, addLog };
}
