import { useEffect, useRef, useCallback } from "react";

const WS_URL = "ws://localhost:8000/ws";

/**
 * useWebSocket
 * Connects to the FastAPI backend.
 * Calls onMessage(data) with parsed JSON from the server.
 * Exposes sendCommand(text) to dispatch a natural-language command string.
 *
 * The callback ref pattern ensures the WS is opened exactly once regardless
 * of how many times the parent re-renders with a new inline callback.
 */
export function useWebSocket(onMessage) {
  const wsRef        = useRef(null);
  const reconnectRef = useRef(null);

  // Keep the callback in a ref so the socket is never reopened
  // just because the parent re-rendered with a new inline function.
  const onMessageRef = useRef(onMessage);
  useEffect(() => { onMessageRef.current = onMessage; });

  // connect is stable (no deps) — only created once
  const connect = useCallback(() => {
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      console.log("[EDITH] WebSocket connected");
      clearTimeout(reconnectRef.current);
    };

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        onMessageRef.current(data);
      } catch (err) {
        console.warn("[EDITH] WS parse error:", err);
      }
    };

    ws.onclose = () => {
      console.warn("[EDITH] WebSocket closed — retrying in 3s");
      reconnectRef.current = setTimeout(connect, 3000);
    };

    ws.onerror = (err) => console.error("[EDITH] WS error:", err);

    wsRef.current = ws;
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const sendCommand = useCallback((text) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command: text }));
    } else {
      console.warn("[EDITH] WS not open, command dropped:", text);
    }
  }, []);

  return { sendCommand };
}
