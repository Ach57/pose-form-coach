from pydantic import BaseModel
from typing import Literal, Optional

class CommandMessage(BaseModel):
    """Incoming natural-language command from the frontend (mirrors edith.py)."""
    command: str  # e.g. "start ohp", "switch to squat", "stop", "status"

class StateChangeEvent(BaseModel):
    """Sent to frontend when system state changes."""
    event: Literal["state_change"] = "state_change"
    state: Literal["idle", "booting", "active", "shutdown"]
    model: Optional[str] = None  # registry key e.g. "ohp", "squat"

class StatsEvent(BaseModel):
    """Sent to frontend periodically while a model is active."""
    event: Literal["stats"] = "stats"
    confidence: int
    fps:        int
    latency:    int

class LogEvent(BaseModel):
    """Sent to frontend to append an entry to the HUD event log."""
    event: Literal["log"] = "log"
    text: str
    type: Literal["info", "warn", "error", "success", "voice"] = "info"

class FrameEvent(BaseModel):
    """Sent to frontend with a base64-encoded annotated JPEG frame."""
    event: Literal["frame"] = "frame"
    data: str  # base64 JPEG

class WakeEvent(BaseModel):
    """Sent to frontend to reflect the backend voice-loop wake state."""
    event: Literal["wake_state"] = "wake_state"
    state: Literal["idle", "awake", "listening"]  # mirrors useVoice wakeState
    transcript: Optional[str] = None              # live partial transcript
