from pydantic import BaseModel
from typing import Optional
from constants.state import *

class CommandMessage(BaseModel):
    """Incoming natural-language command from the frontend (mirrors edith.py)."""
    command: str  # e.g. "start ohp", "switch to squat", "stop", "status"

class StateChangeEvent(BaseModel):
    """Sent to frontend when system state changes."""
    event: EventType = EventType.STATE_CHANGE
    state: SystemState
    model: Optional[str] = None  # registry key e.g. "ohp", "squat"

class StatsEvent(BaseModel):
    """Sent to frontend periodically while a model is active."""
    event: ModelStatsEvent = ModelStatsEvent.STATS
    confidence: int
    fps:        int
    latency:    int

class LogEvent(BaseModel):
    """Sent to frontend to append an entry to the HUD event log."""
    event: LogEventType = LogEventType.LOG
    text: str
    type: Logstate = Logstate.INFO

class FrameEvent(BaseModel):
    """Sent to frontend with a base64-encoded annotated JPEG frame."""
    event: FrameEventType = FrameEventType.FRAME
    data: str  # base64 JPEG

class WakeEvent(BaseModel):
    """Sent to frontend to reflect the backend voice-loop wake state."""
    event: WakeType = WakeType.WAKE_STATE
    state: WakeState  # mirrors useVoice wakeState
    transcript: Optional[str] = None              # live partial transcript
