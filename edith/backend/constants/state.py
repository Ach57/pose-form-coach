from enum import Enum

class SystemState(str, Enum):    
    IDLE = "idle"
    BOOTING = "booting"
    ACTIVE = "active"
    SHUTDOWN = "shutdown"
    
class Logstate(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    SUCCESS = "success"
    VOICE = "voice"

class WakeState(str, Enum):
    IDLE = "idle"
    AWAKE = "awake"
    LISTENING = "listening"
    
class EventType(str, Enum):
    STATE_CHANGE = "state_change"

class WakeType(str, Enum):
    WAKE_STATE = "wake_state"

class ModelStatsEvent(str, Enum):
    STATS = "stats"

class LogEventType(str, Enum):
    LOG = "log"

class FrameEventType(str, Enum):
    FRAME = "frame"

class IntentType(str, Enum):
    START = "start"
    STOP = "stop"
    SWITCH = "switch"
    STATUS = "status"
    SHUTDOWN = "shutdown"
    UNKNOWN = "unknown"    
