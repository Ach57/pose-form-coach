"""Text-to-speech output for Edith.

On macOS, uses the built-in ``say`` command (Siri voices, fully offline, no
dependencies).  Falls back to ``pyttsx3`` on other platforms.

Usage
-----
    from src.interfaces.voice_output import speak
    speak("Starting overhead press. Get into position.")
"""

from __future__ import annotations

import subprocess
import asyncio
import edge_tts


async def _speak_async(text: str, voice: str = "en-US-JennyNeural") -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save("/tmp/edith_tts.mp3")
    subprocess.run(["afplay", "/tmp/edith_tts.mp3"], check=False)

def speak(text: str) -> None:
    """Speak *text* aloud and block until playback is done."""
    print(f"[Edith] {text}")
    asyncio.run(_speak_async(text))



# def speak(text: str) -> None:
#     """Speak *text* aloud and block until playback is done."""
#     print(f"[Edith] {text}")
#     if sys.platform == "darwin":
#         # Use `say` directly — osascript hangs on repeated calls.
#         # Timeout of 15 s prevents blocking the voice loop indefinitely.
#         try:
#             subprocess.run(["say", "-v", "ElevenLabs", "-r", "175", text], check=False, timeout=15)
#         except subprocess.TimeoutExpired:
#             print("[Edith][TTS] say timed out, skipping.")
#     else:
#         try:
#             import pyttsx3
#             engine = pyttsx3.init()
#             engine.setProperty("rate", 165)
#             engine.say(text)
#             engine.runAndWait()
#         except Exception as exc:
#             print(f"[Edith][TTS error] {exc}")
