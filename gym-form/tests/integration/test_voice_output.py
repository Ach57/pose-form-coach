from __future__ import annotations

from src.interfaces.voice_output import speak


def run_test():    
    print("[Test] Running speak function with text `Hello, my name is Edith`")
    speak("Hello, my name is Edith")


if __name__ == "__main__":
    run_test()