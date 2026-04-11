import speech_recognition as sr

def listen_once() -> str:
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Listening . . .")
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        return ""

if __name__ == "__main__":
    listen_once()