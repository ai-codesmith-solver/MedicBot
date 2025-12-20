import os
import tempfile
from gtts import gTTS
from playsound import playsound

# to convert the text to speech
def speak(text: str):
    try:
        # if empty text
        if not text.strip():
            print("⚠️ Error: Empty text provided for TTS.")
            return
        
        tts = gTTS(text=text, lang="en")
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tts.save(tmp.name)
        tmp.close()
        playsound(tmp.name)
        os.unlink(tmp.name)
    except Exception as e:
        print(f"❌ TTS Error: {e}")