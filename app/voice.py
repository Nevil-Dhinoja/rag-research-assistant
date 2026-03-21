from gtts import gTTS
import tempfile
import os

def speak(text: str, lang: str = "en") -> str:
    """Convert answer text to MP3, return path."""
    clean = text.replace("[Page", "Page").replace("]", ".")
    tts   = gTTS(text=clean[:800], lang=lang)
    path  = tempfile.mktemp(suffix=".mp3")
    tts.save(path)
    return path