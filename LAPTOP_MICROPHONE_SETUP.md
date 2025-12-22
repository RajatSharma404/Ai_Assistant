# Laptop Microphone Setup Guide 🎤

## Current Situation

Your voice system software is **fully installed and working**, but the Docker dev container doesn't have access to your laptop's microphone because:

- ❌ `/dev/snd` (audio devices) not accessible in container
- ❌ No sound cards detected
- ❌ PulseAudio not connected

This is **expected** - Docker containers are isolated by default and need explicit configuration to access host audio devices.

## Solution Options

### Option 1: Enable Audio Device Passthrough (Recommended for Local Development)

#### For VS Code Dev Containers

1. **Create `.devcontainer/devcontainer.json` in your workspace:**

```json
{
  "name": "AI Assistant with Audio",
  "image": "mcr.microsoft.com/devcontainers/base:ubuntu",
  
  "runArgs": [
    "--device=/dev/snd:/dev/snd",
    "--group-add=audio"
  ],
  
  "mounts": [
    "source=/run/user/1000/pulse,target=/run/user/1000/pulse,type=bind"
  ],
  
  "containerEnv": {
    "PULSE_SERVER": "unix:/run/user/1000/pulse/native"
  },
  
  "postCreateCommand": "sudo apt-get update && sudo apt-get install -y espeak espeak-ng portaudio19-dev python3-pyaudio alsa-utils pulseaudio && pip install -r requirements.txt",
  
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python"
      ]
    }
  }
}
```

2. **Rebuild Container:**
   - Press `F1` → "Dev Containers: Rebuild Container"
   - Or: "Dev Containers: Reopen in Container"

#### For Manual Docker Run

If running Docker manually, use these flags:

```bash
docker run -it \
  --device /dev/snd:/dev/snd \
  --group-add audio \
  -v /run/user/1000/pulse:/run/user/1000/pulse \
  -e PULSE_SERVER=unix:/run/user/1000/pulse/native \
  your-image
```

#### Linux-Specific Setup

Ensure your user is in the audio group:
```bash
# On host machine (not in container)
sudo usermod -a -G audio $USER
# Log out and back in for changes to take effect
```

---

### Option 2: Use Cloud-Based Speech APIs (Works Now!)

No container configuration needed - works immediately:

#### Google Speech Recognition (Free)
```python
import speech_recognition as sr

recognizer = sr.Recognizer()

# Use audio file or record on host, then process
with sr.AudioFile('audio.wav') as source:
    audio = recognizer.record(source)
    text = recognizer.recognize_google(audio)
    print(f"You said: {text}")
```

#### OpenAI Whisper API
```python
import openai

# Record audio on host machine, then transcribe
with open("audio.mp3", "rb") as audio_file:
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print(transcript.text)
```

#### Microsoft Azure Speech
```python
import azure.cognitiveservices.speech as speechsdk

speech_config = speechsdk.SpeechConfig(
    subscription="YOUR_KEY",
    region="YOUR_REGION"
)
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
```

---

### Option 3: Web-Based Voice Input

Build a simple web interface that uses browser's microphone API:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Voice Input</title>
</head>
<body>
    <button id="startBtn">🎤 Start Recording</button>
    <p id="transcript"></p>
    
    <script>
        const startBtn = document.getElementById('startBtn');
        const transcript = document.getElementById('transcript');
        
        if ('webkitSpeechRecognition' in window) {
            const recognition = new webkitSpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            
            recognition.onresult = (event) => {
                const text = event.results[0][0].transcript;
                transcript.textContent = text;
                
                // Send to backend API
                fetch('/api/voice', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text})
                });
            };
            
            startBtn.onclick = () => recognition.start();
        }
    </script>
</body>
</html>
```

---

### Option 4: Record Audio on Host, Process in Container

**Simple workflow:**

1. **Record on host** using any tool:
   ```bash
   # On your laptop (outside container)
   arecord -f cd -d 5 recording.wav
   ```

2. **Process in container:**
   ```python
   import speech_recognition as sr
   
   recognizer = sr.Recognizer()
   with sr.AudioFile('/workspaces/Ai_Assistant/recording.wav') as source:
       audio = recognizer.record(source)
       text = recognizer.recognize_google(audio)
       print(f"Transcribed: {text}")
   ```

---

## Quick Test After Setup

Once you've enabled audio access, test with:

```bash
# Test microphone detection
python3 -c "
import speech_recognition as sr
mics = sr.Microphone.list_microphone_names()
print(f'Found {len(mics)} microphone(s):')
for i, name in enumerate(mics):
    print(f'  {i}: {name}')
"

# Test recording
python3 -c "
import speech_recognition as sr
r = sr.Recognizer()
with sr.Microphone() as source:
    print('Say something!')
    audio = r.listen(source, timeout=5)
    print('Processing...')
    text = r.recognize_google(audio)
    print(f'You said: {text}')
"
```

---

## Test Script with Microphone

Here's a working test script once audio is enabled:

```python
#!/usr/bin/env python3
"""Test laptop microphone with voice recognition"""

import speech_recognition as sr

def test_microphone():
    recognizer = sr.Recognizer()
    
    # List available microphones
    print("🎤 Available microphones:")
    mics = sr.Microphone.list_microphone_names()
    for i, name in enumerate(mics):
        print(f"  {i}: {name}")
    
    if not mics:
        print("❌ No microphones found!")
        return
    
    # Test recording
    print("\n🎙️ Testing microphone (speak for 5 seconds)...")
    
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        
        print("Listening... Speak now!")
        audio = recognizer.listen(source, timeout=5)
        
        print("Processing speech...")
        try:
            text = recognizer.recognize_google(audio)
            print(f"✅ You said: '{text}'")
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
        except sr.RequestError as e:
            print(f"❌ API error: {e}")

if __name__ == "__main__":
    test_microphone()
```

Save as `test_laptop_mic.py` and run: `python3 test_laptop_mic.py`

---

## Current Working Features (No Mic Needed)

Even without microphone access, these features work now:

✅ **Text-to-Speech**
```python
import pyttsx3
engine = pyttsx3.init()
engine.say("Hello from your AI assistant!")
engine.runAndWait()
```

✅ **Audio File Processing**
```python
import speech_recognition as sr
recognizer = sr.Recognizer()
with sr.AudioFile('audio.wav') as source:
    audio = recognizer.record(source)
    text = recognizer.recognize_google(audio)
```

✅ **Voice Module Testing**
```python
from ai_assistant.voice.advanced_voice import VoiceProfileManager
vpm = VoiceProfileManager()  # Works for testing
```

---

## Recommended Approach

**For Development in GitHub Codespaces/Remote Container:**
- 👍 **Use cloud speech APIs** (Google, Azure, OpenAI Whisper)
- 👍 **Web-based voice input** through browser
- 👍 **Record audio on host**, process in container

**For Local Docker Development:**
- 👍 **Configure dev container** with audio passthrough
- 👍 **Use host microphone** directly in container

**For Production Deployment:**
- 👍 **Web interface** with browser microphone API
- 👍 **Mobile app** integration
- 👍 **Cloud speech services** for scalability

---

## Need Help?

**If you're using:**
- **GitHub Codespaces**: Option 2 (Cloud APIs) or Option 3 (Web Interface)
- **VS Code Remote Container**: Option 1 (Audio Passthrough)
- **Local Docker**: Option 1 (Audio Passthrough)
- **Direct Python**: Everything already works!

Let me know your setup and I can provide specific configuration!

---

**Note:** The voice system software is fully installed and ready. You just need to choose how to connect your laptop's microphone to the container.
