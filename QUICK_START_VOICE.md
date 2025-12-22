# 🎤 Quick Start: Speak → Listen → Process

## Choose Your Method

### ⚡ Method 1: Web Interface (EASIEST - Works Immediately!)

**Uses your browser's microphone - no container setup needed!**

1. **Start the web interface:**
   ```bash
   python voice_web_interface.py
   ```

2. **Open in browser:**
   ```
   http://localhost:5000
   ```

3. **Use it:**
   - Click the 🎤 microphone button
   - Allow microphone access when prompted
   - Speak clearly
   - See your speech transcribed instantly!

✅ **Bypasses all container limitations**
✅ **Works in Chrome, Edge, Safari**
✅ **No configuration needed**

---

### 📁 Method 2: Audio File Processing (Simple)

**Record on your laptop, process in container**

1. **Record audio on your laptop:**
   - Windows: Voice Recorder app
   - Mac: QuickTime Player
   - Linux: `arecord -d 5 -f cd recording.wav`
   - Or use your phone!

2. **Save file** to `/workspaces/Ai_Assistant/`

3. **Process it:**
   ```bash
   python simple_voice_processor.py recording.wav
   ```

✅ **No internet needed (except for transcription)**
✅ **Works with any audio file**
✅ **Simple and reliable**

---

### 🔧 Method 3: Full Container Setup (Advanced)

**Enable laptop microphone directly in container**

1. **Rebuild container** with audio access:
   ```bash
   # I already created .devcontainer/devcontainer.json
   # Press F1 → "Dev Containers: Rebuild Container"
   ```

2. **After rebuild, test:**
   ```bash
   python test_laptop_mic.py
   ```

3. **Use built-in voice features:**
   ```python
   import speech_recognition as sr
   
   recognizer = sr.Recognizer()
   with sr.Microphone() as source:
       audio = recognizer.listen(source)
       text = recognizer.recognize_google(audio)
       print(f"You said: {text}")
   ```

---

## 🚀 Recommended: Start with Method 1 (Web Interface)

It's the **fastest way** to get speaking, listening, and processing!

```bash
# Just run this:
python voice_web_interface.py

# Then open: http://localhost:5000
```

## 🎯 Complete Example

Here's a full working example with AI processing:

```python
#!/usr/bin/env python3
"""Complete voice pipeline with AI"""

import speech_recognition as sr

def speak_listen_process():
    # Method 1: Using microphone (after container setup)
    recognizer = sr.Recognizer()
    
    print("🎤 Listening...")
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    print("⏳ Processing...")
    text = recognizer.recognize_google(audio)
    
    print(f"📝 You said: {text}")
    
    # TODO: Add your AI processing here
    # - Send to Gemini/GPT
    # - Execute commands
    # - Generate response
    # - Text-to-speech reply
    
    return text

# Or Method 2: Using audio file
def process_audio_file(filename):
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    
    text = recognizer.recognize_google(audio)
    print(f"📝 Transcribed: {text}")
    
    return text
```

## 📚 What I Created for You

| File | Purpose |
|------|---------|
| `voice_web_interface.py` | 🌐 Web interface (easiest!) |
| `simple_voice_processor.py` | 📁 Process audio files |
| `test_laptop_mic.py` | 🧪 Test microphone access |
| `.devcontainer/devcontainer.json` | ⚙️ Container audio config |
| `LAPTOP_MICROPHONE_SETUP.md` | 📖 Detailed guide |

## ⚡ Try It Now!

```bash
# Start web interface (RECOMMENDED)
python voice_web_interface.py

# Or process an audio file
python simple_voice_processor.py your_recording.wav

# Or test microphone setup
python test_laptop_mic.py
```

## 🔄 Next Steps After Speech Recognition

Once you have the transcribed text, you can:

1. **Send to AI:**
   ```python
   import google.generativeai as genai
   response = genai.GenerativeModel('gemini-pro').generate_content(text)
   ```

2. **Execute Commands:**
   ```python
   if "open browser" in text.lower():
       os.system("start chrome")
   ```

3. **Text-to-Speech Response:**
   ```python
   import pyttsx3
   engine = pyttsx3.init()
   engine.say("I understood: " + text)
   engine.runAndWait()
   ```

4. **Save/Log:**
   ```python
   with open('voice_log.txt', 'a') as f:
       f.write(f"{datetime.now()}: {text}\n")
   ```

---

**🎯 Bottom line:** Run `python voice_web_interface.py` and open your browser. You'll be speaking and processing in under 30 seconds!
