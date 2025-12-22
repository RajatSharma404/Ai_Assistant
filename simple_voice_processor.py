#!/usr/bin/env python3
"""
Simple Voice Processor - Audio File Version
Record on your laptop, process here

Usage:
    1. Record audio on your laptop (any app)
    2. Save as .wav or .mp3
    3. Run: python simple_voice_processor.py your_audio.wav
"""

import sys
import speech_recognition as sr
from pathlib import Path

def process_audio_file(audio_path):
    """Process audio file and return transcription"""
    print(f"\n🎤 Processing audio file: {audio_path}")
    print("="*50)
    
    if not Path(audio_path).exists():
        print(f"❌ Error: File not found: {audio_path}")
        return None
    
    recognizer = sr.Recognizer()
    
    try:
        # Load audio file
        print("📂 Loading audio file...")
        with sr.AudioFile(audio_path) as source:
            # Optional: adjust for ambient noise
            print("🔇 Adjusting for noise...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            # Record the audio
            print("📝 Reading audio data...")
            audio = recognizer.record(source)
        
        # Transcribe
        print("⏳ Transcribing (using Google Speech Recognition)...")
        text = recognizer.recognize_google(audio)
        
        print("\n✅ Success!")
        print("="*50)
        print(f"📝 Transcription: \"{text}\"")
        print("="*50)
        
        return text
        
    except sr.UnknownValueError:
        print("❌ Could not understand the audio")
        print("   Try recording with less background noise")
        return None
        
    except sr.RequestError as e:
        print(f"❌ API Error: {e}")
        print("   Check your internet connection")
        return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("="*60)
    print("🎤 Simple Voice Processor")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("\n❌ No audio file provided!")
        print("\n📋 Usage:")
        print("   python simple_voice_processor.py <audio_file>")
        print("\n📝 Example:")
        print("   python simple_voice_processor.py recording.wav")
        print("\n💡 How to record:")
        print("   1. On Windows: Use Voice Recorder app")
        print("   2. On Mac: Use QuickTime Player")
        print("   3. On Linux: Use arecord or Audacity")
        print("   4. Or use your phone's voice recorder")
        print("\n   Then copy the file to this directory!")
        print("="*60)
        return 1
    
    audio_file = sys.argv[1]
    text = process_audio_file(audio_file)
    
    if text:
        print("\n💬 What would you like to do with this text?")
        print("   • Process with AI")
        print("   • Execute command")
        print("   • Save to file")
        print("   • etc.")
        
        # TODO: Add your processing logic here
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
