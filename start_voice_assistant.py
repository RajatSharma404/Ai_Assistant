#!/usr/bin/env python3
"""
Always Active Voice Assistant - Quick Start
Run this to start the voice assistant with wake word detection
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def main():
    print("="*60)
    print("🎤 Always Active Voice Assistant")
    print("="*60)
    print()
    
    # Check if backend is running
    print("📋 Checking backend status...")
    result = subprocess.run(
        "ps aux | grep 'modern_web_backend.py' | grep -v grep",
        shell=True,
        capture_output=True,
        text=True
    )
    
    backend_running = bool(result.stdout.strip())
    
    if backend_running:
        print("✅ Backend is already running")
    else:
        print("🚀 Starting backend...")
        subprocess.Popen(
            ["nohup", "python", "modern_web_backend.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent
        )
        print("⏳ Waiting for backend to start...")
        time.sleep(3)
        print("✅ Backend started")
    
    print()
    print("="*60)
    print("🎯 How to Use Always-Active Voice")
    print("="*60)
    print()
    print("1. Open: http://localhost:5000/voice")
    print("2. Click the 💤 button to activate")
    print("3. Say one of these wake words:")
    print("   • Hey Daddy")
    print("   • OK Daddy")
    print("   • Hey Assistant")
    print("4. Wait for 'Yes, I am listening'")
    print("5. Speak your command")
    print("6. Listen to the response")
    print()
    print("💡 The assistant stays active and listens continuously!")
    print("="*60)
    print()
    
    # Open browser
    url = "http://localhost:5000/voice"
    print(f"🌐 Opening {url} in your browser...")
    try:
        webbrowser.open(url)
        print("✅ Browser opened")
    except:
        print("⚠️  Please open manually: http://localhost:5000/voice")
    
    print()
    print("Press Ctrl+C to stop the assistant")

if __name__ == "__main__":
    try:
        main()
        # Keep script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Stopping assistant...")
        subprocess.run("pkill -f modern_web_backend.py", shell=True)
        print("✅ Assistant stopped")
