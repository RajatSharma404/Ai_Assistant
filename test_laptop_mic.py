#!/usr/bin/env python3
"""
Test laptop microphone with voice recognition
Run this after enabling audio passthrough in dev container
"""

import speech_recognition as sr
import sys

def test_microphone_detection():
    """Check if microphones are detected"""
    print("🎤 Checking for microphones...")
    print("="*50)
    
    try:
        mics = sr.Microphone.list_microphone_names()
        
        if not mics:
            print("❌ No microphones detected!")
            print("\n📋 Possible reasons:")
            print("   1. Container doesn't have audio device access")
            print("   2. Need to configure devcontainer.json")
            print("   3. Running in remote/cloud environment")
            print("\n📖 See LAPTOP_MICROPHONE_SETUP.md for solutions")
            return False
        
        print(f"✅ Found {len(mics)} microphone(s):")
        for i, name in enumerate(mics):
            print(f"   {i}: {name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking microphones: {e}")
        return False

def test_microphone_recording():
    """Test actual microphone recording"""
    print("\n🎙️ Testing Microphone Recording")
    print("="*50)
    
    recognizer = sr.Recognizer()
    
    try:
        print("\n📝 Instructions:")
        print("   1. Choose default microphone (press Enter)")
        print("   2. Wait for ambient noise adjustment")
        print("   3. Speak clearly when prompted")
        print("   4. Your speech will be transcribed\n")
        
        input("Press Enter to continue...")
        
        with sr.Microphone() as source:
            print("\n🔇 Adjusting for ambient noise...")
            print("   (Please wait 2 seconds)")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            
            print("\n🎤 Listening... Speak now!")
            print("   (Recording for 5 seconds)")
            
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                print("\n⏳ Processing speech...")
                
                # Try Google Speech Recognition
                text = recognizer.recognize_google(audio)
                print(f"\n✅ Success! You said: '{text}'")
                
                # Show confidence if available
                try:
                    results = recognizer.recognize_google(audio, show_all=True)
                    if results and 'alternative' in results:
                        confidence = results['alternative'][0].get('confidence', 'N/A')
                        print(f"   Confidence: {confidence}")
                except:
                    pass
                
                return True
                
            except sr.WaitTimeoutError:
                print("\n⚠️ No speech detected (timeout)")
                print("   Try speaking louder or closer to microphone")
                return False
                
            except sr.UnknownValueError:
                print("\n⚠️ Could not understand audio")
                print("   Speech was unclear or too quiet")
                return False
                
            except sr.RequestError as e:
                print(f"\n❌ API Error: {e}")
                print("   Check internet connection")
                return False
                
    except OSError as e:
        print(f"\n❌ Microphone access error: {e}")
        print("\n📋 This usually means:")
        print("   • Container needs audio device passthrough")
        print("   • See LAPTOP_MICROPHONE_SETUP.md for setup")
        return False
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_alternative_methods():
    """Show alternative methods for voice input"""
    print("\n🔄 Alternative Voice Input Methods")
    print("="*50)
    
    print("\n1️⃣ Cloud-Based Speech APIs (Works Now):")
    print("   • Google Speech Recognition ✅")
    print("   • OpenAI Whisper API ✅")
    print("   • Azure Speech Services ✅")
    
    print("\n2️⃣ Web Browser Microphone:")
    print("   • HTML5 Speech Recognition API ✅")
    print("   • Works in Chrome, Edge browsers ✅")
    
    print("\n3️⃣ Audio File Processing:")
    print("   • Record on host → process in container ✅")
    print("   • Upload audio files → transcribe ✅")
    
    print("\n📖 See LAPTOP_MICROPHONE_SETUP.md for code examples")

def main():
    print("="*60)
    print("🎤 Laptop Microphone Test")
    print("="*60)
    
    # Check for microphones
    has_microphone = test_microphone_detection()
    
    if not has_microphone:
        print("\n" + "="*60)
        print("⚠️ No microphone access in container")
        print("="*60)
        test_alternative_methods()
        print("\n📖 Read LAPTOP_MICROPHONE_SETUP.md for detailed setup guide")
        return 1
    
    # Offer to test recording
    print("\n" + "="*60)
    test_choice = input("\nTest microphone recording? (y/n): ").strip().lower()
    
    if test_choice == 'y':
        success = test_microphone_recording()
        
        print("\n" + "="*60)
        if success:
            print("✅ Microphone test completed successfully!")
            print("\nYour laptop microphone is working with the voice system.")
        else:
            print("⚠️ Microphone test encountered issues")
            print("\nTry adjusting microphone settings or position.")
        print("="*60)
    else:
        print("\n✅ Microphone detection successful!")
        print("Run this script again when ready to test recording.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
