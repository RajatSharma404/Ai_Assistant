"""
Integration module for Google Assistant quality voice features
Drop-in replacement for existing voice modules
Place this in your main app.py or import it
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple
from enum import Enum

# Set up logging
try:
    from utils.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Import the neural modules
from modules.neural_voice_engine import (
    get_neural_voice_engine,
    NeuralVoiceEngine,
    VoiceGender,
    SpeakingStyle
)

from modules.advanced_speech_recognizer import (
    get_advanced_speech_recognizer,
    AdvancedSpeechRecognizer
)

from modules.wake_word_detector import (
    get_wake_word_manager,
    WakeWordManager,
    WakeWordDetectionMode
)


class GoogleAssistantVoiceIntegration:
    """
    Complete voice integration matching Google Assistant quality
    Easy drop-in replacement for existing voice modules
    """
    
    def __init__(
        self,
        whisper_api_key: Optional[str] = None,
        google_cloud_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        gpu_acceleration: bool = False,
        cache_audio: bool = True
    ):
        """
        Initialize complete Google Assistant quality voice system
        
        Args:
            whisper_api_key: OpenAI API key for speech recognition
            google_cloud_key: Google Cloud API key
            gemini_api_key: Google Gemini API key for AI responses
            gpu_acceleration: Enable GPU for TTS/ASR
            cache_audio: Cache synthesized audio
        """
        # Load from environment if not provided
        self.whisper_api_key = whisper_api_key or os.getenv("OPENAI_API_KEY")
        self.google_cloud_key = google_cloud_key or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        
        self.gpu_acceleration = gpu_acceleration
        self.cache_audio = cache_audio
        
        # Initialize modules
        logger.info("🚀 Initializing Google Assistant Quality Voice System...")
        
        self.voice_engine = get_neural_voice_engine(
            cache_dir="data/voice_cache",
            gpu=gpu_acceleration
        )
        logger.info("✅ Neural Voice Engine initialized")
        
        self.recognizer = get_advanced_speech_recognizer(
            whisper_api_key=self.whisper_api_key,
            google_cloud_key=self.google_cloud_key
        )
        logger.info("✅ Advanced Speech Recognizer initialized")
        
        self.wake_word_manager = get_wake_word_manager(
            detection_mode=WakeWordDetectionMode.ALWAYS_ON
        )
        logger.info("✅ Wake Word Manager initialized")
        
        # State
        self.is_active = False
        self.current_language = "en"
        self.voice_gender = VoiceGender.FEMALE
        self.speaking_style = SpeakingStyle.FRIENDLY
        
        logger.info("✅ Google Assistant Voice System Ready!")
    
    def speak(
        self,
        text: str,
        language: str = "en",
        gender: Optional[VoiceGender] = None,
        style: Optional[SpeakingStyle] = None,
        play_audio: bool = True
    ) -> Optional[str]:
        """
        Speak text with neural voice
        
        Args:
            text: Text to synthesize
            language: Language code
            gender: Voice gender (uses default if None)
            style: Speaking style (uses default if None)
            play_audio: Automatically play the audio
        
        Returns:
            Path to audio file or None if failed
        """
        gender = gender or self.voice_gender
        style = style or self.speaking_style
        
        logger.info(f"🎤 Speaking: {text[:50]}...")
        
        audio_file = self.voice_engine.synthesize(
            text=text,
            language=language,
            gender=gender,
            style=style,
            prefer_online=True
        )
        
        if audio_file and play_audio:
            self._play_audio(audio_file)
        
        return audio_file
    
    def listen(
        self,
        language: str = "en",
        timeout: int = 15,
        context: Optional[str] = None
    ) -> Tuple[Optional[str], float]:
        """
        Listen for user speech and recognize it
        
        Args:
            language: Language code
            timeout: Maximum listening time in seconds
            context: Optional context for better accuracy
        
        Returns:
            Tuple of (recognized_text, confidence)
        """
        logger.info(f"🎙️ Listening ({language})...")
        
        try:
            import speech_recognition as sr
            
            with sr.Microphone() as source:
                text, confidence, model = self.recognizer.recognize(
                    audio_input=source,
                    language=language,
                    context=context
                )
            
            if text:
                logger.info(f"✅ Recognized: {text} (confidence: {confidence:.2%})")
                return text, confidence
            else:
                logger.warning("Could not understand speech")
                return None, 0.0
                
        except Exception as e:
            logger.error(f"Listening failed: {e}")
            return None, 0.0
    
    def set_wake_words(self, wake_words: list[str]):
        """
        Set custom wake words
        
        Args:
            wake_words: List of wake words (e.g., ["hey assistant", "ok assistant"])
        """
        self.wake_word_manager.detector.wake_words = wake_words
        logger.info(f"Set wake words: {wake_words}")
    
    def start_listening(self):
        """Start continuous listening for wake words"""
        logger.info("🎤 Starting wake word detection...")
        self.wake_word_manager.start()
        self.is_active = True
    
    def stop_listening(self):
        """Stop listening for wake words"""
        logger.info("⏸️ Stopping wake word detection...")
        self.wake_word_manager.stop()
        self.is_active = False
    
    def on_wake_word_detected(self, callback):
        """
        Register callback for wake word detection
        
        Args:
            callback: Function to call when wake word detected
                     Signature: callback(wake_word: str, confidence: float)
        """
        self.wake_word_manager.detector.on_wake_word_detected = callback
        logger.info("Wake word callback registered")
    
    def set_voice_preferences(
        self,
        language: str = "en",
        gender: VoiceGender = VoiceGender.FEMALE,
        style: SpeakingStyle = SpeakingStyle.FRIENDLY
    ):
        """
        Set default voice preferences
        
        Args:
            language: Default language
            gender: Default voice gender
            style: Default speaking style
        """
        self.current_language = language
        self.voice_gender = gender
        self.speaking_style = style
        logger.info(f"Voice preferences: {language}, {gender.value}, {style.value}")
    
    def get_stats(self) -> dict:
        """Get system statistics"""
        return {
            "voice_engine": {
                "gpu_enabled": self.gpu_acceleration,
                "cache_enabled": self.cache_audio
            },
            "recognizer": self.recognizer.get_recognition_stats(),
            "wake_word": self.wake_word_manager.get_stats(),
            "active": self.is_active
        }
    
    def _play_audio(self, audio_file: str):
        """Play audio file (platform-independent)"""
        try:
            from playsound import playsound
            playsound(audio_file)
        except ImportError:
            logger.warning("playsound not installed. Install: pip install playsound")
            logger.info(f"Audio file: {audio_file}")
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")


# Global instance
_integration_instance = None


def get_voice_integration(
    whisper_api_key: Optional[str] = None,
    google_cloud_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    gpu_acceleration: bool = False
) -> GoogleAssistantVoiceIntegration:
    """Get or create the voice integration instance"""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = GoogleAssistantVoiceIntegration(
            whisper_api_key=whisper_api_key,
            google_cloud_key=google_cloud_key,
            gemini_api_key=gemini_api_key,
            gpu_acceleration=gpu_acceleration
        )
    return _integration_instance


# Example usage - add to your main app.py
if __name__ == "__main__":
    """
    Example: Integration with your assistant app
    """
    
    # Initialize
    voice = get_voice_integration()
    
    # Set up wake word callback
    def on_wake_word(wake_word, confidence):
        print(f"🎤 Woke up! ({wake_word})")
        voice.speak("I'm listening", style=SpeakingStyle.CHEERFUL)
        
        # Listen for command
        text, confidence = voice.listen()
        
        if text:
            print(f"You said: {text}")
            
            # Process with your AI (Gemini API, etc.)
            response = f"You said: {text}"
            
            # Respond
            voice.speak(response, style=SpeakingStyle.FRIENDLY)
    
    voice.on_wake_word_detected(on_wake_word)
    
    # Start listening
    voice.start_listening()
    print("Listening for wake word...")
    
    # Keep running
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        voice.stop_listening()
        print("Done!")
