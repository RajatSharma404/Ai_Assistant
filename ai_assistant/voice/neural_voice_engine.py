"""
Neural Voice Engine - Google Assistant Quality Voice Synthesis
Combines Edge-TTS (Microsoft Neural), Coqui TTS (offline), and fallback options
Implements prosody, emotion, and context-aware speech generation
"""

import asyncio
import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from enum import Enum
import threading
import time

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    from TTS.api import TTS as CoquiTTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from utils.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class VoiceGender(Enum):
    """Voice gender options"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class SpeakingStyle(Enum):
    """Speaking style options for natural conversation"""
    NORMAL = "normal"
    EXCITED = "excited"
    CALM = "calm"
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    CHEERFUL = "cheerful"


class NeuralVoiceEngine:
    """
    High-quality neural voice synthesis engine
    Matches Google Assistant's natural, human-like voice quality
    """
    
    def __init__(self, cache_dir: str = "data/voice_cache", gpu: bool = False):
        """
        Initialize the neural voice engine
        
        Args:
            cache_dir: Directory for caching synthesized audio
            gpu: Enable GPU acceleration if available
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.gpu = gpu
        
        # Voice configurations
        self.edge_voices = {
            'en': {
                'female': 'en-US-AriaNeural',
                'male': 'en-US-GuyNeural',
                'neutral': 'en-US-AriaNeural'
            },
            'hi': {
                'female': 'hi-IN-SwaraNeural',
                'male': 'hi-IN-MadhurNeural',
                'neutral': 'hi-IN-SwaraNeural'
            },
            'en-GB': {
                'female': 'en-GB-SoniaNeural',
                'male': 'en-GB-RyanNeural',
                'neutral': 'en-GB-SoniaNeural'
            },
            'es': {
                'female': 'es-ES-ConchitaNeural',
                'male': 'es-ES-AlvaroNeural',
                'neutral': 'es-ES-ConchitaNeural'
            },
            'fr': {
                'female': 'fr-FR-DeniseNeural',
                'male': 'fr-FR-HenriNeural',
                'neutral': 'fr-FR-DeniseNeural'
            }
        }
        
        # Initialize engines
        self.edge_tts_available = EDGE_TTS_AVAILABLE
        self.coqui_tts = None
        self.pyttsx3_engine = None
        self.current_voice = 'en-US-AriaNeural'
        
        self._initialize_engines()
        
    def _initialize_engines(self):
        """Initialize all available TTS engines"""
        # Edge-TTS (online, best quality)
        if self.edge_tts_available:
            logger.info("✅ Edge-TTS available (Microsoft Neural Voices)")
        else:
            logger.warning("⚠️ Edge-TTS not available. Install: pip install edge-tts")
        
        # Coqui TTS (offline, good quality)
        if COQUI_AVAILABLE:
            try:
                device = "cuda" if self.gpu else "cpu"
                logger.info(f"🚀 Loading Coqui TTS (device: {device})...")
                # Lazy load to avoid startup delays
                self.coqui_available = True
                logger.info("✅ Coqui TTS available for offline synthesis")
            except Exception as e:
                logger.warning(f"⚠️ Coqui TTS initialization failed: {e}")
                self.coqui_available = False
        else:
            logger.warning("⚠️ Coqui TTS not available. Install: pip install TTS")
            self.coqui_available = False
        
        # pyttsx3 (fallback)
        if PYTTSX3_AVAILABLE:
            try:
                self.pyttsx3_engine = pyttsx3.init()
                self.pyttsx3_engine.setProperty('rate', 150)  # Speaking rate
                logger.info("✅ pyttsx3 available as fallback")
            except Exception as e:
                logger.warning(f"⚠️ pyttsx3 initialization failed: {e}")
    
    async def synthesize_edge_tts(
        self,
        text: str,
        language: str = 'en',
        gender: VoiceGender = VoiceGender.FEMALE,
        rate: float = 0.0,  # -50 to 50, 0 is normal
        pitch: float = 0.0,  # -50 to 50, 0 is normal
        output_file: Optional[str] = None
    ) -> Optional[str]:
        """
        Synthesize speech using Edge-TTS (Microsoft Neural Voices)
        Best quality, matches Google Assistant
        
        Args:
            text: Text to synthesize
            language: Language code (en, hi, en-GB, es, fr, etc.)
            gender: Voice gender
            rate: Speaking rate adjustment
            pitch: Pitch adjustment
            output_file: Output MP3 file path
        
        Returns:
            Path to saved audio file or None if failed
        """
        if not self.edge_tts_available or not text:
            return None
        
        try:
            # Select voice
            voice_key = gender.value if gender.value != 'neutral' else 'female'
            voice = self.edge_voices.get(language, self.edge_voices['en']).get(
                voice_key, 'en-US-AriaNeural'
            )
            
            # Cache key
            cache_key = f"{text[:50].replace(' ', '_')}_{language}_{gender.value}.mp3"
            cache_file = self.cache_dir / cache_key
            
            # Return cached if available
            if cache_file.exists():
                logger.debug(f"Using cached audio: {cache_file}")
                return str(cache_file)
            
            # Synthesize
            output_file = output_file or str(cache_file)
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            
            await communicate.save(output_file)
            logger.info(f"✅ Synthesized: {text[:50]}... -> {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"❌ Edge-TTS synthesis failed: {e}")
            return None
    
    def synthesize_edge_tts_sync(
        self,
        text: str,
        language: str = 'en',
        gender: VoiceGender = VoiceGender.FEMALE,
        rate: float = 0.0,
        pitch: float = 0.0,
        output_file: Optional[str] = None
    ) -> Optional[str]:
        """Synchronous wrapper for Edge-TTS"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.synthesize_edge_tts(text, language, gender, rate, pitch, output_file)
        )
    
    def synthesize_coqui_tts(
        self,
        text: str,
        language: str = 'en',
        speaker: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> Optional[str]:
        """
        Synthesize speech using Coqui TTS (offline)
        Good quality, works without internet
        
        Args:
            text: Text to synthesize
            language: Language code
            speaker: Speaker ID for multi-speaker models
            output_file: Output WAV file path
        
        Returns:
            Path to saved audio file or None if failed
        """
        if not COQUI_AVAILABLE or not text:
            return None
        
        try:
            if self.coqui_tts is None:
                device = "cuda" if self.gpu else "cpu"
                self.coqui_tts = CoquiTTS(gpu=self.gpu)
            
            # Cache key
            cache_key = f"{text[:50].replace(' ', '_')}_{language}_coqui.wav"
            cache_file = self.cache_dir / cache_key
            
            if cache_file.exists():
                logger.debug(f"Using cached audio: {cache_file}")
                return str(cache_file)
            
            # Synthesize
            output_file = output_file or str(cache_file)
            self.coqui_tts.tts_to_file(
                text=text,
                file_path=output_file,
                language=language,
                speaker_idx=speaker if speaker else "p225"
            )
            
            logger.info(f"✅ Coqui TTS synthesized: {text[:50]}...")
            return output_file
            
        except Exception as e:
            logger.error(f"❌ Coqui TTS synthesis failed: {e}")
            return None
    
    def synthesize_pyttsx3_fallback(
        self,
        text: str,
        language: str = 'en',
        output_file: Optional[str] = None
    ) -> Optional[str]:
        """
        Fallback to pyttsx3 (lower quality but always available)
        
        Args:
            text: Text to synthesize
            language: Language code
            output_file: Output file path
        
        Returns:
            Path to saved audio file or None if failed
        """
        if not self.pyttsx3_engine or not text:
            return None
        
        try:
            # Cache key
            cache_key = f"{text[:50].replace(' ', '_')}_{language}_fallback.mp3"
            cache_file = self.cache_dir / cache_key
            
            if cache_file.exists():
                return str(cache_file)
            
            output_file = output_file or str(cache_file)
            self.pyttsx3_engine.save_to_file(text, output_file)
            self.pyttsx3_engine.runAndWait()
            
            logger.info(f"✅ pyttsx3 synthesized: {text[:50]}...")
            return output_file
            
        except Exception as e:
            logger.error(f"❌ pyttsx3 synthesis failed: {e}")
            return None
    
    def synthesize(
        self,
        text: str,
        language: str = 'en',
        gender: VoiceGender = VoiceGender.FEMALE,
        style: SpeakingStyle = SpeakingStyle.NORMAL,
        prefer_online: bool = True,
        output_file: Optional[str] = None
    ) -> Optional[str]:
        """
        Synthesize speech with automatic fallback
        Tries Edge-TTS first (best quality), then Coqui, then pyttsx3
        
        Args:
            text: Text to synthesize
            language: Language code
            gender: Voice gender
            style: Speaking style
            prefer_online: Try online service first
            output_file: Output file path
        
        Returns:
            Path to saved audio file
        """
        if not text:
            return None
        
        # Rate adjustments based on style
        rate_adjustments = {
            SpeakingStyle.EXCITED: 20,
            SpeakingStyle.CALM: -20,
            SpeakingStyle.PROFESSIONAL: 0,
            SpeakingStyle.FRIENDLY: 10,
            SpeakingStyle.CHEERFUL: 25,
            SpeakingStyle.NORMAL: 0
        }
        
        rate = rate_adjustments.get(style, 0)
        
        # Try in preferred order
        if prefer_online and self.edge_tts_available:
            result = self.synthesize_edge_tts_sync(
                text, language, gender, rate, output_file=output_file
            )
            if result:
                return result
        
        if COQUI_AVAILABLE:
            result = self.synthesize_coqui_tts(text, language, output_file=output_file)
            if result:
                return result
        
        # Final fallback
        return self.synthesize_pyttsx3_fallback(text, language, output_file)
    
    def clear_cache(self, older_than_hours: int = 24):
        """Clear old cached audio files"""
        try:
            current_time = time.time()
            for cache_file in self.cache_dir.glob("*.mp3"):
                if (current_time - cache_file.stat().st_mtime) > (older_than_hours * 3600):
                    cache_file.unlink()
            logger.info(f"Cleared cache older than {older_than_hours} hours")
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")


# Global instance
_engine_instance = None


def get_neural_voice_engine(cache_dir: str = "data/voice_cache", gpu: bool = False) -> NeuralVoiceEngine:
    """Get or create the neural voice engine instance"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = NeuralVoiceEngine(cache_dir=cache_dir, gpu=gpu)
    return _engine_instance


# Example usage
if __name__ == "__main__":
    engine = get_neural_voice_engine(gpu=False)
    
    # Test synthesis
    text = "Hello! I'm your AI assistant with neural voice quality."
    print(f"Synthesizing: {text}")
    
    audio_file = engine.synthesize(
        text,
        language='en',
        gender=VoiceGender.FEMALE,
        style=SpeakingStyle.FRIENDLY
    )
    
    print(f"✅ Audio saved to: {audio_file}")
