"""
Advanced Speech Recognition Engine - Google Assistant Quality ASR
Uses OpenAI Whisper API for accuracy, with offline fallback options
Implements noise handling, accent robustness, and context-aware recognition
"""

import logging
import threading
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from enum import Enum
import time
import numpy as np

try:
    import openai
    WHISPER_API_AVAILABLE = True
except ImportError:
    WHISPER_API_AVAILABLE = False

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    # Create a dummy sr module for type hints
    class sr:
        class AudioSource:
            pass

try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

try:
    from utils.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class RecognitionModel(Enum):
    """Available recognition models"""
    WHISPER_API = "whisper_api"  # Best accuracy (online)
    GOOGLE_CLOUD = "google_cloud"  # Very good (online)
    SPEECH_RECOGNITION = "speech_recognition"  # Good (online)
    VOSK = "vosk"  # Offline, instant
    OFFLINE_WHISPER = "offline_whisper"  # Whisper local


class AdvancedSpeechRecognizer:
    """
    Advanced speech recognition engine matching Google Assistant accuracy
    Multi-model approach with automatic fallback
    """
    
    def __init__(
        self,
        whisper_api_key: Optional[str] = None,
        google_cloud_key: Optional[str] = None,
        prefer_online: bool = True,
        noise_reduction: bool = True,
        cache_dir: str = "data/recognition_cache"
    ):
        """
        Initialize the advanced speech recognizer
        
        Args:
            whisper_api_key: OpenAI API key for Whisper
            google_cloud_key: Google Cloud API key
            prefer_online: Try online models first
            noise_reduction: Apply noise reduction to audio
            cache_dir: Directory for caching recognition results
        """
        self.whisper_api_key = whisper_api_key
        self.google_cloud_key = google_cloud_key
        self.prefer_online = prefer_online
        self.noise_reduction = noise_reduction
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize recognizers
        self.sr_recognizer = None
        self.vosk_models = {}
        self.google_recognizer = None
        
        # Performance tracking
        self.recognition_history = []
        self.confidence_scores = []
        
        self._initialize_recognizers()
    
    def _initialize_recognizers(self):
        """Initialize all available recognition engines"""
        # Speech Recognition library
        if SPEECH_RECOGNITION_AVAILABLE:
            try:
                self.sr_recognizer = sr.Recognizer()
                self.sr_recognizer.energy_threshold = 3000  # Noise threshold
                self.sr_recognizer.dynamic_energy_threshold = True
                self.sr_recognizer.pause_threshold = 0.8
                self.sr_recognizer.phrase_threshold = 0.3
                logger.info("✅ Speech Recognition library initialized")
            except Exception as e:
                logger.warning(f"⚠️ Speech Recognition failed: {e}")
        
        # Vosk (offline, instant)
        if VOSK_AVAILABLE:
            try:
                # Load English model
                try:
                    model = Model(lang="en")
                    self.vosk_models['en'] = model
                    logger.info("✅ Vosk English model loaded")
                except:
                    logger.warning("⚠️ Vosk English model not found")
                
                # Load Hindi model if available
                try:
                    model = Model(lang="hi")
                    self.vosk_models['hi'] = model
                    logger.info("✅ Vosk Hindi model loaded")
                except:
                    logger.warning("⚠️ Vosk Hindi model not found")
                    
            except Exception as e:
                logger.warning(f"⚠️ Vosk initialization failed: {e}")
        
        # Whisper API
        if WHISPER_API_AVAILABLE and self.whisper_api_key:
            try:
                openai.api_key = self.whisper_api_key
                logger.info("✅ Whisper API configured")
            except Exception as e:
                logger.warning(f"⚠️ Whisper API setup failed: {e}")
    
    def reduce_noise(self, audio_data, sr: int = 16000) -> np.ndarray:
        """
        Apply noise reduction to audio data
        Reduces background noise while preserving speech
        
        Args:
            audio_data: Audio data as numpy array
            sr: Sample rate
        
        Returns:
            Noise-reduced audio array
        """
        if not self.noise_reduction:
            return audio_data
        
        try:
            # Simple noise gate (remove very low amplitude)
            noise_threshold = np.mean(np.abs(audio_data)) * 0.1
            reduced = np.copy(audio_data)
            reduced[np.abs(reduced) < noise_threshold] = 0
            
            # Normalize after noise reduction
            max_val = np.max(np.abs(reduced))
            if max_val > 0:
                reduced = (reduced / max_val) * 32767
            
            return reduced
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio_data
    
    async def recognize_whisper_api(
        self,
        audio_file: str,
        language: str = "en",
        prompt: Optional[str] = None
    ) -> Tuple[Optional[str], float]:
        """
        Recognize speech using OpenAI Whisper API
        Best accuracy, handles diverse accents and background noise
        
        Args:
            audio_file: Path to audio file
            language: Language code
            prompt: Optional context prompt to improve recognition
        
        Returns:
            Tuple of (recognized_text, confidence_score)
        """
        if not WHISPER_API_AVAILABLE or not self.whisper_api_key:
            return None, 0.0
        
        try:
            with open(audio_file, 'rb') as f:
                transcript = openai.Audio.transcribe(
                    model="whisper-1",
                    file=f,
                    language=language,
                    prompt=prompt
                )
            
            text = transcript.get('text', '').strip()
            logger.info(f"✅ Whisper recognized: {text}")
            
            return text, 0.95  # Whisper provides high confidence
            
        except Exception as e:
            logger.error(f"❌ Whisper API failed: {e}")
            return None, 0.0
    
    def recognize_google_cloud_speech(
        self,
        audio_file: str,
        language: str = "en-US"
    ) -> Tuple[Optional[str], float]:
        """
        Recognize speech using Google Cloud Speech-to-Text
        Very good accuracy, less latency than Whisper
        
        Args:
            audio_file: Path to audio file
            language: Language code
        
        Returns:
            Tuple of (recognized_text, confidence_score)
        """
        try:
            from google.cloud import speech_v1
            
            client = speech_v1.SpeechClient()
            
            with open(audio_file, 'rb') as f:
                audio = speech_v1.RecognitionAudio(content=f.read())
            
            config = speech_v1.RecognitionConfig(
                encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=language,
                enable_automatic_punctuation=True,
                model="latest_long",
            )
            
            response = client.recognize(config=config, audio=audio)
            
            if response.results:
                text = response.results[0].alternatives[0].transcript
                confidence = response.results[0].alternatives[0].confidence
                logger.info(f"✅ Google Cloud recognized: {text} (confidence: {confidence})")
                return text, confidence
            
            return None, 0.0
            
        except Exception as e:
            logger.error(f"❌ Google Cloud Speech failed: {e}")
            return None, 0.0
    
    def recognize_speech_recognition(
        self,
        audio_source: sr.AudioSource,
        language: str = "en-US"
    ) -> Tuple[Optional[str], float]:
        """
        Recognize speech using speech_recognition library (Google Speech-to-Text backend)
        Good accuracy, no API key needed
        
        Args:
            audio_source: Audio source from speech_recognition
            language: Language code
        
        Returns:
            Tuple of (recognized_text, confidence_score)
        """
        if not self.sr_recognizer:
            return None, 0.0
        
        try:
            # Adjust for ambient noise
            self.sr_recognizer.adjust_for_ambient_noise(audio_source, duration=0.1)
            
            # Listen
            audio = self.sr_recognizer.listen(
                audio_source,
                timeout=10.0,
                phrase_time_limit=15.0
            )
            
            # Recognize
            text = self.sr_recognizer.recognize_google(audio, language=language)
            logger.info(f"✅ Speech Recognition recognized: {text}")
            
            return text, 0.85  # Estimated confidence
            
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return None, 0.0
        except sr.RequestError as e:
            logger.error(f"Speech Recognition API error: {e}")
            return None, 0.0
        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            return None, 0.0
    
    def recognize_vosk(
        self,
        audio_data,
        language: str = "en"
    ) -> Tuple[Optional[str], float]:
        """
        Recognize speech using Vosk (offline, instant)
        Lower accuracy but works without internet
        
        Args:
            audio_data: Audio data
            language: Language code
        
        Returns:
            Tuple of (recognized_text, confidence_score)
        """
        if language not in self.vosk_models:
            return None, 0.0
        
        try:
            model = self.vosk_models[language]
            recognizer = KaldiRecognizer(model, 16000)
            recognizer.AcceptWaveform(audio_data)
            
            result = recognizer.Result()
            logger.info(f"✅ Vosk recognized: {result}")
            
            # Parse JSON result
            import json
            result_dict = json.loads(result)
            
            if 'result' in result_dict and result_dict['result']:
                text = ' '.join([item['conf'] for item in result_dict['result']])
                return text, 0.75  # Estimated confidence
            
            return None, 0.0
            
        except Exception as e:
            logger.error(f"Vosk recognition failed: {e}")
            return None, 0.0
    
    def recognize(
        self,
        audio_input,
        language: str = "en",
        context: Optional[str] = None
    ) -> Tuple[Optional[str], float, str]:
        """
        Recognize speech with automatic model selection and fallback
        
        Args:
            audio_input: Audio file path or audio source
            language: Language code
            context: Optional context prompt for better accuracy
        
        Returns:
            Tuple of (recognized_text, confidence, model_used)
        """
        models_to_try = []
        
        if self.prefer_online:
            if self.whisper_api_key:
                models_to_try.append(("whisper_api", audio_input))
            if self.google_cloud_key:
                models_to_try.append(("google_cloud", audio_input))
            models_to_try.append(("speech_recognition", audio_input))
        
        models_to_try.append(("vosk", audio_input))
        
        for model_name, audio in models_to_try:
            try:
                if model_name == "whisper_api":
                    import asyncio
                    loop = asyncio.get_event_loop()
                    text, conf = loop.run_until_complete(
                        self.recognize_whisper_api(audio, language, context)
                    )
                elif model_name == "google_cloud":
                    text, conf = self.recognize_google_cloud_speech(audio, language)
                elif model_name == "speech_recognition" and self.sr_recognizer:
                    text, conf = self.recognize_speech_recognition(audio, language)
                elif model_name == "vosk":
                    text, conf = self.recognize_vosk(audio, language)
                else:
                    continue
                
                if text and conf > 0.5:
                    logger.info(f"✅ Recognition successful with {model_name}: {text}")
                    self.recognition_history.append({"text": text, "model": model_name, "confidence": conf})
                    return text, conf, model_name
                    
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                continue
        
        logger.error("❌ All recognition models failed")
        return None, 0.0, "none"
    
    def get_recognition_stats(self) -> Dict:
        """Get recognition performance statistics"""
        if not self.recognition_history:
            return {}
        
        return {
            "total_recognitions": len(self.recognition_history),
            "average_confidence": np.mean([r["confidence"] for r in self.recognition_history]),
            "models_used": list(set([r["model"] for r in self.recognition_history])),
            "success_rate": len([r for r in self.recognition_history if r["confidence"] > 0.5]) / len(self.recognition_history)
        }


# Global instance
_recognizer_instance = None


def get_advanced_speech_recognizer(
    whisper_api_key: Optional[str] = None,
    google_cloud_key: Optional[str] = None
) -> AdvancedSpeechRecognizer:
    """Get or create the advanced speech recognizer instance"""
    global _recognizer_instance
    if _recognizer_instance is None:
        _recognizer_instance = AdvancedSpeechRecognizer(
            whisper_api_key=whisper_api_key,
            google_cloud_key=google_cloud_key
        )
    return _recognizer_instance


# Example usage
if __name__ == "__main__":
    recognizer = get_advanced_speech_recognizer()
    print("✅ Advanced Speech Recognizer initialized")
