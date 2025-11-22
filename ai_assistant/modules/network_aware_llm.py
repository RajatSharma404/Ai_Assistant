"""
Smart Network-Aware LLM Configuration
Automatically uses GPT/Gemini when online, falls back to local models when offline.
"""

import os
import json
import requests
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class NetworkAwareLLMConfig:
    """Smart configuration that adapts based on network connectivity."""
    
    def __init__(self):
        self.last_check = None
            key_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "api_keys.json")
            if os.path.exists(key_file):
                with open(key_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load api_keys.json: {e}")
        return {}

    def check_internet_connectivity(self) -> bool:
        """Check if internet connection is available."""
        # Cache the result to avoid excessive checks
        now = datetime.now()
        if (self.last_check and 
            self.network_status is not None and 
            now - self.last_check < self.check_interval):
            return self.network_status
        
        try:
            # Test multiple endpoints for reliability
            test_urls = [
                "https://api.openai.com/v1/models",
                "https://generativelanguage.googleapis.com/v1beta/models",
                "https://httpbin.org/status/200",
                "https://google.com"
            ]
            
            for url in test_urls:
                try:
                    response = requests.get(url, timeout=3)
                    if response.status_code == 200:
                        self.network_status = True
                        self.last_check = now
                        logger.info("✅ Internet connectivity confirmed")
                        return True
                except:
                    continue
            
            # All tests failed
            self.network_status = False
            self.last_check = now
            logger.warning("❌ No internet connectivity detected")
            return False
            
        except Exception as e:
            logger.error(f"Network check failed: {e}")
            self.network_status = False
            self.last_check = now
            return False
    
    def get_optimal_provider(self) -> Tuple[str, str]:
        """
        Get the optimal provider based on network status and availability.
        Prioritizes your powerful local models when they're available!
        
        Returns:
            Tuple of (provider_name, model_name)
        """
        # First, always check if your powerful local models are available
        for provider, model, available in self.local_providers:
            if available and self._test_ollama_model(model):
                logger.info(f"🏠 Using your powerful local model: {model}")
                return (provider, model)
        
        # Check network connectivity for online fallback
        is_online = self.check_internet_connectivity()
        
        if is_online:
            # Try online providers as backup
            for provider, model, api_key in self.online_providers:
                if api_key:  # Only use if API key is available
                    try:
                        # Quick test of the provider
                        if self._test_provider(provider, api_key):
                            logger.info(f"🌐 Using online provider: {provider} ({model})")
                            return (provider, model)
                    except Exception as e:
                        logger.warning(f"Provider {provider} test failed: {e}")
                        continue
            
            logger.warning("Online providers failed, no local models available")
        else:
            logger.info("No internet connection and no local models available")
        
        # Final fallback
        logger.error("No providers available, using basic offline mode")
        return ("offline", "offline")
    
    def _test_provider(self, provider: str, api_key: str) -> bool:
        """Quick test of online provider availability."""
        try:
            if provider == "openai":
                response = requests.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=5
                )
                return response.status_code == 200
            
            elif provider == "gemini":
                response = requests.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
                    timeout=5
                )
                return response.status_code == 200
            
            return False
            
        except Exception:
            return False
    
    def _test_ollama_model(self, model: str) -> bool:
        """Test if Ollama model is available."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                return any(model in name for name in model_names)
            return False
        except Exception:
            return False
    
    def get_provider_config(self) -> Dict:
        """Get complete provider configuration."""
        provider, model = self.get_optimal_provider()
        
        config = {
            "provider": provider,
            "model": model,
            "fallback_enabled": True,
            "network_aware": True,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "network_status": self.network_status
        }
        
        # Add provider-specific settings
        if provider == "openai":
            config.update({
                "temperature": 0.7,
                "max_tokens": 4000,
                "top_p": 0.95
            })
        elif provider == "gemini":
            config.update({
                "temperature": 0.7,
                "max_tokens": 4000,
                "safety_settings": "BLOCK_MEDIUM_AND_ABOVE"
            })
        elif provider == "ollama":
            config.update({
                "temperature": 0.7,
                "max_tokens": 4000,
                "context_length": 8192 if "27b" in model else 4096
            })
        
        return config

# Global instance
network_config = NetworkAwareLLMConfig()

def get_optimal_llm_config():
    """Get the optimal LLM configuration based on current network status."""
    return network_config.get_provider_config()

def force_local_mode():
    """Force use of local models regardless of network status."""
    network_config.network_status = False
    return network_config.get_optimal_provider()

def force_online_mode():
    """Force use of online models (will fail if no internet)."""
    network_config.network_status = True
    return network_config.get_optimal_provider()