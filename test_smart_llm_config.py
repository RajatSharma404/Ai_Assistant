#!/usr/bin/env python3
"""
Test script to verify the smart network-aware LLM configuration
"""

import os
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_network_config():
    """Test the network-aware LLM configuration"""
    print("🧪 Testing Smart Network-Aware LLM Configuration")
    print("=" * 50)
    
    try:
        from modules.network_aware_llm import get_optimal_llm_config, network_config
        
        # Test configuration
        config = get_optimal_llm_config()
        print(f"📋 Configuration:")
        print(f"   Provider: {config['provider']}")
        print(f"   Model: {config['model']}")
        print(f"   Network Status: {'Online 🌐' if config['network_status'] else 'Offline 🏠'}")
        print(f"   Fallback Enabled: {config['fallback_enabled']}")
        
        # Test provider creation
        from modules.llm_provider import LLMFactory
        provider_name, model_name = LLMFactory.detect_provider()
        print(f"\n🔧 Detected Provider: {provider_name} ({model_name})")
        
        # Test unified interface
        from modules.llm_provider import UnifiedChatInterface
        chat = UnifiedChatInterface(provider=provider_name, model=model_name, use_fallback=True)
        
        print(f"\n💬 Testing chat with {provider_name}...")
        test_message = "Hello! Can you tell me what AI model you are?"
        
        start_time = time.time()
        response = chat.chat(test_message)
        response_time = time.time() - start_time
        
        print(f"✅ Response received in {response_time:.2f}s:")
        print(f"   {response[:200]}{'...' if len(response) > 200 else ''}")
        
        # Test network status checking
        print(f"\n🔍 Network Connectivity Tests:")
        print(f"   Last check: {config.get('last_check', 'Never')}")
        
        # Force refresh network check
        network_config.last_check = None
        is_online = network_config.check_internet_connectivity()
        print(f"   Fresh check result: {'Online 🌐' if is_online else 'Offline 🏠'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_network_config()
    if success:
        print(f"\n✅ Smart LLM configuration is working correctly!")
        print(f"🚀 Ready to use GPT/Gemini online or your local models offline")
    else:
        print(f"\n❌ Configuration test failed")
    
    sys.exit(0 if success else 1)