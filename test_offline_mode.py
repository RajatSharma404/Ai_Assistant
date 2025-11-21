#!/usr/bin/env python3
"""
Test script for offline mode functionality
Tests connectivity detection, offline LLM providers, and mode switching
"""

import sys
import os
import json
import time
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "modules"))

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_offline_mode_detection():
    """Test offline mode detection."""
    print_section("TEST 1: Offline Mode Detection")
    
    try:
        from modules.offline_mode import get_offline_manager
        
        mgr = get_offline_manager()
        status = mgr.get_status()
        
        print(f"✓ Offline manager initialized")
        print(f"  Is online: {status['is_online']}")
        print(f"  Is offline mode: {status['is_offline_mode']}")
        print(f"  Should use offline: {status['should_use_offline']}")
        print(f"  Mode: {status['mode']}")
        print(f"  Last check: {status['last_check']}")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def test_connectivity_check():
    """Test connectivity checking."""
    print_section("TEST 2: Connectivity Check")
    
    try:
        from modules.offline_mode import OfflineModeManager
        
        mgr = OfflineModeManager(enable_auto_detection=False)
        
        print("Checking connectivity...")
        is_connected = mgr.is_connected()
        
        if is_connected:
            print(f"✓ Internet connectivity detected")
        else:
            print(f"✓ No internet connectivity (running offline mode)")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def test_offline_llm_providers():
    """Test offline LLM providers."""
    print_section("TEST 3: Offline LLM Providers")
    
    results = {}
    
    # Test Ollama
    print("Testing Ollama provider...")
    try:
        from modules.offline_llm_provider import OllamaProvider
        
        ollama = OllamaProvider()
        available = ollama.is_available()
        
        if available:
            print(f"✓ Ollama provider available")
            results['ollama'] = True
        else:
            print(f"⚠ Ollama not running")
            print(f"  Install from: https://ollama.ai")
            print(f"  Then run: ollama serve")
            print(f"  Then pull a model: ollama pull llama2")
            results['ollama'] = False
    except Exception as e:
        print(f"⚠ Ollama check failed: {e}")
        results['ollama'] = False
    
    # Test Transformers
    print("\nTesting Transformers provider...")
    try:
        from modules.offline_llm_provider import TransformersProvider
        
        transformers = TransformersProvider()
        available = transformers.is_available()
        
        if available:
            print(f"✓ Transformers provider available")
            results['transformers'] = True
        else:
            print(f"⚠ Transformers not initialized")
            print(f"  Install with: pip install transformers torch")
            results['transformers'] = False
    except Exception as e:
        print(f"⚠ Transformers check failed: {e}")
        results['transformers'] = False
    
    # Test Simple Provider
    print("\nTesting Simple Offline provider...")
    try:
        from modules.offline_llm_provider import SimpleOfflineProvider
        
        simple = SimpleOfflineProvider()
        available = simple.is_available()
        
        if available:
            print(f"✓ Simple offline provider available (fallback)")
            results['simple'] = True
        else:
            print(f"✗ Simple provider not available")
            results['simple'] = False
    except Exception as e:
        print(f"✗ Simple provider check failed: {e}")
        results['simple'] = False
    
    # Test Manager
    print("\nTesting Offline LLM Manager...")
    try:
        from modules.offline_llm_provider import OfflineLLMManager
        
        manager = OfflineLLMManager()
        info = manager.get_provider_info()
        
        print(f"✓ Manager initialized")
        print(f"  Current provider: {info['current']}")
        print(f"  Available providers: {info['available_providers']}")
        print(f"  Total: {info['count']}")
        results['manager'] = True
    except Exception as e:
        print(f"✗ Manager failed: {e}")
        results['manager'] = False
    
    return all(results.values())

def test_llm_factory():
    """Test LLM factory with offline support."""
    print_section("TEST 4: LLM Factory")
    
    try:
        from modules.llm_provider import LLMFactory
        
        # Test provider detection
        print("Testing provider detection...")
        provider, model = LLMFactory.detect_provider()
        print(f"✓ Detected provider: {provider}")
        print(f"  Model: {model}")
        
        # Test factory creation with fallback
        print("\nTesting factory creation with fallback...")
        llm = LLMFactory.create_with_fallback()
        print(f"✓ LLM provider created")
        print(f"  Type: {type(llm).__name__}")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def test_unified_chat_interface():
    """Test unified chat interface with offline support."""
    print_section("TEST 5: Unified Chat Interface")
    
    try:
        from modules.llm_provider import UnifiedChatInterface
        
        # Create with fallback
        print("Creating chat interface with fallback...")
        chat = UnifiedChatInterface(use_fallback=True)
        
        print(f"✓ Chat interface created")
        print(f"  Provider: {chat.provider_name}")
        print(f"  Model: {chat.model}")
        print(f"  Offline mode: {chat.is_offline()}")
        
        # Test simple message
        print("\nTesting simple response...")
        chat.add_system_message("You are a helpful assistant.")
        response = chat.chat("What is Python?", stream=False)
        
        print(f"✓ Got response: {response[:100]}...")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_caching():
    """Test offline response caching."""
    print_section("TEST 6: Response Caching")
    
    try:
        from modules.offline_mode import get_offline_manager
        
        mgr = get_offline_manager()
        
        # Cache test data
        print("Caching test response...")
        test_data = {
            "question": "What is offline mode?",
            "answer": "Offline mode allows the assistant to work without internet",
            "timestamp": time.time()
        }
        
        mgr.cache_response("test_response", test_data, ttl_hours=1)
        print(f"✓ Response cached")
        
        # Retrieve cached data
        print("\nRetrieving cached response...")
        cached = mgr.get_cached_response("test_response")
        
        if cached:
            print(f"✓ Cache retrieved")
            print(f"  Question: {cached['question']}")
            print(f"  Answer: {cached['answer']}")
        else:
            print(f"✗ Cache not found")
            return False
        
        # Get cache info
        print("\nGetting cache info...")
        info = mgr.get_cache_info()
        print(f"✓ Cache info:")
        print(f"  Total items: {info.get('total_items', 0)}")
        print(f"  Total size: {info.get('total_size_bytes', 0)} bytes")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def test_offline_forced_mode():
    """Test forced offline mode."""
    print_section("TEST 7: Forced Offline Mode")
    
    try:
        from modules.offline_mode import get_offline_manager
        
        mgr = get_offline_manager()
        
        # Force offline
        print("Forcing offline mode...")
        mgr.set_offline_mode(True)
        
        status = mgr.get_status()
        if status['mode'] == 'offline':
            print(f"✓ Offline mode forced successfully")
            print(f"  Mode: {status['mode']}")
        else:
            print(f"✗ Failed to force offline mode")
            return False
        
        # Disable forced offline
        print("\nDisabling forced offline mode...")
        mgr.set_offline_mode(False)
        
        status = mgr.get_status()
        expected = 'offline' if not status['is_online'] else 'online'
        
        if status['mode'] == expected:
            print(f"✓ Forced offline mode disabled")
            print(f"  Mode: {status['mode']}")
        else:
            print(f"⚠ Mode might not match connectivity")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("  YourDaddy Assistant - Offline Mode Tests")
    print("="*60)
    
    tests = [
        ("Offline Mode Detection", test_offline_mode_detection),
        ("Connectivity Check", test_connectivity_check),
        ("Offline LLM Providers", test_offline_llm_providers),
        ("LLM Factory", test_llm_factory),
        ("Unified Chat Interface", test_unified_chat_interface),
        ("Response Caching", test_caching),
        ("Forced Offline Mode", test_offline_forced_mode),
    ]
    
    results = {}
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
            failed += 1
    
    # Summary
    print_section("TEST SUMMARY")
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")
    
    if failed == 0:
        print("✓ All tests passed! Offline mode is working correctly.")
    else:
        print(f"⚠ {failed} test(s) failed. See above for details.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
