#!/usr/bin/env python3
"""
Test LLM provider connections to ensure they work properly
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def test_gemini_connection():
    """Test Gemini API connection"""
    print("🔍 Testing Gemini API connection...")
    
    try:
        from modules.llm_provider import GeminiProvider
        
        # Check if API key exists
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY not found in environment")
            return False
        
        print(f"✅ API key found: {api_key[:10]}...")
        
        # Initialize provider
        provider = GeminiProvider()
        print("✅ Gemini provider initialized")
        
        # Test simple message
        test_messages = [
            {"role": "user", "content": "Hello! Please respond with exactly: 'Gemini connection successful'"}
        ]
        
        response = provider.generate_response(test_messages)
        print(f"📨 Response: {response}")
        
        if "connection successful" in response.lower():
            print("✅ Gemini connection test PASSED")
            return True
        else:
            print("⚠️ Gemini responded but with unexpected content")
            return True  # Still working, just different response
            
    except Exception as e:
        print(f"❌ Gemini connection test FAILED: {e}")
        return False

def test_openai_connection():
    """Test OpenAI API connection"""
    print("\n🔍 Testing OpenAI API connection...")
    
    try:
        from modules.llm_provider import OpenAIProvider
        
        # Check if API key exists
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not found in environment")
            return False
        
        print(f"✅ API key found: {api_key[:10]}...")
        
        # Initialize provider
        provider = OpenAIProvider()
        print("✅ OpenAI provider initialized")
        
        # Test simple message
        test_messages = [
            {"role": "user", "content": "Hello! Please respond with exactly: 'OpenAI connection successful'"}
        ]
        
        response = provider.generate_response(test_messages)
        print(f"📨 Response: {response}")
        
        if "connection successful" in response.lower():
            print("✅ OpenAI connection test PASSED")
            return True
        else:
            print("⚠️ OpenAI responded but with unexpected content")
            return True  # Still working, just different response
            
    except Exception as e:
        print(f"❌ OpenAI connection test FAILED: {e}")
        return False

def test_local_llm_connection():
    """Test local LLM connection"""
    print("\n🔍 Testing Local LLM (Ollama) connection...")
    
    try:
        from modules.llm_provider import LocalLLMProvider
        
        # Initialize provider
        provider = LocalLLMProvider()
        print("✅ Local LLM provider initialized")
        
        # Test simple message
        test_messages = [
            {"role": "user", "content": "Hello! Please respond with exactly: 'Local LLM connection successful'"}
        ]
        
        response = provider.generate_response(test_messages)
        print(f"📨 Response: {response}")
        
        if "connection successful" in response.lower():
            print("✅ Local LLM connection test PASSED")
            return True
        else:
            print("⚠️ Local LLM responded but with unexpected content")
            return True  # Still working, just different response
            
    except Exception as e:
        print(f"❌ Local LLM connection test FAILED: {e}")
        return False

def test_unified_chat_interface():
    """Test unified chat interface auto-detection"""
    print("\n🔍 Testing UnifiedChatInterface auto-detection...")
    
    try:
        from modules.llm_provider import UnifiedChatInterface
        
        # Initialize with auto-detection
        chat = UnifiedChatInterface()
        print(f"✅ Auto-detected provider: {chat.provider_name}")
        print(f"✅ Using model: {chat.model}")
        
        # Test chat
        response = chat.chat("Hello! What's 2+2?")
        print(f"📨 Response: {response}")
        
        print("✅ UnifiedChatInterface test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ UnifiedChatInterface test FAILED: {e}")
        return False

def main():
    """Run all LLM connection tests"""
    print("🚀 LLM Provider Connection Tests\n")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    results = {
        "gemini": test_gemini_connection(),
        "openai": test_openai_connection(), 
        "local": test_local_llm_connection(),
        "unified": test_unified_chat_interface()
    }
    
    print(f"\n📊 Test Results:")
    for provider, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {provider.capitalize()}: {status}")
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    print(f"\nSummary: {passed}/{total} tests passed")
    
    if passed > 0:
        print("\n🎉 At least one LLM provider is working!")
        print("💡 The chat system should be able to connect to AI models.")
    else:
        print("\n😞 No LLM providers are working.")
        print("💡 Check your API keys and internet connection.")

if __name__ == "__main__":
    main()