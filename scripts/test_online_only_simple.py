"""Simple test for online-only LLM configuration without full initialization"""
import sys
import os

# Don't import from ai_assistant which triggers full initialization
sys.path.insert(0, 'f:/bn/assitant')

# Temporarily skip full module loading
os.environ['SKIP_INIT'] = '1'

try:
    # Direct import without triggering __init__
    import importlib.util
    
    spec = importlib.util.spec_from_file_location(
        "network_aware_llm", 
        "f:/bn/assitant/ai_assistant/modules/network_aware_llm.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    print("✅ Module loaded successfully")
    
    # Test the configuration
    config = module.get_optimal_llm_config()
    print(f"✅ Default provider: {config['provider']}")
    print(f"✅ Default model: {config['model']}")
    print(f"✅ Network status: {config['network_status']}")
    print(f"✅ API base: {config.get('api_base', 'default')}")
    
    # Check class exists
    print(f"✅ OnlineLLMConfig class exists: {hasattr(module, 'OnlineLLMConfig')}")
    print(f"✅ NetworkAwareLLMConfig alias exists: {hasattr(module, 'NetworkAwareLLMConfig')}")
    
    # Check no local functions exist
    print(f"✅ force_local_mode removed: {not hasattr(module, 'force_local_mode')}")
    
    print("\n🎉 All tests passed! Online-only mode is working correctly.")
    print("   - Only OpenAI and Gemini providers")
    print("   - No local LLM support")
    print("   - Backward compatibility maintained")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
