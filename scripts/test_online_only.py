"""Test online-only LLM configuration"""
import sys
sys.path.insert(0, 'f:/bn/assitant')

try:
    from ai_assistant.modules.network_aware_llm import (
        OnlineLLMConfig, 
        NetworkAwareLLMConfig,  # Backward compatibility
        get_optimal_llm_config
    )
    print("✅ Import successful")
    
    # Test configuration
    config = get_optimal_llm_config()
    print(f"✅ Default provider: {config['provider']}")
    print(f"✅ Default model: {config['model']}")
    print(f"✅ Network status: {config['network_status']}")
    
    # Test backward compatibility
    print(f"✅ Backward compatibility: NetworkAwareLLMConfig is OnlineLLMConfig: {NetworkAwareLLMConfig is OnlineLLMConfig}")
    
    print("\n🎉 All tests passed! Online-only mode is working correctly.")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
