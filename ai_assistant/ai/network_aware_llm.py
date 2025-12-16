"""
Online LLM Configuration - Re-export Module
Re-export from canonical location.
The actual implementation is in ai_assistant.modules.network_aware_llm
"""

# Re-export everything from the canonical module
from ai_assistant.modules.network_aware_llm import (
    OnlineLLMConfig,
    NetworkAwareLLMConfig,  # Backward compatibility alias
    network_config,
    get_optimal_llm_config,
    force_online_mode,
)

__all__ = [
    'OnlineLLMConfig',
    'NetworkAwareLLMConfig',
    'network_config',
    'get_optimal_llm_config',
    'force_online_mode',
]