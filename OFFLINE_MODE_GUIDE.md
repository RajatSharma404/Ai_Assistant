# Offline Mode Setup Guide

Your YourDaddy Assistant now supports **offline operation**! This means you can continue using the assistant even without internet connectivity.

## Table of Contents
1. [Overview](#overview)
2. [Offline Providers](#offline-providers)
3. [Setup Instructions](#setup-instructions)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Troubleshooting](#troubleshooting)

---

## Overview

Offline mode allows your assistant to:
- ✅ Answer general knowledge questions
- ✅ Process text and perform local computations
- ✅ Execute local automation tasks
- ✅ Handle voice commands (with offline models)
- ✅ Manage files and system operations
- ❌ Cannot access: Weather, News, Web Search, Cloud Services

The assistant automatically detects internet connectivity and switches modes accordingly. You can also manually force offline mode via environment variables.

---

## Offline Providers

### 1. **Ollama** (Recommended - Best Quality)
Fast local inference engine with high-quality models.

#### Models Available:
- `llama2` - 7B parameter model (general purpose)
- `mistral` - 7B model (fast and efficient)
- `neural-chat` - Optimized for chat
- `dolphin-mixtral` - Advanced reasoning
- `orca-mini` - Lightweight

#### Installation:
```bash
# Download and install Ollama
# Windows/Mac: https://ollama.ai
# Linux:
curl https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# In another terminal, pull a model
ollama pull llama2           # ~4GB
ollama pull mistral          # ~4GB
ollama pull neural-chat      # ~3.5GB
```

#### Verify:
```bash
curl http://localhost:11434/api/tags
```

---

### 2. **Hugging Face Transformers** (PyTorch)
Pre-trained models for various NLP tasks.

#### Installation:
```bash
pip install transformers torch
```

#### Available Models:
- `distilbert-base-uncased` - Lightweight
- `bert-base-uncased` - Standard
- `gpt2` - Text generation
- `t5-base` - Sequence-to-sequence

---

### 3. **Simple Offline Mode** (Fallback)
Rule-based responses for basic queries. Always available as fallback.

---

## Setup Instructions

### Step 1: Install Ollama (Recommended)

**Windows:**
1. Download: https://ollama.ai/download/windows
2. Install the executable
3. Start Ollama from Start menu or cmd: `ollama serve`

**macOS:**
```bash
brew install ollama
ollama serve
```

**Linux:**
```bash
curl https://ollama.ai/install.sh | sh
ollama serve
```

### Step 2: Download a Model

In a new terminal:
```bash
# Download Llama 2 (recommended, ~4GB)
ollama pull llama2

# Or try Mistral (smaller, ~4GB)
ollama pull mistral

# Or Neural Chat (optimized for chat, ~3.5GB)
ollama pull neural-chat
```

### Step 3: Configure Your Assistant

Create/update `.env` or `backend.env`:

```env
# Option A: Force offline mode
FORCE_OFFLINE_MODE=true

# Option B: Let it auto-detect
# (No setting needed - will auto-switch when offline)

# Optional: Specify Ollama settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2
```

### Step 4: Test Offline Mode

```bash
# Test with offline mode forced
FORCE_OFFLINE_MODE=true python app.py

# Or start normally - it will auto-detect
python app.py
```

---

## Configuration

### Environment Variables

```env
# Force offline mode
FORCE_OFFLINE_MODE=true

# Ollama configuration
OLLAMA_HOST=http://localhost:11434    # Default
OLLAMA_MODEL=llama2                   # Model to use

# Cache settings
OFFLINE_CACHE_DIR=./offline_cache     # Cache directory
OFFLINE_CACHE_TTL=24                  # Cache TTL in hours

# Connectivity check settings
CONNECTIVITY_CHECK_INTERVAL=30        # Check every 30 seconds
ENABLE_AUTO_DETECTION=true            # Auto-detect mode changes
```

### Python Configuration

```python
from modules.offline_mode import get_offline_manager
from modules.llm_provider import UnifiedChatInterface

# Get offline manager
offline_mgr = get_offline_manager()

# Check status
print(offline_mgr.get_status())
# Output:
# {
#   "is_online": False,
#   "is_offline_mode": True,
#   "should_use_offline": True,
#   "mode": "offline"
# }

# Create chat with fallback support
chat = UnifiedChatInterface(use_fallback=True)

# Check if offline
if chat.is_offline():
    print("Running in offline mode")
```

---

## Usage

### Web Interface
```bash
# Start web backend (auto-detects offline mode)
python backend.py

# Or force offline
FORCE_OFFLINE_MODE=true python backend.py

# Access at: http://localhost:5000
```

### GUI Application
```bash
# Start GUI with offline support
python yourdaddy_app.py

# Or force offline
FORCE_OFFLINE_MODE=true python yourdaddy_app.py
```

### Command Line
```bash
# CLI mode with offline support
python app.py cli

# Force offline mode
FORCE_OFFLINE_MODE=true python app.py cli
```

### Python API
```python
from modules.llm_provider import UnifiedChatInterface
from modules.offline_mode import get_offline_manager

# Initialize with offline fallback
chat = UnifiedChatInterface(use_fallback=True)

# Add system message
chat.add_system_message("You are a helpful assistant.")

# Chat
response = chat.chat("What is Python?", stream=False)
print(response)

# Stream response
for chunk in chat.chat("Tell me a story", stream=True):
    print(chunk, end="", flush=True)

# Check offline status
offline_mgr = get_offline_manager()
print(f"Status: {offline_mgr.get_status()}")

# Cache data for offline use
offline_mgr.cache_response(
    key="weather_london",
    data={"temp": 15, "condition": "cloudy"},
    ttl_hours=24
)

# Retrieve cached data
cached = offline_mgr.get_cached_response("weather_london")
```

---

## Troubleshooting

### Ollama Not Connecting
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve

# If port is different, configure
export OLLAMA_HOST=http://localhost:11435
ollama serve --addr localhost:11435
```

### Model Not Found
```bash
# List available models
ollama list

# Pull the model
ollama pull mistral

# Or download directly
ollama pull https://registry.ollama.ai/library/llama2
```

### Slow Responses
- Use smaller model: `ollama pull orca-mini` (~2.7GB)
- Increase timeout in code:
  ```python
  chat.chat("Query", max_timeout=600)  # 10 minutes
  ```
- Add GPU support (faster generation)

### Memory Issues
- Switch to smaller model
- Reduce context size
- Close other applications

### Connectivity Detection Failing
```python
from modules.offline_mode import get_offline_manager

# Check status
mgr = get_offline_manager()
status = mgr.get_status()
print(status)

# Force offline mode
mgr.set_offline_mode(True)

# Check again
print(mgr.is_connected())  # Should be False
```

### Cache Cleanup
```python
from modules.offline_mode import get_offline_manager

mgr = get_offline_manager()

# View cache info
print(mgr.get_cache_info())

# Clear all cache
mgr.clear_cache()

# Clear cache older than 7 days
mgr.clear_cache(older_than_hours=168)
```

---

## Performance Optimization

### Offline Mode Tuning

```python
from modules.offline_llm_provider import OfflineLLMManager

# Get manager
offline_llm = OfflineLLMManager()

# Generate with options
response = offline_llm.generate_response(
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,      # Lower = more deterministic
    max_tokens=512,       # Limit response length
    top_p=0.9            # Nucleus sampling
)
```

### System Resources
- **Fast Setup**: Use `mistral` or `neural-chat` (~3-4GB)
- **Quality Setup**: Use `llama2` (~4GB, better reasoning)
- **Lightweight**: Use `orca-mini` (~2.7GB, lower quality)

### Network Impact
- No API calls when offline
- Responses generated locally
- ~0.5-2s per token (depends on hardware)

---

## Advanced Features

### Auto-Detect Mode Changes
```python
from modules.offline_mode import get_offline_manager

mgr = get_offline_manager()

# Register callback for mode changes
def on_mode_change(is_online: bool):
    if is_online:
        print("🟢 Back online - switching to cloud AI")
    else:
        print("🔴 Offline - using local AI")

mgr.add_mode_change_callback(on_mode_change)
```

### Response Caching
```python
from modules.offline_mode import get_offline_manager

mgr = get_offline_manager()

# Cache frequently asked questions
mgr.cache_response(
    key="what_is_python",
    data={"answer": "Python is a programming language..."},
    ttl_hours=168  # 1 week
)

# Later, retrieve if offline
cached = mgr.get_cached_response("what_is_python")
```

### Multiple Models
```python
from modules.offline_llm_provider import OllamaProvider

# Use specific model
llama = OllamaProvider(model="llama2")
response = llama.generate_response(
    messages=[{"role": "user", "content": "Hello"}]
)

# Switch models
mistral = OllamaProvider(model="mistral")
response = mistral.generate_response(...)
```

---

## Summary

| Feature | Online | Offline |
|---------|--------|---------|
| AI Chat | ✅ Cloud (GPT/Gemini) | ✅ Local (Ollama/Transformers) |
| Speed | ~1-5s | ~0.5-2s per token |
| Quality | Excellent | Good (depends on model) |
| Privacy | Cloud | Local (100% private) |
| Cost | API charges | Free (compute only) |
| Features | All | Local only |
| Internet Required | Yes | No |
| Works Offline | No | Yes |

---

## Next Steps

1. **Install Ollama**: https://ollama.ai
2. **Pull a model**: `ollama pull llama2`
3. **Start your assistant**: `python app.py`
4. **Go offline**: Disconnect internet - it will auto-switch!

For issues or questions, check the logs in `logs/` directory.
