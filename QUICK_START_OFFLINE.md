# Quick Start: Offline Mode (2 Minutes)

## TL;DR - Get Your Assistant Working Offline

### Option 1: Use Ollama (Recommended) - 3 Steps

```bash
# 1. Download & install Ollama
# Windows/Mac: https://ollama.ai/download
# Linux: curl https://ollama.ai/install.sh | sh

# 2. Start Ollama in one terminal
ollama serve

# 3. In another terminal, download a model
ollama pull llama2          # ~4GB, great quality
# OR
ollama pull mistral         # ~4GB, faster
```

**Done!** Your assistant now works offline. Just run:
```bash
python app.py              # GUI mode
# OR
python backend.py          # Web interface at localhost:5000
```

---

### Option 2: Force Offline Without Installing Anything

```bash
FORCE_OFFLINE_MODE=true python app.py
```

This uses the **simple offline mode** - works but with limited AI capability.

---

## What Works Offline

✅ AI conversations (if using Ollama/Transformers)  
✅ Text processing & analysis  
✅ File operations  
✅ Voice commands (with offline models)  
✅ Local automation  
✅ 100% private - no cloud calls  

❌ Weather, news, web search (need internet)

---

## Test It

```bash
python test_offline_mode.py
```

This checks all offline providers and shows what's available.

---

## Troubleshooting

**Ollama not connecting?**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags
# If fails, start it: ollama serve
```

**Slow responses?**
- Use smaller model: `ollama pull orca-mini`
- Or wait 10-30 seconds for first response (model loads)

**Want to go back online?**
```bash
# Just stop using FORCE_OFFLINE_MODE
python app.py
```

The assistant **auto-detects** internet and switches modes!

---

## Full Setup Guide

For detailed configuration, see: [OFFLINE_MODE_GUIDE.md](OFFLINE_MODE_GUIDE.md)
