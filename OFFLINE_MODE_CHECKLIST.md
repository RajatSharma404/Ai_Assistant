# Offline Mode Setup Checklist

## ✅ What You Need To Know

Your YourDaddy Assistant can now work **completely offline** without any internet connection!

---

## 🚀 Fastest Setup (5 Minutes)

### Choice 1: Use Ollama (Recommended - Best AI Quality)

**Step 1: Install Ollama**
- [ ] Download from https://ollama.ai/download
- [ ] Install (just click next)
- [ ] Launch Ollama (it starts as a service)

**Step 2: Download a Model**
```bash
ollama pull llama2
```
- [ ] Wait for download (~4GB, takes 5-10 minutes)
- [ ] Done!

**Step 3: Run Your Assistant**
```bash
python app.py
```
- [ ] Should just work! No configuration needed.

---

### Choice 2: Use Force Offline (No Installation)

```bash
FORCE_OFFLINE_MODE=true python app.py
```

- ✅ Works immediately
- ✅ No internet needed
- ⚠️ Limited AI capability (simple responses only)

---

## 📋 What Works Offline

| Feature | Status |
|---------|--------|
| 💬 AI Chat (with Ollama) | ✅ Yes |
| 🎤 Voice Commands | ✅ Yes |
| 📁 File Operations | ✅ Yes |
| 🔧 System Automation | ✅ Yes |
| 📝 Text Processing | ✅ Yes |
| 🌡️ Weather/News | ❌ No (needs internet) |
| 🔍 Web Search | ❌ No (needs internet) |
| 🌐 Cloud Services | ❌ No (needs internet) |

---

## 🔧 Configuration (Optional)

Create a `.env` file in your project root:

```env
# Force offline mode (ignore online APIs)
FORCE_OFFLINE_MODE=true

# Or let it auto-detect (recommended)
# FORCE_OFFLINE_MODE=false
```

---

## 🧪 Test It

```bash
# Run the test suite
python test_offline_mode.py
```

You should see:
```
✓ PASS: Offline Mode Detection
✓ PASS: Connectivity Check
✓ PASS: Offline LLM Providers
✓ PASS: LLM Factory
✓ PASS: Unified Chat Interface
✓ PASS: Response Caching
✓ PASS: Forced Offline Mode
```

---

## 🎯 Using Different Modes

### Automatic (Recommended)
```bash
python app.py
# Automatically uses offline when internet is unavailable
```

### Always Online (Uses Cloud AI)
```bash
# Make sure you have API keys set
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
python app.py
```

### Always Offline (Uses Local AI)
```bash
export FORCE_OFFLINE_MODE=true
python app.py
```

---

## 🚨 Troubleshooting

### "Ollama not connecting"
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If that fails, start Ollama
ollama serve
```

### "Model not found"
```bash
# List what you have
ollama list

# Download what you need
ollama pull mistral
```

### "Responses are slow"
1. First response takes 30 seconds (model loads) - this is normal
2. After that, responses are fast (~0.5-2 seconds per response)
3. Try a smaller model: `ollama pull orca-mini`

### "Out of memory"
1. Use a smaller model: `ollama pull orca-mini` (2.7GB)
2. Close other applications
3. Reduce maximum tokens in responses

---

## 📚 Documentation

For more details, see:

- **Quick Start**: [QUICK_START_OFFLINE.md](QUICK_START_OFFLINE.md)
- **Complete Guide**: [OFFLINE_MODE_GUIDE.md](OFFLINE_MODE_GUIDE.md)
- **Implementation Details**: [OFFLINE_MODE_IMPLEMENTATION.md](OFFLINE_MODE_IMPLEMENTATION.md)

---

## 🎮 Quick Examples

### Python Code
```python
from modules.llm_provider import UnifiedChatInterface

# This works both online and offline
chat = UnifiedChatInterface(use_fallback=True)

# Add a system message
chat.add_system_message("You are a helpful assistant")

# Chat
response = chat.chat("What is Python?")
print(response)

# Check if offline
if chat.is_offline():
    print("Running in offline mode")
```

### Web Interface
```bash
python backend.py
# Access at http://localhost:5000
# Works offline or online automatically
```

### CLI
```bash
python app.py cli
# Command line interface that works offline
```

---

## 🎯 Summary

| Aspect | Details |
|--------|---------|
| **Setup Time** | 5 minutes (with Ollama) |
| **Installation Required** | Ollama only (optional) |
| **Internet Needed** | No (completely offline) |
| **Privacy** | 100% - everything stays local |
| **Cost** | Free (no API charges) |
| **AI Quality** | Good (depends on model) |
| **Can Auto-Detect Mode** | Yes ✅ |

---

## ✨ Key Features

🟢 **Automatic Detection**
- Detects when internet goes down
- Automatically switches to offline mode
- Switches back when internet returns

🟢 **Response Caching**
- Caches responses for offline use
- Configurable expiration time
- Smart cache management

🟢 **Multiple Providers**
- Ollama (best quality)
- HuggingFace Transformers
- Simple rule-based fallback
- Automatic fallback chain

🟢 **Backward Compatible**
- Existing code works unchanged
- All online features still available
- Optional offline support

---

## 🎓 What to Do Next

1. **Install Ollama** → https://ollama.ai
2. **Pull a model** → `ollama pull llama2`
3. **Run tests** → `python test_offline_mode.py`
4. **Start assistant** → `python app.py`
5. **Go offline** → Disconnect internet, watch it auto-switch!

---

## 💡 Pro Tips

✅ Use `mistral` for speed (faster than llama2)  
✅ Use `llama2` for better quality  
✅ Use `orca-mini` if low on RAM  
✅ First response loads model (~30 seconds) - wait for it  
✅ Set `FORCE_OFFLINE_MODE=true` to skip API checks  
✅ Check `logs/` directory for detailed logs  

---

## 📞 Need Help?

1. Check logs: `logs/` directory
2. Run tests: `python test_offline_mode.py`
3. Read guides above
4. Check Ollama status: `curl http://localhost:11434/api/tags`

**Everything should just work! Enjoy your offline assistant! 🎉**
