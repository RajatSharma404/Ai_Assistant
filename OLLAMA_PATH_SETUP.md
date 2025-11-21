# Ollama PATH Setup for Windows Git Bash

## ✅ Your Ollama Installation

Found at: `C:\Users\hp\AppData\Local\Programs\Ollama`
Version: 0.5.7

---

## 🔧 Setup Options

### Option 1: Permanent Setup (Recommended - One Time Only)

**Step 1:** Open Git Bash and run this command:

```bash
echo 'export PATH="/c/Users/hp/AppData/Local/Programs/Ollama:$PATH"' >> ~/.bashrc
```

**Step 2:** Close and reopen Git Bash (or run: `source ~/.bashrc`)

**Step 3:** Verify it works:
```bash
ollama --version
ollama pull mistral
```

✅ Done! Ollama will work from anywhere in all future terminal sessions.

---

### Option 2: Current Session Only

If you just want it to work right now:

```bash
export PATH="/c/Users/hp/AppData/Local/Programs/Ollama:$PATH"
```

Then use ollama commands:
```bash
ollama pull mistral
ollama serve
```

(This expires when you close the terminal)

---

### Option 3: Create an Alias

Add this to your `~/.bashrc`:

```bash
echo 'alias ollama="/c/Users/hp/AppData/Local/Programs/Ollama/ollama"' >> ~/.bashrc
```

Then restart Git Bash and use:
```bash
ollama pull mistral
ollama serve
```

---

## 📝 Verify Setup

After setting up, verify in a **new** Git Bash terminal:

```bash
ollama --version
```

Should show: `ollama version is 0.5.7`

---

## 🎯 Next Steps (After Setting Path)

```bash
# Start Ollama service in background
ollama serve &

# In another terminal, download a model
ollama pull mistral

# Or download Llama 2
ollama pull llama2

# Test your assistant
python test_offline_mode.py

# Run your assistant
python app.py
```

---

## 🐛 Troubleshooting

**Still says "command not found"?**
- Make sure to close ALL Git Bash windows and reopen them
- Check that ~/.bashrc was updated: `cat ~/.bashrc | grep Ollama`

**Want to verify PATH is set?**
```bash
echo $PATH | grep Ollama
```

**Need to reset?**
```bash
# Remove from bashrc
sed -i '/Ollama/d' ~/.bashrc

# Then add again (follow Option 1 above)
```

---

## ✨ After This Works

You'll be able to:

```bash
# Download models
ollama pull mistral
ollama pull llama2
ollama pull neural-chat

# Start service
ollama serve

# Use your assistant offline
python app.py
```

All from anywhere in your terminal! 🚀
