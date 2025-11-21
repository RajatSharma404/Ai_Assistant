# Ollama Custom Storage Path Setup

## 🎯 Change Where Models Are Stored

By default, Ollama stores models in your user's AppData folder. You can change this to any location.

---

## 📍 Current Default Location

```
C:\Users\hp\AppData\Local\Ollama\models
```

This is the default and takes up space on your C: drive.

---

## 🔧 Change Storage Path

### Option 1: Use a Different Drive (Recommended if low on C: drive)

**Example:** Store models on D: drive

#### Step 1: Create a folder for models
```bash
mkdir /d/Ollama/models
```

Or use Windows:
```
D:\Ollama\models
```

#### Step 2: Set the environment variable

**For Current Session Only:**
```bash
export OLLAMA_MODELS="/d/Ollama/models"
ollama serve
```

**For Permanent Setup:**

Add to `~/.bashrc`:
```bash
echo 'export OLLAMA_MODELS="/d/Ollama/models"' >> ~/.bashrc
```

Then reload:
```bash
source ~/.bashrc
```

#### Step 3: Verify
```bash
echo $OLLAMA_MODELS
```

Should show: `/d/Ollama/models`

#### Step 4: Download models
```bash
ollama pull mistral
```

Models will now be stored at: `D:\Ollama\models`

---

### Option 2: Use a Project Folder

Store models locally in your assistant project:

#### Step 1: Create a models folder
```bash
mkdir -p ./ollama_models
```

#### Step 2: Set the path
```bash
export OLLAMA_MODELS="$PWD/ollama_models"
ollama serve
```

Or add to `~/.bashrc`:
```bash
echo 'export OLLAMA_MODELS="$PWD/ollama_models"' >> ~/.bashrc
```

#### Step 3: Download models
```bash
ollama pull mistral
ollama pull llama2
```

Models stored in: `F:\bn\assitant\ollama_models\`

---

## 🔄 If You Already Downloaded Models

If you already have models in the default location and want to move them:

### Step 1: Find current models
```bash
ls "C:\Users\hp\AppData\Local\Ollama\models"
```

Or in Git Bash:
```bash
ls /c/Users/hp/AppData/Local/Ollama/models
```

### Step 2: Create new location
```bash
mkdir /d/Ollama/models
```

### Step 3: Copy models (if you want to keep them)
```bash
cp -r "/c/Users/hp/AppData/Local/Ollama/models"/* "/d/Ollama/models"
```

### Step 4: Set environment variable
```bash
export OLLAMA_MODELS="/d/Ollama/models"
```

### Step 5: Verify Ollama sees them
```bash
ollama list
```

Should show your existing models.

---

## 📋 Complete Setup Example

### Set Custom Path and Download Model

```bash
# 1. Create custom folder on D: drive
mkdir /d/OllamaModels

# 2. Set environment variable for current session
export OLLAMA_MODELS="/d/OllamaModels"

# 3. Start Ollama service
ollama serve

# 4. In another terminal, download a model
export OLLAMA_MODELS="/d/OllamaModels"
ollama pull mistral

# 5. Verify
ollama list
```

### Make it Permanent

Add to `~/.bashrc`:
```bash
echo 'export OLLAMA_MODELS="/d/OllamaModels"' >> ~/.bashrc
source ~/.bashrc
```

---

## 💾 Storage Space Needed

| Model | Size |
|-------|------|
| mistral | ~4GB |
| llama2 | ~4GB |
| neural-chat | ~3.5GB |
| orca-mini | ~2.7GB |
| dolphin-mixtral | ~26GB |

Make sure you have enough space in your chosen location!

---

## 🎯 Recommended Setup

```bash
# Create a dedicated folder
mkdir /d/OllamaModels

# Set it permanently in ~/.bashrc
echo 'export OLLAMA_MODELS="/d/OllamaModels"' >> ~/.bashrc

# Reload
source ~/.bashrc

# Verify
echo $OLLAMA_MODELS

# Download models
ollama pull mistral
ollama pull llama2

# See what you have
ollama list
```

---

## ✅ Verify Storage Path

```bash
# Check where Ollama will store models
echo $OLLAMA_MODELS

# List downloaded models
ollama list

# See actual folder size
du -sh /path/to/models
```

---

## 🔍 Common Paths

| Location | Path |
|----------|------|
| C: drive (default) | `/c/Users/hp/AppData/Local/Ollama/models` |
| D: drive | `/d/OllamaModels` |
| Project folder | `./ollama_models` |
| Custom | `/e/LargeStorage/ollama` |

Choose based on where you have the most space!

---

## 📌 Important Notes

1. **Set BEFORE downloading** - Models go to the path set at download time
2. **Must be set for both serve and pull** - Use the same path in both commands
3. **Ollama must be running** - For `ollama pull` to work, `ollama serve` should be active
4. **Restart required** - After setting in bashrc, restart Git Bash for permanent change

---

## 🚀 Quick Copy-Paste for D: Drive

```bash
# One-time setup
mkdir /d/OllamaModels
export OLLAMA_MODELS="/d/OllamaModels"

# Make permanent
echo 'export OLLAMA_MODELS="/d/OllamaModels"' >> ~/.bashrc
source ~/.bashrc

# Download models
ollama serve &
ollama pull mistral
```

**Done!** Models will be stored on D: drive. 🎉
