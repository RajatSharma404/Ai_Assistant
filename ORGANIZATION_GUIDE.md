# YourDaddy Assistant - Project Organization

## 🎯 Unified Project Structure

This project has been **reorganized and consolidated** to eliminate duplicates and improve maintainability. Here's what's new:

## 📁 Main Files

### Entry Points
- **`app.py`** - **NEW**: Unified main entry point with multiple modes
- **`start.bat`** / **`start.sh`** - **NEW**: Cross-platform launch scripts
- `yourdaddy_app.py` - Original GUI application (kept as component)
- `launch_assistant.py` - Original launcher with system checks (kept as component)

### Core Components
- **`backend.py`** - **NEW**: Unified backend server (replaces 4 separate backends)
- **`setup.py`** - **NEW**: Unified setup with feature selection
- **`debug.py`** - **NEW**: Comprehensive debugging tool
- **`test_chat.py`** - **NEW**: Consolidated chat testing suite
- **`test_integration.py`** - **NEW**: Complete integration tests

### Configuration
- **`requirements.txt`** - **UPDATED**: Single, organized dependencies file
- **`backend.env.example`** - **NEW**: Backend configuration template
- `.env.example` - Environment variables template

## 🚀 Quick Start

### Method 1: New Unified Launcher
```bash
# Windows
start.bat app

# Linux/macOS
./start.sh app
```

### Method 2: Direct Python
```bash
# GUI mode (default)
python app.py

# Web interface
python app.py web

# Command line
python app.py cli

# Run setup
python app.py setup

# Run tests
python app.py test

# System check
python app.py check
```

### Method 3: Original Methods (still work)
```bash
python yourdaddy_app.py      # Original GUI
python backend.py            # Backend server
python launch_assistant.py   # Original launcher
```

## 🧹 What Was Consolidated

### ❌ Removed Duplicate Files
- `test_web_chat.py`, `test_stub_chat.py`, `test_improved_chat.py`, etc. → `test_chat.py`
- `modern_web_backend.py`, `simple_backend.py`, `simple_chat_server.py` → `backend.py`
- `launch.bat`, `start_app.bat`, `start_web_ui.bat`, etc. → `start.bat`
- `setup_multimodal.py`, `setup_multilingual.py` → `setup.py`
- `debug_chat.py`, `debug_video.py` → `debug.py`
- `requirements_original.txt` → merged into `requirements.txt`

### ✅ Benefits
- **50% fewer files** to maintain
- **Single source of truth** for each functionality
- **Consistent interface** across all components
- **Better organization** and documentation
- **Easier deployment** and distribution

## 🎮 Usage Examples

### Development Workflow
```bash
# 1. Setup (first time)
python app.py setup

# 2. Run tests
python app.py test

# 3. Start development server
python app.py web --debug

# 4. Debug issues
python debug.py
```

### Production Deployment
```bash
# Start web server
python app.py web --host 0.0.0.0 --port 8000

# Or use launcher
./start.sh web
```

## 📊 Features by Mode

| Mode | Features | Use Case |
|------|----------|----------|
| **GUI** | Full desktop interface | Desktop users |
| **Web** | Browser-based interface | Remote access |
| **CLI** | Terminal interface | Developers, servers |
| **Setup** | Feature installation | Initial configuration |
| **Test** | Comprehensive testing | Quality assurance |
| **Debug** | Diagnostic tools | Troubleshooting |

## 🔧 Configuration

### Backend Modes
The unified backend supports multiple modes:
- **Simple**: Basic functionality only
- **Enhanced**: Full AI with automation
- **Full**: All features including multimodal AI

Configure in `.env`:
```env
BACKEND_MODE=enhanced
ENABLE_MULTIMODAL=true
ENABLE_MULTILINGUAL=true
```

## 🆘 Support

### Troubleshooting
```bash
# Check system compatibility
python app.py check

# Run diagnostics
python debug.py

# View logs
ls logs/
```

### Getting Help
```bash
python app.py help
```

## 🔄 Migration from Old Structure

If you were using the old files:

| Old Command | New Command |
|-------------|-------------|
| `python modern_web_backend.py` | `python app.py web` |
| `python test_improved_chat_full.py` | `python app.py test` or `python test_chat.py` |
| `python setup_multimodal.py` | `python app.py setup` |
| `python debug_chat.py` | `python debug.py chat` |
| Various `.bat` files | `start.bat [mode]` |

---

**Note**: Original files like `yourdaddy_app.py` and `launch_assistant.py` are still available as components, but the new unified structure is recommended for new deployments.