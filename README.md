# AI Assistant

An intelligent AI assistant with voice recognition, automation capabilities, and integration with various services.

## 🏗️ Project Structure

```
ai-assistant/
├── main.py                    # Main entry point
├── pyproject.toml            # Project configuration
├── requirements.txt          # Dependencies
├── start.bat / start.sh      # Platform-specific startup scripts
│
├── src/ai_assistant/         # Main package
│   ├── __init__.py
│   ├── ai/                   # AI & machine learning modules
│   │   ├── conversational_ai.py
│   │   ├── llm_provider.py
│   │   ├── memory.py
│   │   └── ...
│   ├── voice/                # Voice processing modules
│   │   ├── advanced_speech_recognizer.py
│   │   ├── neural_voice_engine.py
│   │   ├── wake_word_detector.py
│   │   └── ...
│   ├── integrations/         # External service integrations
│   │   ├── google_calendar.py
│   │   ├── email_handler.py
│   │   ├── web_search_integration.py
│   │   └── ...
│   ├── automation/           # System automation
│   │   ├── smart_automation.py
│   │   ├── app_discovery.py
│   │   └── ...
│   ├── interfaces/           # Web & network interfaces
│   │   ├── modern_interfaces.py
│   │   ├── websocket_handlers.py
│   │   └── ...
│   ├── core/                 # Core system functionality
│   │   ├── core.py
│   │   ├── system.py
│   │   └── ...
│   └── apps/                 # Application entry points
│       ├── app.py            # CLI interface
│       ├── backend.py        # Backend server
│       ├── modern_web_backend.py  # Modern web interface
│       └── ...
│
├── config/                   # Configuration files
│   ├── multimodal_config.json
│   ├── user_settings.json
│   └── ...
│
├── scripts/                  # Utility scripts
│   ├── setup/               # Setup and installation scripts
│   ├── analysis/            # Analysis and diagnostic scripts
│   └── debug/               # Debug and troubleshooting scripts
│
├── tests/                   # Test files
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
│
├── docs/                    # Documentation
│   ├── README.md
│   ├── CONTRIBUTING.md
│   └── ...
│
├── data/                    # Data files
├── logs/                    # Log files
├── static/                  # Static web assets
├── templates/               # Web templates
└── user_data/              # User-specific data
```

## 🚀 Quick Start

### Using the main entry point:
```bash
# Start web interface (default)
python main.py

# Start with specific interface
python main.py --interface web --port 8080
python main.py --interface cli
python main.py --interface desktop

# Enable verbose logging
python main.py --verbose
```

### Using individual applications:
```bash
# Web backend
python src/ai_assistant/apps/modern_web_backend.py

# CLI interface
python src/ai_assistant/apps/app.py

# Desktop GUI
python src/ai_assistant/apps/yourdaddy_app.py
```

## 📦 Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run setup scripts in `scripts/setup/`
4. Configure settings in `config/`
5. Start the assistant: `python main.py`

## 🔧 Configuration

Configuration files are located in the `config/` directory:
- `multimodal_config.json` - Multimodal processing settings
- `user_settings.json` - User preferences
- `backend.env.example` - Environment variables template

## 📚 Package Organization

- **ai/**: Conversational AI, LLM providers, memory systems
- **voice/**: Speech recognition, TTS, wake word detection
- **integrations/**: External service integrations (Google, email, web)
- **automation/**: System automation and app discovery
- **interfaces/**: Web interfaces and websocket handlers
- **core/**: Core system functionality and utilities
- **apps/**: Application entry points for different interfaces

## 🧪 Testing

Run tests using:
```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/

# All tests
python -m pytest tests/
```

## 📖 Documentation

Detailed documentation is available in the `docs/` directory.

## 🤝 Contributing

See `docs/CONTRIBUTING.md` for contribution guidelines.

## 📄 License

See `docs/LICENSE.txt` for license information.