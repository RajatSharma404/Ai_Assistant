"""
AI Assistant - Modular architecture for intelligent assistance

This package contains specialized modules organized by functionality:

Packages:
- ai: Conversational AI, LLM providers, memory systems
- voice: Speech recognition, TTS, wake word detection
- integrations: External service integrations (Google, email, web)
- automation: System automation and app discovery
- interfaces: Web interfaces and websocket handlers
- core: Core system functionality and utilities

Utility modules:
- file_ops: File operations and management
- document_ocr: Document OCR processing
- web_scraping: Web scraping utilities
- multimodal: Multimodal processing
- multilingual: Multi-language support
- music: Music and media control
"""

__version__ = "4.0.0"
__author__ = "AI Assistant"

# Import main packages
from . import ai
from . import voice
from . import integrations
from . import automation
from . import interfaces
from . import core

# Import utility modules for backward compatibility
try:
    from .file_ops import *
except ImportError:
    pass

try:
    from .document_ocr import *
except ImportError:
    pass

try:
    from .web_scraping import *
except ImportError:
    pass

try:
    from .multimodal import *
except ImportError:
    pass

try:
    from .multilingual import *
except ImportError:
    pass

try:
    from .music import *
except ImportError:
    pass