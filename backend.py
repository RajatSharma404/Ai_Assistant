#!/usr/bin/env python3
"""
YourDaddy Assistant - Unified Backend Server
A consolidated Flask backend with configurable features and multiple operation modes.
Supports simple, enhanced, and full-featured modes based on configuration.
"""

import os
import sys
import time
import json
import re
import secrets
import logging
import threading
import webbrowser
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# Flask and related imports
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, jwt_required, create_access_token, 
    get_jwt_identity, verify_jwt_in_request
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
    sys.stderr.reconfigure(encoding='utf-8') if hasattr(sys.stderr, 'reconfigure') else None

# Load environment variables
load_dotenv()

class BackendConfig:
    """Configuration class for backend features"""
    
    def __init__(self):
        self.mode = os.getenv('BACKEND_MODE', 'enhanced')  # simple, enhanced, full
        self.port = int(os.getenv('PORT', '5000'))
        self.host = os.getenv('HOST', '0.0.0.0')
        self.debug = os.getenv('DEBUG', 'true').lower() == 'true'
        
        # Feature flags
        self.enable_auth = os.getenv('ENABLE_AUTH', 'true').lower() == 'true'
        self.enable_rate_limiting = os.getenv('ENABLE_RATE_LIMITING', 'true').lower() == 'true'
        self.enable_multimodal = os.getenv('ENABLE_MULTIMODAL', 'true').lower() == 'true'
        self.enable_multilingual = os.getenv('ENABLE_MULTILINGUAL', 'true').lower() == 'true'
        self.enable_voice = os.getenv('ENABLE_VOICE', 'true').lower() == 'true'
        self.enable_automation = os.getenv('ENABLE_AUTOMATION', 'true').lower() == 'true'
        self.enable_logging = os.getenv('ENABLE_LOGGING', 'true').lower() == 'true'
        
        # Security settings
        self.jwt_secret = os.getenv('JWT_SECRET_KEY', secrets.token_hex(16))
        self.admin_password = os.getenv('ADMIN_PASSWORD', 'changeme123')
        self.allowed_origins = os.getenv(
            'ALLOWED_ORIGINS', 
            'http://localhost:3000,http://localhost:5000,http://127.0.0.1:3000,http://127.0.0.1:5000'
        ).split(',')

config = BackendConfig()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = config.jwt_secret
app.config['JWT_SECRET_KEY'] = config.jwt_secret
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)

# Initialize extensions based on config
if config.enable_auth:
    jwt = JWTManager(app)

if config.enable_rate_limiting:
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per hour", "50 per minute"],
        storage_uri="memory://"
    )

# CORS Configuration
CORS(app, resources={
    r"/api/*": {
        "origins": config.allowed_origins,
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Initialize SocketIO
socketio = SocketIO(
    app, 
    cors_allowed_origins=config.allowed_origins,
    async_mode='threading',
    engineio_logger=False
)

# Setup logging
if config.enable_logging:
    try:
        from utils.logging_config import get_logger, get_api_logger
        from utils.user_data_logger import log_query, log_reply, log_action, log_module_usage
        logger = get_logger('unified_backend', log_category='backend')
        api_logger = get_api_logger('api_requests')
        LOGGING_AVAILABLE = True
    except ImportError:
        print("⚠️ Logging utilities not available")
        LOGGING_AVAILABLE = False
        logger = None
        api_logger = None
else:
    LOGGING_AVAILABLE = False
    logger = None
    api_logger = None

def log_info(message):
    if logger:
        logger.info(message)
    else:
        print(f"ℹ️ {message}")

def log_error(message):
    if logger:
        logger.error(message)
    else:
        print(f"❌ {message}")

log_info("="*80)
log_info("YourDaddy Assistant - Unified Backend Starting")
log_info(f"Mode: {config.mode.upper()}")
log_info("="*80)

# Configuration validation
if config.mode in ['enhanced', 'full']:
    try:
        from config_validator import validate_config
        log_info("🔍 Validating configuration...")
        config_validator = validate_config(exit_on_failure=True)
    except Exception as e:
        log_error(f"Configuration validation failed: {e}")
        if config.mode == 'full':
            sys.exit(1)

# Module imports based on configuration
AUTOMATION_AVAILABLE = False
CONVERSATIONAL_AI_AVAILABLE = False
MULTIMODAL_AVAILABLE = False
MULTILINGUAL_AVAILABLE = False

# Import automation tools
if config.enable_automation:
    try:
        from automation_tools_new import (
            write_a_note, open_application, search_google, search_youtube,
            close_application, speak, set_system_volume, get_app_path_from_name,
            setup_memory, save_to_memory, get_memory, search_memory,
            get_conversation_summary, save_knowledge, get_knowledge,
            discover_applications, smart_open_application, list_installed_apps,
            refresh_app_database, search_apps_by_name, get_app_usage_stats, get_apps_for_web,
            get_system_status, get_running_processes, cleanup_temp_files,
            get_network_info, get_upcoming_events, get_inbox_summary,
            get_spotify_status, spotify_play_pause, spotify_next_track,
            spotify_previous_track, search_and_play_spotify,
            get_weather_info, get_latest_news, get_stock_price,
            detect_taskbar_apps, can_see_taskbar
        )
        AUTOMATION_AVAILABLE = True
        log_info("✅ Automation tools loaded successfully")
    except ImportError as e:
        log_error(f"Automation tools not available: {e}")

# Import conversational AI
if config.mode in ['enhanced', 'full']:
    try:
        from modules.conversational_ai import AdvancedConversationalAI
        CONVERSATIONAL_AI_AVAILABLE = True
        log_info("✅ Conversational AI loaded successfully")
    except ImportError as e:
        log_error(f"Conversational AI not available: {e}")

# Import multimodal AI
if config.enable_multimodal and config.mode == 'full':
    try:
        from modules.multimodal import MultiModalAI
        MULTIMODAL_AVAILABLE = True
        log_info("✅ Multimodal AI loaded successfully")
    except ImportError as e:
        log_error(f"Multimodal AI not available: {e}")

# Import multilingual support
if config.enable_multilingual:
    try:
        from modules.multilingual import MultilingualSupport, Language, LanguageContext
        MULTILINGUAL_AVAILABLE = True
        log_info("✅ Multilingual support loaded successfully")
    except ImportError as e:
        log_error(f"Multilingual support not available: {e}")

# User Management
USERS_DB = {
    "admin": {
        "password_hash": generate_password_hash(config.admin_password),
        "role": "admin"
    }
}

# Input Validation Patterns
VALIDATION_PATTERNS = {
    'command': re.compile(r'^[\w\s\-.,!?@#$%()+=:;"\']+$'),
    'app_name': re.compile(r'^[\w\s\-.]+$'),
    'username': re.compile(r'^[a-zA-Z0-9_]{3,20}$'),
}

def validate_input(data, field, pattern_name):
    """Validate input data against pattern"""
    if not data or field not in data:
        return False, f"{field} is required"
    
    value = data[field]
    if not isinstance(value, str):
        return False, f"{field} must be a string"
    
    if len(value) > 1000:
        return False, f"{field} too long (max 1000 characters)"
    
    pattern = VALIDATION_PATTERNS.get(pattern_name)
    if pattern and not pattern.match(value):
        return False, f"{field} contains invalid characters"
    
    return True, None

# Global instances
conversational_ai = None
multimodal_ai = None
multilingual_support = None

def initialize_ai_components():
    """Initialize AI components based on configuration"""
    global conversational_ai, multimodal_ai, multilingual_support
    
    if CONVERSATIONAL_AI_AVAILABLE:
        try:
            # Create automation callback if available
            automation_callback = None
            if AUTOMATION_AVAILABLE:
                def automation_callback(action, param):
                    try:
                        if action == 'open_application':
                            return open_application(param)
                        elif action == 'search_google':
                            return search_google(param)
                        elif action == 'play_music':
                            return search_and_play_spotify(param)
                        elif action == 'close_application':
                            return close_application(param)
                        elif action == 'set_volume':
                            return set_system_volume(int(param))
                        else:
                            return f"Action {action} not supported"
                    except Exception as e:
                        return f"Error executing {action}: {str(e)}"
            
            conversational_ai = AdvancedConversationalAI(automation_callback=automation_callback)
            log_info("✅ Conversational AI initialized")
        except Exception as e:
            log_error(f"Failed to initialize conversational AI: {e}")
    
    if MULTIMODAL_AVAILABLE:
        try:
            multimodal_ai = MultiModalAI()
            log_info("✅ Multimodal AI initialized")
        except Exception as e:
            log_error(f"Failed to initialize multimodal AI: {e}")
    
    if MULTILINGUAL_AVAILABLE:
        try:
            multilingual_support = MultilingualSupport()
            log_info("✅ Multilingual support initialized")
        except Exception as e:
            log_error(f"Failed to initialize multilingual support: {e}")

# Initialize components
initialize_ai_components()

# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/')
def index():
    """Serve the main page"""
    try:
        return send_from_directory('templates', 'index.html')
    except:
        return jsonify({
            'status': 'success',
            'message': f'YourDaddy Assistant Backend ({config.mode.title()} Mode)',
            'features': {
                'automation': AUTOMATION_AVAILABLE,
                'conversational_ai': CONVERSATIONAL_AI_AVAILABLE,
                'multimodal': MULTIMODAL_AVAILABLE,
                'multilingual': MULTILINGUAL_AVAILABLE,
                'auth': config.enable_auth,
                'rate_limiting': config.enable_rate_limiting
            }
        })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'mode': config.mode,
        'features': {
            'automation': AUTOMATION_AVAILABLE,
            'conversational_ai': CONVERSATIONAL_AI_AVAILABLE,
            'multimodal': MULTIMODAL_AVAILABLE,
            'multilingual': MULTILINGUAL_AVAILABLE
        }
    })

@app.route('/api/login', methods=['POST'])
def login():
    """User login endpoint"""
    if not config.enable_auth:
        return jsonify({'error': 'Authentication disabled'}), 400
    
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        user = USERS_DB.get(username)
        if user and check_password_hash(user['password_hash'], password):
            access_token = create_access_token(identity=username)
            return jsonify({
                'access_token': access_token,
                'user': {'username': username, 'role': user['role']}
            })
        
        return jsonify({'error': 'Invalid credentials'}), 401
    
    except Exception as e:
        log_error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/command', methods=['POST'])
def process_command():
    """Process user commands"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input
        is_valid, error_msg = validate_input(data, 'command', 'command')
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        command = data['command'].strip()
        if not command:
            return jsonify({'error': 'Empty command'}), 400
        
        # Log the query
        if LOGGING_AVAILABLE:
            log_query(command, context={'endpoint': 'api/command'})
        
        # Process through conversational AI if available
        if CONVERSATIONAL_AI_AVAILABLE and conversational_ai:
            response = conversational_ai.process_message(command)
        elif AUTOMATION_AVAILABLE:
            # Fallback to direct automation for simple mode
            response = process_simple_command(command)
        else:
            response = f"Command received: {command} (Processing not available)"
        
        # Log the response
        if LOGGING_AVAILABLE:
            log_reply(response, context={'endpoint': 'api/command'})
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'mode': config.mode
        })
    
    except Exception as e:
        log_error(f"Command processing error: {e}")
        return jsonify({'error': 'Failed to process command'}), 500

def process_simple_command(command):
    """Process commands in simple mode without conversational AI"""
    command_lower = command.lower()
    
    if 'hello' in command_lower or 'hi' in command_lower:
        return "👋 Hello! I'm your assistant. How can I help you?"
    elif 'time' in command_lower:
        return f"Current time: {datetime.now().strftime('%I:%M %p')}"
    elif 'date' in command_lower:
        return f"Today's date: {datetime.now().strftime('%B %d, %Y')}"
    elif command_lower.startswith('open '):
        app_name = command[5:].strip()
        if AUTOMATION_AVAILABLE:
            return open_application(app_name)
        else:
            return f"Would open: {app_name}"
    elif command_lower.startswith('search ') or command_lower.startswith('google '):
        query = command.split(' ', 1)[1] if ' ' in command else ""
        if AUTOMATION_AVAILABLE:
            return search_google(query)
        else:
            return f"Would search for: {query}"
    else:
        return f"I received your command: {command}. In simple mode, try 'open [app]' or 'search [query]'"

@app.route('/api/apps')
def get_apps():
    """Get available applications"""
    try:
        if AUTOMATION_AVAILABLE:
            apps = get_apps_for_web()
            return jsonify({'apps': apps})
        else:
            return jsonify({'apps': [], 'message': 'Automation not available'})
    except Exception as e:
        log_error(f"Apps retrieval error: {e}")
        return jsonify({'error': 'Failed to get apps'}), 500

@app.route('/api/system/status')
def get_system_status_api():
    """Get system status"""
    try:
        if AUTOMATION_AVAILABLE:
            status = get_system_status()
            return jsonify({'status': status})
        else:
            return jsonify({
                'status': {
                    'cpu': 'N/A',
                    'memory': 'N/A',
                    'disk': 'N/A',
                    'message': 'System monitoring not available'
                }
            })
    except Exception as e:
        log_error(f"System status error: {e}")
        return jsonify({'error': 'Failed to get system status'}), 500

@app.route('/api/voice/recognition', methods=['POST'])
def voice_recognition():
    """Handle voice recognition requests"""
    if not config.enable_voice:
        return jsonify({'error': 'Voice recognition disabled'}), 400
    
    try:
        # This would handle voice data from frontend
        # For now, return a placeholder
        return jsonify({
            'text': 'Voice recognition would process audio here',
            'confidence': 0.95
        })
    except Exception as e:
        log_error(f"Voice recognition error: {e}")
        return jsonify({'error': 'Voice recognition failed'}), 500

@app.route('/api/multimodal/analyze', methods=['POST'])
def multimodal_analyze():
    """Analyze images or screenshots"""
    if not config.enable_multimodal or not MULTIMODAL_AVAILABLE:
        return jsonify({'error': 'Multimodal analysis not available'}), 400
    
    try:
        # Handle image analysis
        return jsonify({
            'analysis': 'Multimodal analysis would process image here',
            'confidence': 0.90
        })
    except Exception as e:
        log_error(f"Multimodal analysis error: {e}")
        return jsonify({'error': 'Analysis failed'}), 500

# =============================================================================
# WEBSOCKET EVENTS
# =============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    log_info(f"Client connected: {request.sid}")
    emit('status', {
        'message': 'Connected to YourDaddy Assistant',
        'features': {
            'automation': AUTOMATION_AVAILABLE,
            'conversational_ai': CONVERSATIONAL_AI_AVAILABLE,
            'multimodal': MULTIMODAL_AVAILABLE,
            'multilingual': MULTILINGUAL_AVAILABLE
        }
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    log_info(f"Client disconnected: {request.sid}")

@socketio.on('chat_message')
def handle_chat_message(data):
    """Handle chat messages through WebSocket"""
    try:
        message = data.get('message', '').strip()
        if not message:
            emit('error', {'message': 'Empty message'})
            return
        
        # Process the message
        if CONVERSATIONAL_AI_AVAILABLE and conversational_ai:
            response = conversational_ai.process_message(message)
        else:
            response = process_simple_command(message)
        
        # Send response back
        emit('chat_response', {
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        log_error(f"Chat message error: {e}")
        emit('error', {'message': 'Failed to process message'})

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'error': 'Rate limit exceeded'}), 429

# =============================================================================
# SERVER STARTUP
# =============================================================================

def open_browser():
    """Open browser after server starts"""
    time.sleep(2)  # Wait for server to start
    url = f"http://localhost:{config.port}"
    log_info(f"Opening browser to {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        log_error(f"Failed to open browser: {e}")

def main():
    """Main server startup function"""
    log_info("="*60)
    log_info("🚀 YourDaddy Assistant Unified Backend")
    log_info(f"📊 Mode: {config.mode.upper()}")
    log_info(f"🌐 Server: http://{config.host}:{config.port}")
    log_info(f"🔧 Features: Auth={config.enable_auth}, Rate Limiting={config.enable_rate_limiting}")
    log_info(f"🎯 Modules: AI={CONVERSATIONAL_AI_AVAILABLE}, Auto={AUTOMATION_AVAILABLE}")
    log_info("="*60)
    
    # Open browser if in development mode
    if config.debug and not os.getenv('NO_BROWSER'):
        threading.Thread(target=open_browser, daemon=True).start()
    
    # Start the server
    try:
        if config.enable_rate_limiting or True:  # Use socketio for WebSocket support
            socketio.run(
                app,
                host=config.host,
                port=config.port,
                debug=config.debug,
                use_reloader=False
            )
        else:
            app.run(
                host=config.host,
                port=config.port,
                debug=config.debug,
                use_reloader=False,
                threaded=True
            )
    except KeyboardInterrupt:
        log_info("\n🛑 Server stopped by user")
    except Exception as e:
        log_error(f"Server error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()