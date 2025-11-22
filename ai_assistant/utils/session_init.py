"""
Session Initialization Module
============================

This module initializes a new logging session every time the assistant starts.
Import this module at the beginning of your main application files.
"""

from utils.logging_config import SessionManager, get_logger
from utils.session_activity_logger import session_activity_logger
import sys
import os
from datetime import datetime

# Start new session immediately when this module is imported
session_id = SessionManager.start_new_session()

# Create a startup logger
startup_logger = get_logger('session_startup', log_category='system')

# Log session initialization
startup_logger.info(f"🎯 NEW ASSISTANT SESSION INITIALIZED")
startup_logger.info(f"Session ID: {session_id}")
startup_logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
startup_logger.info(f"Platform: {sys.platform}")
startup_logger.info(f"Python Version: {sys.version.split()[0]}")
startup_logger.info(f"Working Directory: {os.getcwd()}")

# Log startup activity
session_activity_logger.log_system_command(
    'session_initialization',
    command_type='system_startup',
    success=True
)

# Export session information for other modules
CURRENT_SESSION_ID = session_id
SESSION_START_TIME = SessionManager.get_session_start_time()

def get_session_info():
    """Get current session information"""
    return {
        'session_id': CURRENT_SESSION_ID,
        'start_time': SESSION_START_TIME.isoformat() if SESSION_START_TIME else None,
        'uptime_seconds': (datetime.now() - SESSION_START_TIME).total_seconds() if SESSION_START_TIME else 0
    }

def log_module_initialization(module_name: str, details: dict = None):
    """Log when a module is initialized"""
    startup_logger.info(f"📦 Module initialized: {module_name}")
    
    session_activity_logger.log_system_command(
        f'module_init_{module_name}',
        command_type='module_initialization',
        success=True
    )
    
    if details:
        startup_logger.info(f"Module details: {details}")

# Print session start info to console
print(f"")
print(f"🚀 YourDaddy Assistant - New Session Started")
print(f"📅 Session ID: {session_id}")
print(f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📁 Logs will be saved in: logs/*/[module_name]_{session_id}.log")
print(f"📊 Activity tracking: logs/activities/session_summary_{session_id}.json")
print(f"")