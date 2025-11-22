
"""
App Discovery Module
Handles scanning, indexing, and smart launching of applications on Windows.
"""

import os
import json
import subprocess
import logging
from typing import Dict, List, Optional
import winreg

logger = logging.getLogger(__name__)

APP_DB_FILE = 'apps.json'

def list_installed_apps() -> List[str]:
    """Returns a list of all indexed application names."""
    try:
        if os.path.exists(APP_DB_FILE):
            with open(APP_DB_FILE, 'r') as f:
                apps = json.load(f)
            return list(apps.keys())
    except Exception:
        pass
    return []

def refresh_app_database() -> str:
    """Force rescan of installed applications."""
    return discover_applications()

def discover_applications() -> str:
    """Scans Windows Start Menu for .lnk shortcuts and saves them to apps.json."""
    logger.info("Scanning for applications...")
    apps = {}
    
    # Common Start Menu paths
    paths = [
        os.path.join(os.environ.get('APPDATA', ''), r'Microsoft\Windows\Start Menu\Programs'),
        os.path.join(os.environ.get('PROGRAMDATA', ''), r'Microsoft\Windows\Start Menu\Programs')
    ]

    count = 0
    for path in paths:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith('.lnk'):
                        # Clean name: "Google Chrome.lnk" -> "google chrome"
                        name = file[:-4].lower()
                        full_path = os.path.join(root, file)
                        apps[name] = full_path
                        count += 1
    
    try:
        # Save to project root or current directory
        # Try to save in the same directory as main.py if possible, otherwise current cwd
        db_path = APP_DB_FILE
        
        with open(db_path, 'w') as f:
            json.dump(apps, f, indent=4)
            
        return f"Scan complete. Found {len(apps)} applications."
    except Exception as e:
        logger.error(f"Error saving app database: {e}")
        return f"Error saving app list: {e}"

def smart_open_application(app_name: str) -> str:
    """
    Smartly opens an application.
    1. Checks exact match in indexed DB.
    2. Checks fuzzy match in indexed DB.
    3. Tries direct 'start' command.
    """
    app_name = app_name.lower().strip()
    
    # 1. Load DB
    apps = {}
    if os.path.exists(APP_DB_FILE):
        try:
            with open(APP_DB_FILE, 'r') as f:
                apps = json.load(f)
        except:
            pass
    
    # If DB is empty, try to scan first
    if not apps:
        discover_applications()
        if os.path.exists(APP_DB_FILE):
            try:
                with open(APP_DB_FILE, 'r') as f:
                    apps = json.load(f)
            except:
                pass

    # 2. Search in DB
    target_path = None
    
    # Exact match
    if app_name in apps:
        target_path = apps[app_name]
    
    # Fuzzy match (contains)
    if not target_path:
        for name, path in apps.items():
            if app_name in name:
                target_path = path
                break
    
    # 3. Launch
    if target_path:
        try:
            # Use os.startfile for Windows shortcuts
            os.startfile(target_path)
            return f"Opening {app_name}..."
        except Exception as e:
            return f"Found {app_name} but failed to open: {e}"
    
    # 4. Fallback to system command
    try:
        subprocess.Popen(f"start {app_name}", shell=True)
        return f"Attempting to open {app_name} via system command..."
    except Exception as e:
        return f"Could not find or open {app_name}. Error: {e}"
