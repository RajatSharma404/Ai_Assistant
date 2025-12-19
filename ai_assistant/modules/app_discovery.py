# Dynamic Application Discovery Module
"""
This module scans the system to discover all installed applications
and provides intelligent app launching based on voice commands.
"""

import os
import winreg
import json
import subprocess
import glob
from pathlib import Path
import time
import sqlite3
from typing import Dict, List, Tuple
from datetime import datetime

class AppDiscovery:
    def __init__(self):
        # Get the project root directory (2 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "config"
        
        # Ensure config directory exists
        config_dir.mkdir(exist_ok=True)
        
        # Set paths relative to config directory
        self.apps_cache_file = str(config_dir / "discovered_apps.json")
        self.usage_db_file = str(config_dir / "app_usage.db")
        self.apps_database = {}
        self._is_refreshing = False
        self._last_refresh_time = None
        
        # Load cache first for fast startup
        self.load_cache()
        self._init_usage_database()
        
        # Start background refresh
        self._start_background_refresh()
    
    def scan_installed_applications(self) -> Dict[str, str]:
        """
        Scan only officially registered applications:
        - Windows Registry (Apps & Features in Settings)
        - Start Menu shortcuts (All Apps in Start Menu)
        - Essential Windows Store apps (Camera, etc.)
        """
        print("🔍 Scanning Windows registered applications...")
        apps = {}
        
        # Method 1: Windows Registry - Apps shown in Settings > Apps & Features
        print("  📋 Scanning Windows Registry (Apps & Features)...")
        apps.update(self._scan_registry_programs())
        
        # Method 2: Start Menu shortcuts - All Apps in Start Menu
        print("  📂 Scanning Start Menu (All Apps)...")
        apps.update(self._scan_start_menu())
        
        # Method 3: Essential Windows Store apps
        print("  📱 Scanning essential Windows Store apps...")
        apps.update(self._scan_essential_store_apps())
        
        # DISABLED: Raw Program Files scanning (finds unregistered apps)
        # DISABLED: Manual AppData scanning (finds portable apps)
        # DISABLED: Hardcoded system utilities
        
        # Save to cache
        self.apps_database = apps
        self.save_cache()
        
        print(f"✅ Discovery complete! Found {len(apps)} registered applications.")
        return apps
    
    def _scan_registry_programs(self) -> Dict[str, str]:
        """Scan Windows Registry for installed programs"""
        apps = {}
        registry_paths = [
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"
        ]
        
        for reg_path in registry_paths:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as key:
                    for i in range(winreg.QueryInfoKey(key)[0]):
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            with winreg.OpenKey(key, subkey_name) as subkey:
                                try:
                                    display_name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                                    install_location = winreg.QueryValueEx(subkey, "InstallLocation")[0]
                                    
                                    if display_name and install_location:
                                        # Look for executable files in install location
                                        exe_files = glob.glob(os.path.join(install_location, "*.exe"))
                                        if exe_files:
                                            # Select first valid executable
                                            apps[display_name.lower()] = exe_files[0]
                                except FileNotFoundError:
                                    continue
                        except OSError:
                            continue
            except Exception as e:
                print(f"Registry scan error: {e}")
        
        return apps
    
    def _scan_essential_store_apps(self) -> Dict[str, str]:
        """Scan for essential Windows Store apps that users commonly need"""
        apps = {}
        # List of common Windows Store apps with their protocol handlers
        essential_apps = {
            'camera': 'microsoft.windows.camera:',
            'mail': 'outlookmail:',
            'calendar': 'outlookcal:',
            'photos': 'ms-photos:',
            'calculator': 'calculator:',
            'maps': 'bingmaps:',
            'store': 'ms-windows-store:',
            'settings': 'ms-settings:',
        }
        
        for app_name, protocol in essential_apps.items():
            # Use the protocol handler as the "path" - Windows will handle it correctly
            apps[app_name] = protocol
        
        return apps
    
    def _scan_start_menu(self) -> Dict[str, str]:
        """Scan Start Menu shortcuts"""
        apps = {}
        start_menu_paths = [
            os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs"),
            os.path.expandvars(r"%PROGRAMDATA%\Microsoft\Windows\Start Menu\Programs")
        ]
        
        for start_path in start_menu_paths:
            if os.path.exists(start_path):
                for root, dirs, files in os.walk(start_path):
                    for file in files:
                        if file.endswith('.lnk'):
                            shortcut_path = os.path.join(root, file)
                            app_name = file[:-4]  # Remove .lnk extension
                            target = self._resolve_shortcut(shortcut_path)
                            # Accept .exe files OR empty targets (UWP apps use the shortcut itself)
                            if target and target.lower().endswith('.exe'):
                                # Check if it's a PWA (browser proxy executables)
                                pwa_proxies = ['chrome_proxy.exe', 'msedge_proxy.exe', 'brave_proxy.exe', 
                                              'opera_proxy.exe', 'vivaldi_proxy.exe', 'arc_proxy.exe']
                                is_pwa = any(proxy in target.lower() for proxy in pwa_proxies)
                                
                                if is_pwa:
                                    # Store the .lnk path for PWAs to preserve app-id arguments
                                    apps[app_name.lower()] = shortcut_path
                                else:
                                    apps[app_name.lower()] = target
                            elif not target or not target.strip():
                                # UWP/Store apps - use the shortcut path itself
                                apps[app_name.lower()] = shortcut_path
        
        return apps
    
    def _resolve_shortcut(self, shortcut_path: str) -> str:
        """Resolve .lnk shortcut to actual target with improved methods."""
        try:
            # Method 1: Try with win32com if available
            try:
                import win32com.client
                shell = win32com.client.Dispatch("WScript.Shell")
                shortcut = shell.CreateShortCut(shortcut_path)
                target = shortcut.Targetpath
                if target and os.path.exists(target):
                    return target
            except ImportError:
                pass
            except Exception as e:
                print(f"win32com shortcut resolution failed: {e}")
            
            # Method 2: PowerShell approach
            try:
                cmd = f'powershell -Command "(New-Object -ComObject WScript.Shell).CreateShortcut(\'{shortcut_path}\').TargetPath"'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    target = result.stdout.strip()
                    if os.path.exists(target):
                        return target
            except Exception as e:
                print(f"PowerShell shortcut resolution failed: {e}")
            
            # Method 3: Python-only approach using struct (parse .lnk binary)
            try:
                with open(shortcut_path, 'rb') as f:
                    # Read .lnk file header
                    data = f.read()
                    # Look for local path in the file (simplified approach)
                    # .lnk files contain paths as null-terminated strings
                    if b'\\' in data:
                        # Find potential path strings
                        parts = data.split(b'\\x00\\x00')
                        for part in parts:
                            try:
                                path_str = part.decode('utf-8', errors='ignore')
                                # Look for executable paths
                                if '.exe' in path_str.lower() and ':\\' in path_str:
                                    # Extract the path
                                    for line in path_str.split('\\n'):
                                        if '.exe' in line.lower() and os.path.exists(line.strip()):
                                            return line.strip()
                            except:
                                continue
            except Exception as e:
                print(f"Binary shortcut resolution failed: {e}")
            
        except Exception as e:
            print(f"Error resolving shortcut {shortcut_path}: {e}")
        
        return ""
    
    def save_cache(self):
        """Save discovered apps to cache file"""
        try:
            with open(self.apps_cache_file, 'w') as f:
                json.dump(self.apps_database, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def load_cache(self):
        """Load discovered apps from cache file"""
        try:
            if os.path.exists(self.apps_cache_file):
                with open(self.apps_cache_file, 'r') as f:
                    data = json.load(f)
                    
                # Handle different cache formats
                if isinstance(data, dict):
                    if 'applications' in data:
                        # New complex format - extract app data
                        self.apps_database = {}
                        for app in data['applications']:
                            app_name = app['name'].lower().replace(' ', '_')
                            self.apps_database[app_name] = app['path']
                    else:
                        # Old simple format - use as is
                        self.apps_database = data
                else:
                    self.apps_database = {}
        except Exception as e:
            print(f"Error loading cache: {e}")
            self.apps_database = {}
    
    def _start_background_refresh(self):
        """Start background thread to refresh app list"""
        import threading
        thread = threading.Thread(target=self._background_refresh, daemon=True)
        thread.start()
    
    def _background_refresh(self):
        """Background refresh of app database"""
        try:
            self._is_refreshing = True
            print("🔄 Background app refresh started...")
            
            # Scan for apps
            new_apps = self.scan_installed_applications()
            
            # Update timestamp
            from datetime import datetime
            self._last_refresh_time = datetime.now()
            
            print(f"✅ Background refresh complete! Found {len(new_apps)} apps")
        except Exception as e:
            print(f"⚠️ Background refresh failed: {e}")
        finally:
            self._is_refreshing = False
    
    def _init_usage_database(self):
        """Initialize SQLite database for tracking app usage."""
        try:
            with sqlite3.connect(self.usage_db_file) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS app_launches (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        app_name TEXT NOT NULL,
                        app_path TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        success BOOLEAN DEFAULT 1
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_app_launches_name 
                    ON app_launches(app_name)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_app_launches_timestamp 
                    ON app_launches(timestamp DESC)
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS app_frequency (
                        app_name TEXT PRIMARY KEY,
                        launch_count INTEGER DEFAULT 0,
                        last_launched DATETIME,
                        avg_daily_launches REAL DEFAULT 0.0
                    )
                """)
                
                conn.commit()
        except Exception as e:
            print(f"Error initializing usage database: {e}")
    
    def track_app_launch(self, app_name: str, app_path: str = "", success: bool = True):
        """Track an application launch for usage statistics."""
        try:
            with sqlite3.connect(self.usage_db_file) as conn:
                # Record launch
                conn.execute("""
                    INSERT INTO app_launches (app_name, app_path, success)
                    VALUES (?, ?, ?)
                """, (app_name, app_path, success))
                
                # Update frequency table
                conn.execute("""
                    INSERT INTO app_frequency (app_name, launch_count, last_launched)
                    VALUES (?, 1, CURRENT_TIMESTAMP)
                    ON CONFLICT(app_name) DO UPDATE SET
                        launch_count = launch_count + 1,
                        last_launched = CURRENT_TIMESTAMP
                """, (app_name,))
                
                conn.commit()
        except Exception as e:
            print(f"Error tracking app launch: {e}")
    
    def get_most_used_apps(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently used applications."""
        try:
            with sqlite3.connect(self.usage_db_file) as conn:
                cursor = conn.execute("""
                    SELECT app_name, launch_count
                    FROM app_frequency
                    ORDER BY launch_count DESC
                    LIMIT ?
                """, (limit,))
                return cursor.fetchall()
        except Exception as e:
            print(f"Error getting most used apps: {e}")
            return []
    
    def get_recent_apps(self, limit: int = 10) -> List[Tuple[str, str]]:
        """Get recently launched applications."""
        try:
            with sqlite3.connect(self.usage_db_file) as conn:
                cursor = conn.execute("""
                    SELECT DISTINCT app_name, MAX(timestamp) as last_time
                    FROM app_launches
                    WHERE success = 1
                    GROUP BY app_name
                    ORDER BY last_time DESC
                    LIMIT ?
                """, (limit,))
                return cursor.fetchall()
        except Exception as e:
            print(f"Error getting recent apps: {e}")
            return []
    
    def _split_camel_case(self, text: str) -> str:
        """Split camelCase and PascalCase words with spaces."""
        import re
        # Insert space before capital letters (except at start)
        result = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
        result = result.lower()
        
        # Also try to split known compound words in lowercase
        # This handles cases like "microsoftstickynotes" -> "microsoft sticky notes"
        compound_patterns = [
            (r'microsoft', 'microsoft '),
            (r'sticky', ' sticky '),
            (r'notes', ' notes '),
            (r'google', 'google '),
            (r'chrome', ' chrome '),
            (r'spotify', ' spotify '),
            (r'adobe', 'adobe '),
            (r'reader', ' reader '),
            (r'player', ' player '),
            (r'media', ' media '),
            (r'video', ' video '),
            (r'music', ' music '),
        ]
        
        for pattern, replacement in compound_patterns:
            result = re.sub(pattern, replacement, result)
        
        # Clean up extra spaces
        result = ' '.join(result.split())
        return result
    
    def find_app(self, app_name: str) -> str:
        """Find application by name using advanced fuzzy matching with usage-based ranking."""
        app_name_lower = app_name.lower().strip()
        
        # Normalize the search query (remove special chars, handle spaces)
        normalized_query = app_name_lower.replace('.', ' ').replace('_', ' ').replace('-', ' ')
        
        # Get usage statistics for ranking boost
        most_used = {name.lower(): count for name, count in self.get_most_used_apps(100)}
        
        matches = []
        
        for db_name, db_path in self.apps_database.items():
            # Normalize database name for comparison
            # First split camelCase, then replace special chars
            normalized_db_name = self._split_camel_case(db_name)
            normalized_db_name = normalized_db_name.replace('.', ' ').replace('_', ' ').replace('-', ' ')
            
            score = self._calculate_match_score(normalized_query, normalized_db_name, most_used.get(db_name, 0))
            if score > 0:
                matches.append((score, db_name, db_path))
        
        if not matches:
            return ""
        
        # Sort by score (highest first)
        matches.sort(reverse=True, key=lambda x: x[0])
        
        # Only return if the score is good enough (minimum threshold of 30)
        best_match = matches[0]
        if best_match[0] >= 30:  # score threshold
            return best_match[2]
        else:
            return ""  # No good match found
    
    def _calculate_match_score(self, query: str, app_name: str, usage_count: int = 0) -> int:
        """Calculate match score for fuzzy search with usage-based ranking."""
        score = 0
        
        # Exact match (highest priority)
        if query == app_name:
            score += 100
        
        # Direct substring match
        if query in app_name:
            score += 50
            # Boost if match is at the start
            if app_name.startswith(query):
                score += 20
        
        # Reverse substring match
        if app_name in query:
            score += 40
        
        # Word-based matching - check if ALL query words are present
        query_words = set(query.split())
        app_words = set(app_name.split())
        
        # Check if all query words exist in app name (important for multi-word searches)
        if query_words and query_words.issubset(app_words):
            score += 80  # High score for containing all words
        
        # Some query words present (partial match)
        common_words = query_words & app_words
        if common_words and not query_words.issubset(app_words):
            score += len(common_words) * 10
        
        # Character-level fuzzy matching (Levenshtein-like)
        if score == 0:  # Only if no other matches
            similarity = self._string_similarity(query, app_name)
            if similarity > 0.7:
                score += int(similarity * 20)
        
        # Boost by usage frequency (logarithmic scale)
        if usage_count > 0:
            import math
            score += int(math.log(usage_count + 1) * 5)
        
        return score
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity (0.0 to 1.0) using simple approach."""
        if not s1 or not s2:
            return 0.0
        
        # Convert to sets of character bigrams
        def get_bigrams(s):
            return set(s[i:i+2] for i in range(len(s) - 1))
        
        bigrams1 = get_bigrams(s1)
        bigrams2 = get_bigrams(s2)
        
        if not bigrams1 or not bigrams2:
            return 0.0
        
        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)
        
        return intersection / union if union > 0 else 0.0
    
    def search_apps(self, query: str, limit: int = 10) -> List[Tuple[str, str, int]]:
        """Search for apps with scoring and ranking."""
        query_lower = query.lower().strip()
        most_used = {name.lower(): count for name, count in self.get_most_used_apps(100)}
        
        matches = []
        for db_name, db_path in self.apps_database.items():
            score = self._calculate_match_score(query_lower, db_name, most_used.get(db_name, 0))
            if score > 0:
                matches.append((db_name, db_path, score))
        
        # Sort by score and return top matches
        matches.sort(reverse=True, key=lambda x: x[2])
        return matches[:limit]
    
    def get_all_apps(self) -> Dict[str, str]:
        """Get all discovered applications"""
        return self.apps_database
    
    def get_apps_for_api(self) -> List[Dict[str, str]]:
        """Get applications formatted for API responses"""
        apps_list = []
        
        for app_name, app_path in self.apps_database.items():
            # Get usage data
            most_used = dict(self.get_most_used_apps(100))
            usage_count = most_used.get(app_name, 0)
            
            # Categorize app
            category = self._categorize_app(app_name)
            
            apps_list.append({
                "name": app_name.replace('_', ' ').title(),
                "path": app_path,
                "category": category,
                "usage": usage_count,
                "description": self._generate_description(app_name)
            })
        
        # Sort by usage count
        apps_list.sort(key=lambda x: x['usage'], reverse=True)
        return apps_list
    
    def _categorize_app(self, app_name: str) -> str:
        """Categorize application by name"""
        app_lower = app_name.lower()
        
        if any(word in app_lower for word in ['chrome', 'firefox', 'edge', 'browser']):
            return "Browser"
        elif any(word in app_lower for word in ['notepad', 'word', 'excel', 'powerpoint', 'office', 'sticky', 'onenote']):
            return "Productivity"
        elif any(word in app_lower for word in ['code', 'visual', 'studio', 'terminal', 'cmd', 'powershell']):
            return "Development"
        elif any(word in app_lower for word in ['vlc', 'media', 'music', 'video', 'spotify']):
            return "Media"
        elif any(word in app_lower for word in ['mail', 'discord', 'slack', 'teams']):
            return "Communication"
        elif any(word in app_lower for word in ['calculator', 'notepad', 'paint', 'control', 'task']):
            return "System Tools"
        else:
            return "Other"
    
    def _generate_description(self, app_name: str) -> str:
        """Generate description for application"""
        descriptions = {
            'chrome': 'Google Chrome web browser',
            'firefox': 'Mozilla Firefox web browser',
            'edge': 'Microsoft Edge web browser',
            'notepad': 'Simple text editor',
            'calculator': 'Windows calculator utility',
            'paint': 'Microsoft Paint image editor',
            'word': 'Microsoft Word document editor',
            'excel': 'Microsoft Excel spreadsheet application',
            'powerpoint': 'Microsoft PowerPoint presentation software',
            'code': 'Visual Studio Code editor',
            'terminal': 'Command line interface',
            'cmd': 'Command prompt',
            'powershell': 'PowerShell command interface',
            'vlc': 'VLC multimedia player',
            'spotify': 'Music streaming application',
            'discord': 'Voice and text communication platform',
            'control': 'Windows Control Panel',
            'task_manager': 'Windows Task Manager',
            'file_explorer': 'Windows File Explorer'
        }
        
        app_lower = app_name.lower().replace(' ', '_')
        return descriptions.get(app_lower, f"{app_name.replace('_', ' ').title()} application")

    def refresh_database(self) -> int:
        """Refresh the applications database"""
        old_count = len(self.apps_database)
        self.scan_installed_applications()
        new_count = len(self.apps_database)
        return new_count - old_count# Global instance
app_discovery = AppDiscovery()

def discover_applications() -> str:
    """Main function to discover all applications"""
    try:
        apps = app_discovery.scan_installed_applications()
        return f"Successfully discovered {len(apps)} applications on your system."
    except Exception as e:
        return f"Error during application discovery: {e}"

def smart_open_application(app_name: str) -> str:
    """Intelligently open any application by name with usage tracking."""
    print(f"🚀 Smart app launcher: Looking for '{app_name}'...")
    
    # Validate app_name to prevent injection
    if len(app_name) > 200:
        return "❌ Application name is too long"
    
    # First, try to find in discovered apps
    app_path = app_discovery.find_app(app_name)
    
    if app_path:
        # Check if this is a browser proxy (web app, not native)
        is_browser_proxy = any(x in app_path.lower() for x in ['chrome_proxy', 'chrome.exe --app', 'msedge.exe --app'])
        
        # For Spotify specifically, prefer web version if only browser proxy exists
        if is_browser_proxy and 'spotify' in app_name.lower():
            try:
                import webbrowser
                webbrowser.open('https://open.spotify.com')
                app_discovery.track_app_launch(app_name, 'https://open.spotify.com', success=True)
                return f"✅ Opened {app_name} in web browser (native app not installed)"
            except Exception as e:
                app_discovery.track_app_launch(app_name, "", success=False)
                return f"❌ Failed to open {app_name}: {e}"
        
        try:
            if app_path.startswith('explorer.exe shell:appsFolder'):
                # Windows Store app - use subprocess for security
                import subprocess
                subprocess.Popen(app_path, shell=True)
            elif app_path.endswith(':'):
                # Protocol handler (e.g., microsoft.windows.camera:, ms-photos:)
                import subprocess
                subprocess.Popen(['cmd', '/c', 'start', app_path], shell=False)
            elif app_path.lower().endswith('.lnk'):
                # Shortcut file - launch directly to preserve PWA arguments
                import subprocess
                # Use start command to properly handle .lnk files with all their properties
                subprocess.Popen(['cmd', '/c', 'start', '', app_path], shell=False)
            else:
                # Regular executable - use os.startfile (safe)
                os.startfile(app_path)
            
            # Track successful launch
            app_discovery.track_app_launch(app_name, app_path, success=True)
            return f"✅ Successfully opened {app_name}"
        except Exception as e:
            # Track failed launch
            app_discovery.track_app_launch(app_name, app_path, success=False)
            return f"❌ Found {app_name} but failed to launch: {e}"
    else:
        # If not found, try web fallbacks
        web_fallbacks = {
            'youtube music': 'https://music.youtube.com',
            'spotify': 'https://open.spotify.com',
            'whatsapp': 'https://web.whatsapp.com',
            'discord': 'https://discord.com/app',
            'slack': 'https://app.slack.com',
            'zoom': 'https://zoom.us/join',
            'teams': 'https://teams.microsoft.com'
        }
        
        app_lower = app_name.lower()
        if app_lower in web_fallbacks:
            try:
                # Use webbrowser module for security
                import webbrowser
                webbrowser.open(web_fallbacks[app_lower])
                app_discovery.track_app_launch(app_name, web_fallbacks[app_lower], success=True)
                return f"✅ Opened {app_name} web version (desktop app not found)"
            except Exception as e:
                app_discovery.track_app_launch(app_name, "", success=False)
                return f"❌ Failed to open {app_name}: {e}"
        
        app_discovery.track_app_launch(app_name, "", success=False)
        return f"❌ Could not find '{app_name}' on your system. Try saying the full application name or check if it's installed."

def refresh_app_database() -> str:
    """Refresh the application database"""
    try:
        new_apps = app_discovery.refresh_database()
        total_apps = len(app_discovery.get_all_apps())
        return f"Database refreshed! Found {new_apps} new apps. Total: {total_apps} applications."
    except Exception as e:
        return f"Error refreshing database: {e}"

def list_installed_apps() -> str:
    """List all discovered applications"""
    apps = app_discovery.get_all_apps()
    if not apps:
        return "No applications discovered yet. Run application discovery first."
    
    app_list = "\n".join([f"• {name.title()}" for name in sorted(apps.keys())][:50])  # Limit to 50
    total = len(apps)
    
    return f"Found {total} applications (showing first 50):\n{app_list}"

def get_apps_for_web() -> List[Dict[str, str]]:
    """Get applications formatted for web API responses"""
    return app_discovery.get_apps_for_api()

def get_app_usage_stats() -> str:
    """Get application usage statistics."""
    most_used = app_discovery.get_most_used_apps(10)
    recent = app_discovery.get_recent_apps(10)
    
    report = "📊 APPLICATION USAGE STATISTICS\n"
    report += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    
    report += "🔥 MOST USED APPS:\n"
    for i, (app_name, count) in enumerate(most_used, 1):
        report += f"{i}. {app_name.title()}: {count} launches\n"
    
    report += "\n⏰ RECENTLY USED:\n"
    for i, (app_name, last_time) in enumerate(recent, 1):
        report += f"{i}. {app_name.title()} (Last: {last_time})\n"
    
    return report

def search_apps_by_name(query: str) -> str:
    """Search for applications by name."""
    results = app_discovery.search_apps(query, limit=10)
    
    if not results:
        return f"No applications found matching '{query}'"
    
    report = f"🔍 SEARCH RESULTS for '{query}':\n"
    for i, (name, path, score) in enumerate(results, 1):
        report += f"{i}. {name.title()} (Score: {score})\n   Path: {path}\n"
    
    return report