# Dynamic Application Discovery Module
"""
This module scans the system to discover all installed applications
and provides intelligent app launching based on voice commands.
"""

import os
import platform
try:
    import winreg
    HAS_WINREG = True
except ImportError:
    HAS_WINREG = False
    winreg = None
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
        Scan officially registered applications from all sources
        """
        print("🔍 Scanning system applications...")
        apps = {}
        
        system = platform.system().lower()
        
        if system == "windows":
            print("\n📋 Windows App Discovery")
            print("=" * 50)
            
            # Method 1: Windows Registry - Apps & Features from Settings
            print("\n1️⃣  Programs & Features (Registry)")
            apps.update(self._scan_registry_programs())
            
            # Method 2: Start Menu - All Apps
            print("\n2️⃣  Start Menu (All Apps)")
            apps.update(self._scan_start_menu())
            
            # Method 3: Essential Windows Store apps
            print("\n3️⃣  Windows Store Apps (Essential)")
            essential_apps = self._scan_essential_store_apps()
            print(f"    ✅ Added {len(essential_apps)} essential Store apps")
            apps.update(essential_apps)
            
            # Method 4: Common system utilities
            print("\n4️⃣  System Utilities")
            system_utils = self._get_system_utilities()
            print(f"    ✅ Added {len(system_utils)} system utilities")
            apps.update(system_utils)
            
        elif system == "linux":
            print("\n🐧 Linux App Discovery")
            print("=" * 50)
            
            # Method 1: Desktop files
            apps.update(self._scan_linux_desktop_files())
            
            # Method 2: Common binary directories
            apps.update(self._scan_linux_bin_dirs())
            
            # Method 3: Snap packages
            apps.update(self._scan_snap_packages())
            
            # Method 4: Flatpak packages
            apps.update(self._scan_flatpak_packages())
            
            # Method 5: Common system utilities
            apps.update(self._get_linux_system_utilities())
            
        elif system == "darwin":  # macOS
            print("\n🍎 macOS App Discovery")
            print("=" * 50)
            
            # macOS-specific scanning
            apps.update(self._scan_macos_applications())
            apps.update(self._get_macos_system_utilities())
        
        else:
            print(f"⚠️  Unsupported operating system: {system}")
            apps.update(self._get_cross_platform_utilities())
        
        # Save to cache
        self.apps_database = apps
        self.save_cache()
        
        print(f"\n{'=' * 50}")
        print(f"✅ Discovery complete! Found {len(apps)} applications.")
        print(f"{'=' * 50}\n")
        
        return apps
    
    def _scan_registry_programs(self) -> Dict[str, str]:
        """Scan Windows Registry for installed programs (Programs & Features)"""
        apps = {}
        
        if not HAS_WINREG:
            return apps  # Skip registry scan on non-Windows systems
        
        print("    🔍 Scanning registry keys...")
        
        # Scan both HKEY_LOCAL_MACHINE and HKEY_CURRENT_USER
        registry_locations = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")
        ]
        
        total_found = 0
        for hive, reg_path in registry_locations:
            try:
                with winreg.OpenKey(hive, reg_path) as key:
                    num_subkeys = winreg.QueryInfoKey(key)[0]
                    for i in range(num_subkeys):
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            with winreg.OpenKey(key, subkey_name) as subkey:
                                try:
                                    # Get DisplayName
                                    display_name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                                    if not display_name or len(display_name.strip()) == 0:
                                        continue
                                    
                                    # Skip system components and updates
                                    if any(skip in display_name.lower() for skip in 
                                          ['update', 'hotfix', 'security update', 'kb', 'redistributable']):
                                        continue
                                    
                                    app_key = display_name.lower().strip()
                                    
                                    # Try to get executable path in priority order
                                    exe_path = None
                                    
                                    # Method 1: DisplayIcon (often points to exe)
                                    try:
                                        icon_path = winreg.QueryValueEx(subkey, "DisplayIcon")[0]
                                        if icon_path:
                                            # Remove quotes and icon index
                                            icon_path = icon_path.strip('"').split(',')[0]
                                            if icon_path.lower().endswith('.exe') and os.path.exists(icon_path):
                                                exe_path = icon_path
                                    except FileNotFoundError:
                                        pass
                                    
                                    # Method 2: InstallLocation + search for exe
                                    if not exe_path:
                                        try:
                                            install_location = winreg.QueryValueEx(subkey, "InstallLocation")[0]
                                            if install_location and os.path.exists(install_location):
                                                # Look for main executable
                                                exe_files = glob.glob(os.path.join(install_location, "*.exe"))
                                                if exe_files:
                                                    exe_path = self._find_main_executable(exe_files, display_name)
                                        except FileNotFoundError:
                                            pass
                                    
                                    # Method 3: UninstallString (may contain exe path)
                                    if not exe_path:
                                        try:
                                            uninstall_string = winreg.QueryValueEx(subkey, "UninstallString")[0]
                                            if uninstall_string:
                                                # Extract potential exe path
                                                parts = uninstall_string.strip('"').split('"')
                                                for part in parts:
                                                    if '.exe' in part.lower() and os.path.exists(part.strip()):
                                                        # Verify it's the app exe, not uninstaller
                                                        if 'uninstall' not in part.lower():
                                                            exe_path = part.strip()
                                                            break
                                        except FileNotFoundError:
                                            pass
                                    
                                    if exe_path:
                                        apps[app_key] = exe_path
                                        total_found += 1
                                        
                                except (FileNotFoundError, OSError):
                                    continue
                        except OSError:
                            continue
            except FileNotFoundError:
                # Registry key doesn't exist
                continue
            except Exception as e:
                print(f"    ⚠️  Error scanning {reg_path}: {e}")
        
        print(f"    ✅ Found {total_found} apps from registry")
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
        """Scan Start Menu shortcuts (All Apps)"""
        apps = {}
        
        print("    🔍 Scanning Start Menu locations...")
        
        # All Start Menu locations
        start_menu_paths = [
            os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs"),  # User apps
            os.path.expandvars(r"%PROGRAMDATA%\Microsoft\Windows\Start Menu\Programs"),  # All users
            os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu"),  # Root user
            os.path.expandvars(r"%PROGRAMDATA%\Microsoft\Windows\Start Menu")  # Root all users
        ]
        
        total_found = 0
        processed_shortcuts = set()  # Avoid duplicates
        
        for start_path in start_menu_paths:
            if not os.path.exists(start_path):
                continue
            
            # Recursively walk through all folders
            for root, dirs, files in os.walk(start_path):
                # Skip uninstall folders
                if 'uninstall' in root.lower():
                    continue
                
                for file in files:
                    if not file.endswith('.lnk'):
                        continue
                    
                    shortcut_path = os.path.join(root, file)
                    
                    # Skip if already processed
                    if shortcut_path in processed_shortcuts:
                        continue
                    processed_shortcuts.add(shortcut_path)
                    
                    # Get app name from filename
                    app_name = file[:-4]  # Remove .lnk extension
                    
                    # Skip uninstall shortcuts
                    if 'uninstall' in app_name.lower():
                        continue
                    
                    app_key = app_name.lower().strip()
                    
                    # Skip if already found (prefer user apps over system apps)
                    if app_key in apps:
                        continue
                    
                    # Resolve the shortcut target
                    target = self._resolve_shortcut(shortcut_path)
                    
                    if target and os.path.exists(target):
                        # Valid target found
                        if target.lower().endswith('.exe'):
                            # Check if it's a PWA (browser proxy executables)
                            pwa_proxies = ['chrome_proxy.exe', 'msedge_proxy.exe', 'brave_proxy.exe', 
                                          'opera_proxy.exe', 'vivaldi_proxy.exe', 'arc_proxy.exe']
                            is_pwa = any(proxy in target.lower() for proxy in pwa_proxies)
                            
                            if is_pwa:
                                # Store the .lnk path for PWAs to preserve app-id arguments
                                apps[app_key] = shortcut_path
                            else:
                                apps[app_key] = target
                            total_found += 1
                        elif target.endswith('.lnk'):
                            # Nested shortcut, use it
                            apps[app_key] = target
                            total_found += 1
                    else:
                        # No valid target - likely UWP/Store app
                        # Use the shortcut path itself - Windows will handle it
                        apps[app_key] = shortcut_path
                        total_found += 1
        
        print(f"    ✅ Found {total_found} apps from Start Menu")
        return apps
    
    def _get_system_utilities(self) -> Dict[str, str]:
        """Get common Windows system utilities"""
        return {
            'notepad': 'notepad.exe',
            'calculator': 'calc.exe',
            'paint': 'mspaint.exe',
            'wordpad': 'wordpad.exe',
            'command prompt': 'cmd.exe',
            'powershell': 'powershell.exe',
            'task manager': 'taskmgr.exe',
            'control panel': 'control.exe',
            'registry editor': 'regedit.exe',
            'file explorer': 'explorer.exe',
            'snipping tool': 'SnippingTool.exe',
            'magnifier': 'magnify.exe'
        }
    
    def _scan_linux_desktop_files(self) -> Dict[str, str]:
        """Scan Linux .desktop files for installed applications"""
        apps = {}
        desktop_dirs = [
            "/usr/share/applications",
            "/usr/local/share/applications",
            os.path.expanduser("~/.local/share/applications")
        ]
        
        for desktop_dir in desktop_dirs:
            if os.path.exists(desktop_dir):
                for file in os.listdir(desktop_dir):
                    if file.endswith('.desktop'):
                        desktop_path = os.path.join(desktop_dir, file)
                        try:
                            app_info = self._parse_desktop_file(desktop_path)
                            if app_info:
                                apps.update(app_info)
                        except Exception as e:
                            # Skip problematic desktop files
                            continue
        
        return apps
    
    def _parse_desktop_file(self, file_path: str) -> Dict[str, str]:
        """Parse a .desktop file and extract app information"""
        apps = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if it's a valid application
            if 'Type=Application' not in content:
                return apps
            
            # Extract name and exec
            name = ""
            exec_cmd = ""
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('Name=') and not name:
                    name = line.split('=', 1)[1]
                elif line.startswith('Exec=') and not exec_cmd:
                    exec_cmd = line.split('=', 1)[1]
            
            if name and exec_cmd:
                # Clean up exec command (remove % parameters)
                exec_cmd = exec_cmd.split()[0]  # Take first part
                if os.path.exists(exec_cmd) or self._is_command_available(exec_cmd):
                    apps[name.lower()] = exec_cmd
                    
        except Exception:
            pass
        
        return apps
    
    def _is_command_available(self, command: str) -> bool:
        """Check if a command is available in PATH"""
        try:
            subprocess.run(['which', command], capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def _scan_linux_bin_dirs(self) -> Dict[str, str]:
        """Scan common Linux binary directories"""
        apps = {}
        bin_dirs = [
            "/usr/bin",
            "/usr/local/bin",
            "/bin",
            "/opt/bin",
            os.path.expanduser("~/.local/bin")
        ]
        
        common_apps = [
            'firefox', 'chromium', 'chrome', 'opera', 'vivaldi',
            'code', 'vscode', 'sublime_text', 'atom', 'gedit', 'nano', 'vim',
            'libreoffice', 'soffice', 'thunderbird', 'evolution',
            'rhythmbox', 'vlc', 'totem', 'mpv', 'smplayer',
            'gimp', 'inkscape', 'blender', 'krita',
            'terminal', 'gnome-terminal', 'konsole', 'xfce4-terminal', 'xterm',
            'nautilus', 'dolphin', 'thunar', 'pcmanfm',
            'calculator', 'gnome-calculator', 'kcalc', 'galculator'
        ]
        
        for app in common_apps:
            for bin_dir in bin_dirs:
                app_path = os.path.join(bin_dir, app)
                if os.path.exists(app_path) and os.access(app_path, os.X_OK):
                    apps[app] = app_path
                    break
        
        return apps
    
    def _scan_snap_packages(self) -> Dict[str, str]:
        """Scan for Snap packages"""
        apps = {}
        try:
            result = subprocess.run(['snap', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    parts = line.split()
                    if parts:
                        app_name = parts[0]
                        # Snap apps are launched with 'snap run app_name'
                        apps[app_name] = f"snap run {app_name}"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return apps
    
    def _scan_flatpak_packages(self) -> Dict[str, str]:
        """Scan for Flatpak packages"""
        apps = {}
        try:
            result = subprocess.run(['flatpak', 'list', '--app'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        app_id = parts[0]
                        app_name = parts[1] if len(parts) > 1 else app_id
                        # Flatpak apps are launched with 'flatpak run app_id'
                        apps[app_name.lower()] = f"flatpak run {app_id}"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return apps
    
    def _get_linux_system_utilities(self) -> Dict[str, str]:
        """Get common Linux system utilities"""
        return {
            'terminal': 'xterm',
            'file manager': 'nautilus',
            'text editor': 'gedit',
            'calculator': 'gnome-calculator',
            'system monitor': 'gnome-system-monitor',
            'settings': 'gnome-control-center',
            'software center': 'gnome-software'
        }
    
    def _scan_macos_applications(self) -> Dict[str, str]:
        """Scan macOS Applications directory"""
        apps = {}
        app_dirs = [
            "/Applications",
            "/System/Applications",
            os.path.expanduser("~/Applications")
        ]
        
        for app_dir in app_dirs:
            if os.path.exists(app_dir):
                for item in os.listdir(app_dir):
                    if item.endswith('.app'):
                        app_path = os.path.join(app_dir, item)
                        app_name = item[:-4]  # Remove .app extension
                        apps[app_name.lower()] = f"open -a '{app_name}'"
        
        return apps
    
    def _get_macos_system_utilities(self) -> Dict[str, str]:
        """Get common macOS system utilities"""
        return {
            'terminal': 'open -a Terminal',
            'finder': 'open -a Finder',
            'textedit': 'open -a TextEdit',
            'calculator': 'open -a Calculator',
            'system preferences': 'open -a "System Preferences"',
            'activity monitor': 'open -a "Activity Monitor"'
        }
    
    def _get_cross_platform_utilities(self) -> Dict[str, str]:
        """Get basic cross-platform utilities"""
        return {
            'python': 'python3',
            'python3': 'python3',
            'pip': 'pip3'
        }
    
    def _find_main_executable(self, exe_files: List[str], app_name: str) -> str:
        """Find the main executable from a list of exe files"""
        if not exe_files:
            return ""
        
        # Prefer executable that matches app name
        for exe in exe_files:
            exe_name = os.path.basename(exe).lower()
            if app_name.lower() in exe_name or exe_name.replace('.exe', '') in app_name.lower():
                return exe
        
        # Prefer shortest path (likely main executable)
        return min(exe_files, key=len)
    
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
                    
                # Always ensure system utilities are included
                system = platform.system().lower()
                if system == "windows":
                    self.apps_database.update(self._get_system_utilities())
                elif system == "linux":
                    self.apps_database.update(self._get_linux_system_utilities())
                elif system == "darwin":
                    self.apps_database.update(self._get_macos_system_utilities())
                else:
                    self.apps_database.update(self._get_cross_platform_utilities())
        except Exception as e:
            print(f"Error loading cache: {e}")
            self.apps_database = {}
            # Ensure system utilities are available even if cache fails
            system = platform.system().lower()
            if system == "windows":
                self.apps_database.update(self._get_system_utilities())
            elif system == "linux":
                self.apps_database.update(self._get_linux_system_utilities())
            elif system == "darwin":
                self.apps_database.update(self._get_macos_system_utilities())
            else:
                self.apps_database.update(self._get_cross_platform_utilities())
    
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