#!/usr/bin/env python3
"""
AI Assistant - Main Entry Point

This is the main entry point for the AI Assistant application.
It provides a unified interface to start the assistant with different
configurations and interfaces.

Features PIN-based authentication for secure access.
"""

import sys
import os
import argparse
from pathlib import Path

# Add the project directories to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def main():
    """Main entry point for the AI Assistant."""
    # Initialize config files from examples if needed
    try:
        from setup_config import setup_config_files
        setup_config_files()
    except Exception as e:
        print(f"⚠️  Warning: Could not auto-initialize config files: {e}")
    
    # Show welcome banner
    print("\n" + "=" * 60)
    print("YourDaddy AI Assistant")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="AI Assistant - Your intelligent companion")
    parser.add_argument("--interface", choices=["cli", "web", "desktop"], default="web",
                       help="Interface to use (default: web)")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--port", type=int, default=8000, help="Port for web interface (default: 8000)")
    parser.add_argument("--setup-pin", action="store_true", help="Setup or change PIN")
    parser.add_argument("--skip-auth", action="store_true", help="Skip PIN authentication (development only)")
    
    args = parser.parse_args()
    
    # Handle PIN setup
    if args.setup_pin:
        try:
            from ai_assistant.auth import setup_pin_cli
            setup_pin_cli()
            return
        except ImportError as e:
            print(f"❌ Error importing PIN authentication: {e}")
            print("Please ensure the authentication module is properly installed.")
            sys.exit(1)
    
    # Authenticate user (unless skipped for development)
    if not args.skip_auth:
        try:
            from ai_assistant.auth import authenticate
            if not authenticate():
                print("❌ Authentication failed. Exiting...")
                sys.exit(1)
        except ImportError as e:
            print(f"❌ Error importing PIN authentication: {e}")
            print("Setting up authentication system...")
            try:
                from ai_assistant.auth import setup_pin_cli
                if not setup_pin_cli():
                    print("❌ Failed to setup authentication. Exiting...")
                    sys.exit(1)
                print("✅ Authentication setup complete. Please restart the assistant.")
                return
            except ImportError:
                print("❌ Authentication module not available. Please check installation.")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            sys.exit(1)
    
    print("\n🚀 Starting AI Assistant...")
    
    try:
        if args.interface == "web":
            print(f"Starting web interface on port {args.port}...")
            # Start the web backend
            try:
                from ai_assistant.apps.modern_web_backend import app, socketio
                print("🌐 Starting YourDaddy Assistant Web Backend...")
                socketio.run(app, host='0.0.0.0', port=args.port, debug=args.verbose)
            except ImportError as e:
                print(f"❌ Web backend import failed: {e}")
                print("Please check your installation and dependencies.")
                sys.exit(1)
                    
        elif args.interface == "cli":
            # Try multiple possible locations for CLI
            try:
                from ai_assistant.apps.app import main as cli_main
                print("Starting CLI interface...")
                cli_main()  # Actually call the CLI main function
            except ImportError:
                try:
                    import app
                    print("Starting CLI interface...")
                    if hasattr(app, 'main'):
                        app.main()
                    else:
                        print("❌ CLI main function not found.")
                        sys.exit(1)
                except ImportError:
                    print("❌ CLI interface not found. Please check your installation.")
                    sys.exit(1)
                    
        elif args.interface == "desktop":
            # Try multiple possible locations for desktop GUI
            try:
                from ai_assistant.apps.yourdaddy_app import main as desktop_main
                print("Starting desktop interface...")
                desktop_main()  # Actually call the desktop main function
            except ImportError:
                try:
                    import yourdaddy_app
                    print("Starting desktop interface...")
                    if hasattr(yourdaddy_app, 'main'):
                        yourdaddy_app.main()
                    else:
                        print("❌ Desktop main function not found.")
                        sys.exit(1)
                except ImportError:
                    print("❌ Desktop interface not found. Please check your installation.")
                    sys.exit(1)
            
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting AI Assistant: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()