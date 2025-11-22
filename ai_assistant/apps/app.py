#!/usr/bin/env python3
"""
YourDaddy Assistant - Unified Main Application
Entry point that includes system checks, configuration, and application startup
"""

import sys
import os
import traceback
import time
import argparse
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))
sys.path.insert(0, str(project_root / "utils"))

# Import utilities
try:
    from utils.logging_config import get_logger, setup_logging
    setup_logging()
    logger = get_logger(__name__)
except ImportError:
    # Fallback logging if advanced logging fails
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

def show_help():
    """Show help information"""
    print("🤖 YourDaddy Assistant v3.1 - Usage Guide")
    print("=" * 50)
    print()
    print("Usage: python app.py [mode] [options]")
    print()
    print("Modes:")
    print("  gui        Launch with graphical interface (default)")
    print("  cli        Launch command-line interface")
    print("  web        Start web backend server only")
    print("  check      Run system check only")
    print("  setup      Run setup wizard")
    print("  test       Run test suite")
    print("  help       Show this help message")
    print()
    print("Options:")
    print("  --no-checks    Skip system compatibility checks")
    print("  --debug        Enable debug mode")
    print("  --port PORT    Set web server port (default: 5000)")
    print("  --host HOST    Set web server host (default: 0.0.0.0)")
    print()
    print("Examples:")
    print("  python app.py                # Start GUI mode")
    print("  python app.py web            # Start web server")
    print("  python app.py cli --debug    # CLI with debug mode")
    print("  python app.py check          # Check system only")

def run_system_check():
    """Run system compatibility check"""
    try:
        from launch_assistant import SystemChecker
        checker = SystemChecker()
        return checker.run_full_check()
    except ImportError:
        logger.warning("System checker not available")
        return True

def start_web_server(host='0.0.0.0', port=5000):
    """Start the web backend server"""
    try:
        logger.info(f"🌐 Starting web server on {host}:{port}")
        
        # Set environment variables for backend configuration
        os.environ['HOST'] = str(host)
        os.environ['PORT'] = str(port)
        
        # Import and start backend
        from backend import main as start_backend
        start_backend()
        
    except ImportError:
        logger.error("❌ Web backend not available")
        return False
    except Exception as e:
        logger.error(f"❌ Web server failed: {e}")
        traceback.print_exc()
        return False

def start_gui_app():
    """Start the GUI application"""
    try:
        logger.info("🖥️ Starting GUI application")
        
        from yourdaddy_app import YourDaddyAssistant
        assistant = YourDaddyAssistant()
        assistant.run()
        
    except ImportError:
        logger.error("❌ GUI application not available")
        return False
    except Exception as e:
        logger.error(f"❌ GUI failed: {e}")
        traceback.print_exc()
        return False

def start_cli_app():
    """Start the command-line interface"""
    try:
        logger.info("💻 Starting CLI application")
        
        # Simple CLI loop
        try:
            from modules.conversational_ai import AdvancedConversationalAI
            ai = AdvancedConversationalAI()
            
            print("\n🤖 YourDaddy Assistant CLI")
            print("Type 'quit', 'exit', or Ctrl+C to stop\n")
            
            while True:
                user_input = input("You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                if user_input:
                    print("Assistant: ", end="", flush=True)
                    def stream_callback(token):
                        print(token, end="", flush=True)
                    response = ai.generate_model_response(user_input, stream_callback=stream_callback)
                    print("\n")
        
        except ImportError:
            # Fallback simple CLI
            print("\n🤖 YourDaddy Assistant - Simple CLI")
            print("(Advanced AI not available)")
            print("Type 'quit' to exit\n")
            
            while True:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                if user_input:
                    print(f"Assistant: I received your message: {user_input}")
                    print("(Connect to web interface for full functionality)\n")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"❌ CLI failed: {e}")
        traceback.print_exc()
        return False

def run_setup():
    """Run the setup wizard"""
    try:
        logger.info("⚙️ Starting setup wizard")
        from setup import main as setup_main
        setup_main()
    except ImportError:
        logger.error("❌ Setup wizard not available")
        return False
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        traceback.print_exc()
        return False

def run_tests():
    """Run the test suite"""
    try:
        logger.info("🧪 Starting test suite")
        
        # Try different test options
        test_files = [
            ('test_chat.py', 'Chat functionality tests'),
            ('test_integration.py', 'Integration tests'),
            ('debug.py', 'Debug and diagnostics')
        ]
        
        print("\nAvailable test suites:")
        for i, (file, desc) in enumerate(test_files, 1):
            exists = os.path.exists(file)
            status = "✅" if exists else "❌"
            print(f"{i}. {desc} {status}")
        
        choice = input("\nSelect test suite (1-3, or Enter for all): ").strip()
        
        if choice == "1" and os.path.exists('test_chat.py'):
            os.system(f"{sys.executable} test_chat.py")
        elif choice == "2" and os.path.exists('test_integration.py'):
            os.system(f"{sys.executable} test_integration.py")
        elif choice == "3" and os.path.exists('debug.py'):
            os.system(f"{sys.executable} debug.py")
        else:
            # Run all available tests
            for file, desc in test_files:
                if os.path.exists(file):
                    print(f"\n{'='*50}")
                    print(f"Running: {desc}")
                    print(f"{'='*50}")
                    os.system(f"{sys.executable} {file}")
        
    except Exception as e:
        logger.error(f"❌ Tests failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='YourDaddy Assistant v3.1')
    parser.add_argument('mode', nargs='?', default='gui', 
                       choices=['gui', 'cli', 'web', 'check', 'setup', 'test', 'help'],
                       help='Application mode')
    parser.add_argument('--no-checks', action='store_true', 
                       help='Skip system compatibility checks')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode')
    parser.add_argument('--port', type=int, default=5000, 
                       help='Web server port (default: 5000)')
    parser.add_argument('--host', default='0.0.0.0', 
                       help='Web server host (default: 0.0.0.0)')
    
    args = parser.parse_args()
    
    # Show help
    if args.mode == 'help':
        show_help()
        return
    
    # Set debug mode
    if args.debug:
        logger.setLevel(logging.DEBUG)
        os.environ['DEBUG'] = 'true'
    
    print("🤖 YourDaddy Assistant v3.1")
    print("=" * 40)
    
    # Run system checks (unless skipped)
    if not args.no_checks and args.mode != 'check':
        logger.info("🔍 Running system checks...")
        if not run_system_check():
            logger.warning("⚠️ System check failed, but continuing anyway...")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                logger.info("Exiting due to system check failure")
                return
    
    # Route to appropriate mode
    try:
        if args.mode == 'check':
            success = run_system_check()
            if success:
                print("✅ All system checks passed!")
            else:
                print("❌ Some system checks failed")
            return
        
        elif args.mode == 'web':
            start_web_server(args.host, args.port)
        
        elif args.mode == 'gui':
            start_gui_app()
        
        elif args.mode == 'cli':
            start_cli_app()
        
        elif args.mode == 'setup':
            run_setup()
        
        elif args.mode == 'test':
            run_tests()
        
        else:
            logger.error(f"Unknown mode: {args.mode}")
            show_help()
    
    except KeyboardInterrupt:
        logger.info("\n👋 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Application failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()