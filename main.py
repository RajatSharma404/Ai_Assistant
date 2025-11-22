#!/usr/bin/env python3
"""
AI Assistant - Main Entry Point

This is the main entry point for the AI Assistant application.
It provides a unified interface to start the assistant with different
configurations and interfaces.
"""

import sys
import os
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Main entry point for the AI Assistant."""
    parser = argparse.ArgumentParser(description="AI Assistant - Your intelligent companion")
    parser.add_argument("--interface", choices=["cli", "web", "desktop"], default="web",
                       help="Interface to use (default: web)")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--port", type=int, default=8000, help="Port for web interface (default: 8000)")
    
    args = parser.parse_args()
    
    try:
        if args.interface == "web":
            from src.ai_assistant.apps import modern_web_backend
            print(f"Starting web interface on port {args.port}...")
            # Start web backend
        elif args.interface == "cli":
            from src.ai_assistant.apps import app
            print("Starting CLI interface...")
            # Start CLI version
        elif args.interface == "desktop":
            from src.ai_assistant.apps import yourdaddy_app
            print("Starting desktop interface...")
            # Start desktop GUI
            
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting AI Assistant: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()