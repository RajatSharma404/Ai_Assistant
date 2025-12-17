# Production server improvements
# Apply these changes to the __main__ block at the end of modern_web_backend.py

if __name__ == '__main__':
    print("=" * 60)
    print("Ì∫Ä YourDaddy Assistant - Modern Web Backend")
    print("=" * 60)
    print("Ìºê Server starting on: http://localhost:5000")
    print("Ì≥± React frontend will be served automatically")
    print("‚ö° Real-time features enabled via WebSockets")
    print("Ì¥ß API endpoints available at /api/*")
    print("Ìªë Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Bind to localhost only for security
        host = os.getenv('HOST', '127.0.0.1')
        port = int(os.getenv('PORT', 5000))
        debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'
        
        print(f"Ì¥í Security: JWT authentication enabled")
        print(f"Ì¥í Security: Rate limiting enabled (200/hour, 50/min)")
        print(f"Ì¥í Security: CORS restricted to: {', '.join(ALLOWED_ORIGINS)}")
        print(f"Ì¥í Security: Host binding: {host}")
        print("")
        
        # Password warning with better visibility
        admin_pwd = os.getenv('ADMIN_PASSWORD', 'YourDaddy2025!SecureAdmin')
        if admin_pwd == 'YourDaddy2025!SecureAdmin':
            print("‚ö†Ô∏è  WARNING: Using default admin password!")
            print(f"Ì¥ë Default credentials: username='admin', password='{admin_pwd}'")
            print("‚ö†Ô∏è  CHANGE THE PASSWORD in .env file before production!")
        else:
            print("‚úÖ Custom admin password configured")
        print("")
        
        # Production deployment warning
        if not debug_mode:
            print("‚ö†Ô∏è  DEVELOPMENT SERVER - Not suitable for production!")
            print("Ì¥ß For production deployment, use a proper WSGI server:")
            print("   Example: gunicorn -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker")
            print("            -w 1 ai_assistant.services.modern_web_backend:app")
            print("")
        
        socketio.run(app, host=host, port=port, debug=debug_mode, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n\nÌªë Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        print(f"\n‚ùå Server failed to start: {e}")
        print("\nÌ¥ß Troubleshooting:")
        print("   1. Check if port is already in use: netstat -ano | findstr :5000")
        print("   2. Verify all dependencies are installed: pip install -r requirements.txt")
        print("   3. Check logs in logs/backend/ for detailed error messages")
        sys.exit(1)
