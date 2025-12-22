#!/usr/bin/env python3
"""
Simple Voice Interface - Speak, Listen, Process
Web-based solution that bypasses container microphone limitations
"""

from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import os
from pathlib import Path

app = Flask(__name__)
UPLOAD_FOLDER = Path('temp_audio')
UPLOAD_FOLDER.mkdir(exist_ok=True)

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🎤 Voice Assistant</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            max-width: 600px;
            width: 100%;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 2em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        .status {
            text-align: center;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-weight: 500;
            transition: all 0.3s;
        }
        .status.idle { background: #e3f2fd; color: #1976d2; }
        .status.listening { background: #fff3e0; color: #f57c00; animation: pulse 1.5s infinite; }
        .status.processing { background: #f3e5f5; color: #7b1fa2; }
        .status.success { background: #e8f5e9; color: #388e3c; }
        .status.error { background: #ffebee; color: #d32f2f; }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        
        .mic-button {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 3em;
            cursor: pointer;
            display: block;
            margin: 30px auto;
            transition: all 0.3s;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }
        .mic-button:hover {
            transform: scale(1.05);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
        }
        .mic-button:active {
            transform: scale(0.95);
        }
        .mic-button.listening {
            animation: pulse-mic 1s infinite;
            background: linear-gradient(135deg, #f57c00 0%, #ff6f00 100%);
        }
        
        @keyframes pulse-mic {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        
        .result-box {
            background: #f5f5f5;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            min-height: 100px;
            display: none;
        }
        .result-box.show { display: block; }
        .result-label {
            font-size: 0.8em;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .result-text {
            font-size: 1.2em;
            color: #333;
            line-height: 1.5;
        }
        .instructions {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            font-size: 0.9em;
            color: #666;
        }
        .instructions ul {
            margin-left: 20px;
            margin-top: 10px;
        }
        .instructions li {
            margin: 5px 0;
        }
        .browser-support {
            text-align: center;
            margin-top: 15px;
            font-size: 0.85em;
            color: #999;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Voice Assistant</h1>
        <p class="subtitle">Speak → Listen → Process</p>
        
        <div id="status" class="status idle">
            Ready to listen
        </div>
        
        <button id="micButton" class="mic-button" onclick="toggleRecording()">
            🎤
        </button>
        
        <div id="resultBox" class="result-box">
            <div class="result-label">You said:</div>
            <div id="resultText" class="result-text"></div>
        </div>
        
        <div class="instructions">
            <strong>📋 How to use:</strong>
            <ul>
                <li>Click the microphone button</li>
                <li>Allow microphone access when prompted</li>
                <li>Speak clearly</li>
                <li>Your speech will be transcribed automatically</li>
            </ul>
        </div>
        
        <div class="browser-support">
            ✅ Works in Chrome, Edge, Safari
        </div>
    </div>

    <script>
        let recognition;
        let isListening = false;
        
        // Check browser support
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';
            
            recognition.onstart = () => {
                updateStatus('listening', '🎤 Listening... Speak now!');
            };
            
            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                const confidence = event.results[0][0].confidence;
                
                document.getElementById('resultText').textContent = transcript;
                document.getElementById('resultBox').classList.add('show');
                
                updateStatus('processing', '⏳ Processing...');
                
                // Send to backend for processing
                fetch('/process_speech', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        text: transcript,
                        confidence: confidence
                    })
                })
                .then(response => response.json())
                .then(data => {
                    updateStatus('success', '✅ Processed: ' + data.message);
                    console.log('Backend response:', data);
                })
                .catch(error => {
                    updateStatus('error', '❌ Processing failed');
                    console.error('Error:', error);
                });
            };
            
            recognition.onerror = (event) => {
                updateStatus('error', '❌ Error: ' + event.error);
                isListening = false;
                document.getElementById('micButton').classList.remove('listening');
            };
            
            recognition.onend = () => {
                isListening = false;
                document.getElementById('micButton').classList.remove('listening');
                if (document.getElementById('status').classList.contains('listening')) {
                    updateStatus('idle', 'Ready to listen');
                }
            };
        } else {
            updateStatus('error', '❌ Speech recognition not supported in this browser');
        }
        
        function toggleRecording() {
            if (!recognition) {
                alert('Speech recognition not supported in this browser. Please use Chrome, Edge, or Safari.');
                return;
            }
            
            if (isListening) {
                recognition.stop();
                isListening = false;
                document.getElementById('micButton').classList.remove('listening');
            } else {
                recognition.start();
                isListening = true;
                document.getElementById('micButton').classList.add('listening');
            }
        }
        
        function updateStatus(type, message) {
            const statusEl = document.getElementById('status');
            statusEl.className = 'status ' + type;
            statusEl.textContent = message;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/process_speech', methods=['POST'])
def process_speech():
    """Process the transcribed speech"""
    data = request.json
    text = data.get('text', '')
    confidence = data.get('confidence', 0)
    
    print(f"\n🎤 Received speech: '{text}'")
    print(f"   Confidence: {confidence:.2%}")
    
    # Here you can add your AI processing logic
    # For now, just acknowledge receipt
    
    response = {
        'status': 'success',
        'message': f'Received: {text}',
        'text': text,
        'confidence': confidence
    }
    
    # TODO: Add your AI processing here
    # - Send to AI model
    # - Generate response
    # - Execute commands
    # etc.
    
    return jsonify(response)

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """Alternative: Upload audio file for processing"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    audio_file = request.files['audio']
    filepath = UPLOAD_FOLDER / audio_file.filename
    audio_file.save(filepath)
    
    # Process with speech recognition
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(str(filepath)) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            
        return jsonify({
            'status': 'success',
            'text': text
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    finally:
        # Cleanup
        if filepath.exists():
            filepath.unlink()

if __name__ == '__main__':
    print("="*60)
    print("🎤 Voice Assistant Web Interface")
    print("="*60)
    print("\n✅ Starting server...")
    print("\n📋 Instructions:")
    print("   1. Open your browser to: http://localhost:5000")
    print("   2. Click the microphone button")
    print("   3. Allow microphone access")
    print("   4. Speak clearly")
    print("   5. Your speech will be transcribed and processed")
    print("\n⚠️  Note: Works in Chrome, Edge, Safari")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
