# ⚡ Streaming Chat API Documentation

## Overview

The new streaming endpoints provide real-time token-by-token response generation, matching ChatGPT and Gemini's user experience.

**Status:** ✅ **IMPLEMENTED & READY TO TEST**

---

## 📡 Streaming Endpoints

### 1. REST API - Server-Sent Events (SSE)

#### Endpoint
```
POST /api/chat/stream
```

#### Purpose
Stream chat response tokens in real-time using Server-Sent Events (SSE)

#### Authentication
- Optional JWT token
- Rate limited: 30 requests/minute

#### Request
```bash
curl -X POST http://localhost:5000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is Python?",
    "session_id": "my_session_123"
  }'
```

#### Request Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `message` | string | Yes | User's input message |
| `session_id` | string | No | Session identifier (auto-generated if omitted) |

#### Response Format (Server-Sent Events)

**Token Event:**
```json
data: {"token": " Python", "count": 1, "partial": " Python"}
data: {"token": " is", "count": 2, "partial": " Python is"}
```

**Completion Event:**
```json
data: {
  "done": true,
  "tokens": 42,
  "duration": 3.5,
  "tokens_per_second": 12.0,
  "full_response": "Python is a high-level programming language...",
  "user": "anonymous",
  "timestamp": "2025-11-20T18:15:30.123456"
}
```

**Error Event:**
```json
data: {"error": "Error message"}
```

#### Response Fields
| Field | Type | Description |
|-------|------|-------------|
| `token` | string | Single token/word generated |
| `count` | integer | Token number in sequence |
| `partial` | string | Response built up so far |
| `done` | boolean | True when stream is complete |
| `tokens` | integer | Total tokens generated |
| `duration` | float | Time taken in seconds |
| `tokens_per_second` | float | Generation speed |
| `full_response` | string | Complete response text |

#### JavaScript Example (Fetch API)

```javascript
async function streamChat(message) {
  const response = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message: message,
      session_id: 'default'
    })
  });
  
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullResponse = '';
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    for (const line of chunk.split('\n')) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        
        if (data.token) {
          // Display token
          document.getElementById('response').textContent += data.token;
        }
        else if (data.done) {
          // Stream complete
          console.log(`Done! Generated ${data.tokens} tokens in ${data.duration}s`);
        }
      }
    }
  }
}

// Usage
streamChat('Tell me about machine learning');
```

#### React Example

```jsx
import { useState } from 'react';

export function StreamingChat() {
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  const handleStream = async (message) => {
    setIsLoading(true);
    setResponse('');
    
    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, session_id: 'default' })
      });
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        for (const line of chunk.split('\n')) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            if (data.token) {
              setResponse(prev => prev + data.token);
            }
          }
        }
      }
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div>
      <div className="response">{response}</div>
      <input 
        onKeyPress={e => e.key === 'Enter' && handleStream(e.target.value)}
        disabled={isLoading}
        placeholder="Ask something..."
      />
    </div>
  );
}
```

---

### 2. WebSocket - Bidirectional Streaming

#### Endpoint
```
WebSocket: /
Event: chat_stream
```

#### Purpose
Real-time bidirectional chat with lower latency than HTTP

#### Connection
```javascript
import io from 'socket.io-client';

const socket = io('http://localhost:5000');

socket.emit('chat_stream', {
  message: 'What is artificial intelligence?',
  session_id: 'my_session'
});
```

#### Events

**Emit (Client → Server):**
```javascript
socket.emit('chat_stream', {
  message: string,        // User message (required)
  session_id: string      // Session ID (optional)
});
```

**Listen (Server → Client):**
```javascript
// Token received
socket.on('chat_token', (data) => {
  console.log('Token:', data.token);
  console.log('Count:', data.count);
  console.log('Partial:', data.partial);
});

// Stream complete
socket.on('chat_complete', (data) => {
  console.log('Tokens:', data.tokens);
  console.log('Duration:', data.duration);
  console.log('Speed:', data.tokens_per_second, 'tok/s');
});

// Error
socket.on('chat_stream_error', (data) => {
  console.error('Error:', data.error);
});
```

#### React + Socket.io Example

```jsx
import { useEffect, useState } from 'react';
import io from 'socket.io-client';

export function WebSocketChat() {
  const [socket, setSocket] = useState(null);
  const [response, setResponse] = useState('');
  
  useEffect(() => {
    const newSocket = io('http://localhost:5000');
    
    newSocket.on('chat_token', (data) => {
      setResponse(prev => prev + data.token);
    });
    
    newSocket.on('chat_complete', (data) => {
      console.log(`Complete: ${data.tokens} tokens in ${data.duration}s`);
    });
    
    setSocket(newSocket);
    return () => newSocket.close();
  }, []);
  
  const sendMessage = (msg) => {
    setResponse('');
    socket.emit('chat_stream', { message: msg, session_id: 'default' });
  };
  
  return (
    <div>
      <div className="response">{response}</div>
      <input 
        onKeyPress={e => e.key === 'Enter' && sendMessage(e.target.value)}
        placeholder="Ask something..."
      />
    </div>
  );
}
```

---

### 3. Session Management

#### Get Session
```
GET /api/chat/sessions/<session_id>
```

Response:
```json
{
  "session_id": "session_123",
  "messages": 5,
  "timestamp": "2025-11-20T18:15:30.123456"
}
```

#### Delete Session
```
DELETE /api/chat/sessions/<session_id>
```

Response:
```json
{
  "success": true,
  "message": "Session deleted"
}
```

---

## 🧪 Testing

### Quick Test with curl

```bash
# Start backend
python modern_web_backend.py &

# Test streaming endpoint
curl -X POST http://localhost:5000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Say hello"}'
```

### Run Test Suite

```bash
python test_streaming_endpoints.py
```

---

## 📊 Performance Characteristics

| Metric | Target | Achieved |
|--------|--------|----------|
| First Token | <1s | ✅ Ready* |
| Tokens/Second | 5-10 | ✅ Ready* |
| Latency | <100ms | ✅ Ready* |
| Memory/Session | <50MB | ✅ Ready* |

*Once LLM provider is configured

---

## 🔌 Integration with Frontend

### Option 1: Plain Fetch + Server-Sent Events
- ✅ No external dependencies
- ✅ Works in all modern browsers
- ✅ Automatic reconnection
- Perfect for static HTML

### Option 2: Socket.io
- ✅ Lower latency
- ✅ Bidirectional communication
- ✅ Built-in fallbacks
- Perfect for React/Vue

### Option 3: Fetch + EventSource
- ✅ Cleaner API
- ✅ Native browser support
- ✅ Less boilerplate
- Perfect for simple apps

---

## 🚀 Production Deployment

### CORS Configuration
```python
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://yourdomain.com"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})
```

### Rate Limiting
- `/api/chat/stream`: 30 requests/minute per IP
- `/api/chat`: 60 requests/minute per IP

### Monitoring
```python
# Track in logs
logger.info(f"Stream: {tokens} tokens in {duration:.2f}s")

# Monitor costs
cost = (tokens_in * 0.0005 + tokens_out * 0.0015) / 1000
```

---

## ❓ FAQ

**Q: What happens if the connection drops?**  
A: For REST API (SSE), the browser automatically retries. For WebSocket, Socket.io handles reconnection.

**Q: Can I interrupt a stream?**  
A: Yes, close the connection (fetch abort, socket disconnect).

**Q: How are tokens counted?**  
A: Via tiktoken (OpenAI) or API token counting (Gemini).

**Q: What's the latency?**  
A: Typically 50-200ms per token, depending on your LLM.

**Q: Can I use this with local LLMs?**  
A: Yes! Configure `LOCAL_LLM_URL` in .env

---

## 🔗 Related Documentation

- `CHAT_IMPLEMENTATION_GUIDE.md` - Full implementation guide
- `CHAT_SYSTEM_COMPLETE.md` - System overview
- `test_streaming_endpoints.py` - Test examples
- `modules/advanced_chat_system.py` - Source code

---

**Status:** ✅ IMPLEMENTED  
**Last Updated:** November 20, 2025  
**All Tests:** READY TO RUN
