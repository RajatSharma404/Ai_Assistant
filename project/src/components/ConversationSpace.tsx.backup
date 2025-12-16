import { useState, useEffect, useRef } from 'react';
import { Send, Smile, Paperclip, MoreVertical } from 'lucide-react';
import { io, Socket } from 'socket.io-client';

interface Message {
  id: number;
  text: string;
  sender: 'user' | 'ai';
  timestamp: string;
  mood?: string;
}

const ConversationSpace = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [selectedModel, setSelectedModel] = useState<'openai' | 'gemini'>('openai');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const quickReplies = [
    '🌤️ Check weather',
    '🎵 Play music', 
    '💻 System status',
    '📝 Take a note',
    '🚀 Open Chrome',
    '🔍 Search Google',
    '⚙️ Show help'
  ];

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 128) + 'px';
    }
  }, [inputText]);

  // Initialize WebSocket connection
  useEffect(() => {
    const newSocket = io(window.location.origin, {
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5,
    });

    newSocket.on('connect', () => {
      console.log('Connected to WebSocket');
      setIsConnected(true);
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from WebSocket');
      setIsConnected(false);
    });

    newSocket.on('command_response', (data: { response?: string; error?: string; success?: boolean }) => {
      const responseText = data.error || data.response || 'No response received';
      const mood = data.error ? 'confused' : 'thoughtful';
      
      const aiResponse: Message = {
        id: Date.now(),
        text: responseText,
        sender: 'ai',
        timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        mood,
      };
      setMessages((prev) => [...prev, aiResponse]);
      setIsProcessing(false);
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleQuickReply = async (reply: string) => {
    if (isProcessing) return;
    
    // Add user message immediately
    const userMessage: Message = {
      id: Date.now(),
      text: reply,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
    };
    
    setMessages((prev) => [...prev, userMessage]);
    setIsProcessing(true);

    try {
      // Send via WebSocket if connected
      if (socket && isConnected) {
        socket.emit('command', { message: reply, model: selectedModel });
      } else {
        // Fallback to REST API
        const response = await fetch('/api/command', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ command: reply, model: selectedModel }),
        });
        
        if (response.ok) {
          const data = await response.json();
          if (data.success) {
            const aiResponse: Message = {
              id: Date.now() + 1,
              text: data.response,
              sender: 'ai',
              timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
              mood: 'thoughtful',
            };
            setMessages((prev) => [...prev, aiResponse]);
          } else {
            throw new Error(data.error || 'Unknown error occurred');
          }
        } else {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
        }
        setIsProcessing(false);
      }
    } catch (error) {
      console.error('Error sending quick reply:', error);
      const errorResponse: Message = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error processing your request. Please try again.',
        sender: 'ai',
        timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        mood: 'confused',
      };
      setMessages((prev) => [...prev, errorResponse]);
      setIsProcessing(false);
    }
  };

  const handleSend = async () => {
    if (!inputText.trim() || isProcessing) return;

    const userMessage: Message = {
      id: Date.now(),
      text: inputText,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
    };
    
    setMessages((prev) => [...prev, userMessage]);
    const commandText = inputText;
    setInputText('');
    setIsProcessing(true);

    try {
      // Send via WebSocket if connected
      if (socket && isConnected) {
        socket.emit('command', { message: commandText, model: selectedModel });
      } else {
        // Fallback to REST API
        const response = await fetch('/api/command', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ command: commandText, model: selectedModel }),
        });
        
        if (response.ok) {
          const data = await response.json();
          if (data.success) {
            const aiResponse: Message = {
              id: Date.now() + 1,
              text: data.response,
              sender: 'ai',
              timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
              mood: 'thoughtful',
            };
            setMessages((prev) => [...prev, aiResponse]);
          } else {
            throw new Error(data.error || 'Unknown error occurred');
          }
        } else {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
        }
        setIsProcessing(false);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorResponse: Message = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error processing your request. Please try again.',
        sender: 'ai',
        timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        mood: 'confused',
      };
      setMessages((prev) => [...prev, errorResponse]);
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen py-8 animate-fade-in">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2 text-gradient">Conversation Space</h1>
        <p className="text-[#DDDDDD]">Chat with your AI assistant</p>
      </div>

      <div className="max-w-4xl mx-auto">
        <div className="glass-strong rounded-2xl overflow-hidden flex flex-col conversation-container">
          <div className="flex items-center justify-between p-4 border-b border-white/10">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#00CEC9] to-[#6C5CE7] flex items-center justify-center">
                YD
              </div>
              <div>
                <div className="font-semibold">YourDaddy Assistant</div>
                <div className="text-xs text-[#00B894] flex items-center gap-1">
                  <div className="w-2 h-2 bg-[#00B894] rounded-full"></div>
                  Online
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <select 
                title="Select AI Model"
                value={selectedModel} 
                onChange={(e) => setSelectedModel(e.target.value as 'openai' | 'gemini')}
                className="bg-white/10 border border-white/20 rounded-lg px-3 py-1.5 text-sm outline-none focus:ring-2 focus:ring-[#00CEC9] text-white cursor-pointer hover:bg-white/20 transition-colors"
              >
                <option value="openai" className="bg-[#1a1a1a] text-white">ChatGPT (OpenAI)</option>
                <option value="gemini" className="bg-[#1a1a1a] text-white">Gemini (Google)</option>
              </select>
              <button className="w-8 h-8 rounded-lg hover:bg-white/10 flex items-center justify-center transition-colors" title="More options">
                <MoreVertical size={20} />
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-6 space-y-4 scrollbar-custom">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center py-12">
                <div className="w-20 h-20 rounded-full bg-gradient-to-br from-[#00CEC9] to-[#6C5CE7] flex items-center justify-center mb-4 text-3xl font-bold">
                  🤖
                </div>
                <h3 className="text-xl font-semibold mb-2">Start a Conversation</h3>
                <p className="text-[#DDDDDD] max-w-md mb-4">
                  I'm your intelligent AI assistant! I can help you with:
                </p>
                <div className="text-sm text-[#00B894] grid grid-cols-2 gap-2 max-w-md">
                  <div>🌤️ Weather updates</div>
                  <div>🚀 App management</div>
                  <div>🎵 Music control</div>
                  <div>💻 System monitoring</div>
                  <div>📝 Note taking</div>
                  <div>🔍 Web searches</div>
                </div>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}
                >
                  <div
                    className={`max-w-[70%] ${
                      message.sender === 'user'
                        ? 'bg-gradient-to-br from-[#00CEC9] to-[#6C5CE7] rounded-2xl rounded-tr-sm'
                        : 'glass rounded-2xl rounded-tl-sm'
                    } p-4`}
                  >
                    <p className="text-white mb-1">{message.text}</p>
                    <div className="flex items-center justify-between gap-4">
                      <span className="text-xs text-white/70">{message.timestamp}</span>
                      {message.mood && (
                        <span className="text-xs px-2 py-1 rounded-full bg-white/20">{message.mood}</span>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
            {isProcessing && (
              <div className="flex justify-start animate-fade-in">
                <div className="glass rounded-2xl rounded-tl-sm p-4">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-[#00CEC9] rounded-full animate-pulse"></div>
                    <div className="w-2 h-2 bg-[#00CEC9] rounded-full animate-pulse delay-100"></div>
                    <div className="w-2 h-2 bg-[#00CEC9] rounded-full animate-pulse delay-200"></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="p-4 border-t border-white/10">
            <div className="flex flex-wrap gap-2 mb-3">
              {quickReplies.map((reply, index) => (
                <button
                  key={index}
                  onClick={() => handleQuickReply(reply)}
                  disabled={isProcessing}
                  className="px-4 py-2 rounded-xl glass hover:bg-white/10 transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {reply}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-3">
              <button className="w-10 h-10 rounded-xl hover:bg-white/10 flex items-center justify-center transition-colors" title="Attach file">
                <Paperclip size={20} className="text-[#DDDDDD]" />
              </button>
              <button className="w-10 h-10 rounded-xl hover:bg-white/10 flex items-center justify-center transition-colors" title="Add emoji">
                <Smile size={20} className="text-[#DDDDDD]" />
              </button>
              <textarea
                ref={textareaRef}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyPress}
                disabled={isProcessing}
                placeholder={isProcessing ? "Processing..." : "Type your message... (Enter to send, Shift+Enter for new line)"}
                className="flex-1 bg-white/5 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-[#00CEC9] transition-all disabled:opacity-50 resize-none min-h-[48px] max-h-32"
                rows={1}
              />
              <button
                onClick={handleSend}
                disabled={isProcessing || !inputText.trim()}
                className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#00CEC9] to-[#6C5CE7] flex items-center justify-center hover:scale-105 transition-transform disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
                title="Send message"
              >
                {isProcessing ? (
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                ) : (
                  <Send size={20} />
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConversationSpace;
