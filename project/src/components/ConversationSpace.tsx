import { useState, useEffect, useRef } from 'react';
import { Send, Smile, Paperclip, MoreVertical, Copy, Check, RefreshCw, Edit2, StopCircle, Sparkles, ChevronDown, Trash2 } from 'lucide-react';
import { io, Socket } from 'socket.io-client';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import 'highlight.js/styles/github-dark.css';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: string;
  isStreaming?: boolean;
  suggestions?: string[];
}

interface ChatSession {
  id: string;
  title: string;
  timestamp: string;
  preview: string;
}

const ConversationSpace = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [selectedModel, setSelectedModel] = useState<'openai' | 'gemini'>('openai');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [showSessions, setShowSessions] = useState(false);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const inputContainerRef = useRef<HTMLDivElement>(null);

  const conversationStarters = [
    { icon: '✨', text: 'What can you help me with?', category: 'general' },
    { icon: '🌤️', text: "What's the weather like today?", category: 'weather' },
    { icon: '💻', text: 'Show me system information', category: 'system' },
    { icon: '🎵', text: 'Play some music for me', category: 'media' },
    { icon: '📝', text: 'Create a new note', category: 'productivity' },
    { icon: '🔍', text: 'Search for something online', category: 'search' },
  ];

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 200) + 'px';
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
      
      // Get the last user message for context
      const lastUserMsg = messages.filter(m => m.sender === 'user').pop();
      
      const aiResponse: Message = {
        id: `ai-${Date.now()}`,
        text: responseText,
        sender: 'ai',
        timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        suggestions: generateSmartSuggestions(responseText, lastUserMsg?.text),
      };
      setMessages((prev) => [...prev, aiResponse]);
      setIsProcessing(false);
    });

    // Handle streaming responses
    newSocket.on('chat_stream_token', (data: { token: string; done?: boolean }) => {
      if (data.done) {
        setMessages((prev) => {
          const updated = [...prev];
          const lastMsg = updated[updated.length - 1];
          const lastUserMsg = updated.filter(m => m.sender === 'user').pop();
          if (lastMsg?.sender === 'ai') {
            lastMsg.isStreaming = false;
            lastMsg.suggestions = generateSmartSuggestions(lastMsg.text, lastUserMsg?.text);
          }
          return updated;
        });
        setIsProcessing(false);
      } else {
        setMessages((prev) => {
          const updated = [...prev];
          const lastMsg = updated[updated.length - 1];
          if (lastMsg?.sender === 'ai' && lastMsg.isStreaming) {
            lastMsg.text += data.token;
          }
          return updated;
        });
      }
    });

    setSocket(newSocket);

    // Load sessions from localStorage
    const savedSessions = localStorage.getItem('chatSessions');
    if (savedSessions) {
      setSessions(JSON.parse(savedSessions));
    }

    // Create or load current session
    const sessionId = `session-${Date.now()}`;
    setCurrentSessionId(sessionId);

    return () => {
      newSocket.close();
    };
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Save session when messages change
  useEffect(() => {
    if (messages.length > 0) {
      saveCurrentSession();
    }
  }, [messages]);

  const generateSmartSuggestions = (text: string, userQuery?: string): string[] => {
    const suggestions: string[] = [];
    const lowerText = text.toLowerCase();
    const lowerQuery = userQuery?.toLowerCase() || '';

    // Only generate suggestions for substantial responses
    if (text.length < 50) return [];

    // Avoid generic suggestions for error messages
    if (lowerText.includes('error') || lowerText.includes('sorry') || lowerText.includes('failed')) {
      return ['Try again', 'Check system status'];
    }

    // Context-aware suggestions based on response content
    if (lowerText.includes('weather') || lowerText.includes('temperature') || lowerText.includes('forecast')) {
      suggestions.push('Show 5-day forecast', 'Weather alerts?');
    } else if (lowerText.includes('music') || lowerText.includes('song') || lowerText.includes('playing')) {
      suggestions.push('Next song', 'Show queue');
    } else if (lowerText.includes('system') || lowerText.includes('cpu') || lowerText.includes('memory') || lowerText.includes('disk')) {
      suggestions.push('Running processes', 'Network status');
    } else if (lowerText.includes('application') || lowerText.includes('app') || lowerText.includes('program')) {
      suggestions.push('List all apps', 'Search for app');
    } else if (lowerText.includes('file') || lowerText.includes('folder') || lowerText.includes('directory')) {
      suggestions.push('Show recent files', 'Search files');
    } else if (lowerText.includes('note') || lowerText.includes('reminder') || lowerText.includes('task')) {
      suggestions.push('Show my notes', 'Create reminder');
    } else if (lowerText.includes('search') || lowerText.includes('google') || lowerText.includes('find')) {
      suggestions.push('Search something else', 'Open browser');
    } else {
      // Only show generic suggestions for first few responses
      if (messages.length <= 4) {
        suggestions.push('What else can you do?');
      }
      return suggestions.slice(0, 1);
    }

    return suggestions.slice(0, 2); // Limit to 2 specific suggestions
  };

  const saveCurrentSession = () => {
    if (messages.length === 0) return;

    const session: ChatSession = {
      id: currentSessionId,
      title: messages[0]?.text.slice(0, 50) + (messages[0]?.text.length > 50 ? '...' : ''),
      timestamp: new Date().toISOString(),
      preview: messages[messages.length - 1]?.text.slice(0, 100),
    };

    const updatedSessions = [session, ...sessions.filter(s => s.id !== currentSessionId)].slice(0, 20);
    setSessions(updatedSessions);
    localStorage.setItem('chatSessions', JSON.stringify(updatedSessions));
    localStorage.setItem(`chat-${currentSessionId}`, JSON.stringify(messages));
  };

  const loadSession = (sessionId: string) => {
    const savedMessages = localStorage.getItem(`chat-${sessionId}`);
    if (savedMessages) {
      setMessages(JSON.parse(savedMessages));
      setCurrentSessionId(sessionId);
      setShowSessions(false);
    }
  };

  const deleteSession = (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const updatedSessions = sessions.filter(s => s.id !== sessionId);
    setSessions(updatedSessions);
    localStorage.setItem('chatSessions', JSON.stringify(updatedSessions));
    localStorage.removeItem(`chat-${sessionId}`);
    
    if (sessionId === currentSessionId) {
      setMessages([]);
      const newSessionId = `session-${Date.now()}`;
      setCurrentSessionId(newSessionId);
    }
  };

  const startNewChat = () => {
    if (messages.length > 0) {
      saveCurrentSession();
    }
    setMessages([]);
    const newSessionId = `session-${Date.now()}`;
    setCurrentSessionId(newSessionId);
    setShowSessions(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleCopy = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const handleRegenerate = async (messageId: string) => {
    const messageIndex = messages.findIndex(m => m.id === messageId);
    if (messageIndex === -1 || messageIndex === 0) return;

    const previousUserMessage = messages[messageIndex - 1];
    if (previousUserMessage.sender !== 'user') return;

    // Remove the old AI response
    setMessages(prev => prev.filter(m => m.id !== messageId));
    
    // Resend the user message
    await sendMessage(previousUserMessage.text);
  };

  const handleEdit = (messageId: string) => {
    const message = messages.find(m => m.id === messageId);
    if (message && message.sender === 'user') {
      setEditingId(messageId);
      setInputText(message.text);
      textareaRef.current?.focus();
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInputText(suggestion);
    textareaRef.current?.focus();
    // Send directly with the suggestion text instead of waiting for state update
    setTimeout(() => sendMessage(suggestion), 50);
  };

  const sendMessage = async (text: string) => {
    if (!text.trim() || isProcessing) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      text: text,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
    };
    
    if (editingId) {
      // Replace the edited message and remove all subsequent messages
      const editIndex = messages.findIndex(m => m.id === editingId);
      setMessages(prev => [...prev.slice(0, editIndex), userMessage]);
      setEditingId(null);
    } else {
      setMessages(prev => [...prev, userMessage]);
    }

    setIsProcessing(true);

    // Add streaming placeholder
    const streamingMessage: Message = {
      id: `ai-streaming-${Date.now()}`,
      text: '',
      sender: 'ai',
      timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
      isStreaming: true,
    };
    setMessages(prev => [...prev, streamingMessage]);

    try {
      if (socket && isConnected) {
        socket.emit('command', { message: text, model: selectedModel });
      } else {
        // Fallback to REST API
        const response = await fetch('/api/command', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ command: text, model: selectedModel }),
        });
        
        if (response.ok) {
          const data = await response.json();
          setMessages(prev => {
            const updated = prev.filter(m => !m.isStreaming);
            return [...updated, {
              id: `ai-${Date.now()}`,
              text: data.response || data.error || 'No response',
              sender: 'ai',
              timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
              suggestions: generateSmartSuggestions(data.response || '', text),
            }];
          });
        } else {
          throw new Error('Network response was not ok');
        }
        setIsProcessing(false);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => {
        const updated = prev.filter(m => !m.isStreaming);
        return [...updated, {
          id: `ai-error-${Date.now()}`,
          text: 'Sorry, I encountered an error processing your request. Please try again.',
          sender: 'ai',
          timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        }];
      });
      setIsProcessing(false);
    }
  };

  const handleSend = async () => {
    const text = inputText;
    setInputText('');
    await sendMessage(text);
  };

  const stopGeneration = () => {
    setIsProcessing(false);
    setMessages(prev => {
      const updated = [...prev];
      const lastMsg = updated[updated.length - 1];
      if (lastMsg?.isStreaming) {
        lastMsg.isStreaming = false;
        lastMsg.text += ' [Generation stopped]';
      }
      return updated;
    });
  };

  return (
    <div className="min-h-screen py-8 animate-fade-in">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold mb-2 text-gradient">Chat Space</h1>
          <p className="text-[#DDDDDD]">Powered by advanced AI models</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={startNewChat}
            className="px-4 py-2 rounded-xl glass hover:bg-white/10 transition-colors flex items-center gap-2"
          >
            <Sparkles size={18} />
            New Chat
          </button>
          <button
            onClick={() => setShowSessions(!showSessions)}
            className="px-4 py-2 rounded-xl glass hover:bg-white/10 transition-colors flex items-center gap-2"
          >
            <ChevronDown size={18} className={`transition-transform ${showSessions ? 'rotate-180' : ''}`} />
            History
          </button>
        </div>
      </div>

      {showSessions && (
        <div className="mb-6 glass-strong rounded-2xl p-4 max-h-64 overflow-y-auto scrollbar-custom">
          <h3 className="text-lg font-semibold mb-3">Recent Conversations</h3>
          {sessions.length === 0 ? (
            <p className="text-[#AAAAAA] text-sm text-center py-4">No conversation history yet</p>
          ) : (
            <div className="space-y-2">
              {sessions.map(session => (
                <div
                  key={session.id}
                  onClick={() => loadSession(session.id)}
                  className="p-3 rounded-xl glass hover:bg-white/10 cursor-pointer transition-colors group"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="font-medium mb-1">{session.title}</div>
                      <div className="text-xs text-[#AAAAAA]">
                        {new Date(session.timestamp).toLocaleDateString()} • {session.preview}
                      </div>
                    </div>
                    <button
                      onClick={(e) => deleteSession(session.id, e)}
                      className="opacity-0 group-hover:opacity-100 p-2 hover:bg-red-500/20 rounded-lg transition-all"
                      title="Delete session"
                    >
                      <Trash2 size={16} className="text-red-400" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="max-w-5xl mx-auto">
        <div className="glass-strong rounded-2xl overflow-hidden flex flex-col" style={{ height: 'calc(100vh - 250px)', minHeight: '500px' }}>
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-white/10 bg-white/5">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#00CEC9] to-[#6C5CE7] flex items-center justify-center font-bold">
                AI
              </div>
              <div>
                <div className="font-semibold">YourDaddy Assistant</div>
                <div className="text-xs text-[#00B894] flex items-center gap-1">
                  <div className="w-2 h-2 bg-[#00B894] rounded-full animate-pulse"></div>
                  {isConnected ? 'Connected' : 'Reconnecting...'}
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
                <option value="openai" className="bg-[#1a1a1a] text-white">ChatGPT</option>
                <option value="gemini" className="bg-[#1a1a1a] text-white">Gemini</option>
              </select>
              <button className="w-8 h-8 rounded-lg hover:bg-white/10 flex items-center justify-center transition-colors" title="More options">
                <MoreVertical size={20} />
              </button>
            </div>
          </div>

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-custom">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <div className="w-24 h-24 rounded-full bg-gradient-to-br from-[#00CEC9] to-[#6C5CE7] flex items-center justify-center mb-6 text-4xl animate-pulse">
                  🤖
                </div>
                <h3 className="text-2xl font-semibold mb-3">How can I help you today?</h3>
                <p className="text-[#AAAAAA] max-w-md mb-8">
                  I'm your intelligent AI assistant. Ask me anything or try one of these:
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl w-full">
                  {conversationStarters.map((starter, index) => (
                    <button
                      key={index}
                      onClick={() => handleSuggestionClick(starter.text)}
                      className="p-4 rounded-xl glass hover:bg-white/10 transition-all text-left group hover:scale-[1.02]"
                    >
                      <div className="flex items-center gap-3">
                        <div className="text-2xl">{starter.icon}</div>
                        <div className="flex-1">
                          <div className="font-medium group-hover:text-[#00CEC9] transition-colors">
                            {starter.text}
                          </div>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((message, index) => (
                <div
                  key={message.id}
                  className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in group`}
                >
                  <div className={`max-w-[85%] ${message.sender === 'user' ? 'order-2' : 'order-1'}`}>
                    <div
                      className={`${
                        message.sender === 'user'
                          ? 'bg-gradient-to-br from-[#00CEC9] to-[#6C5CE7] rounded-2xl rounded-tr-md'
                          : 'glass rounded-2xl rounded-tl-md'
                      } p-4 ${message.isStreaming ? 'animate-pulse' : ''}`}
                    >
                      {message.sender === 'user' ? (
                        <p className="text-white whitespace-pre-wrap">{message.text}</p>
                      ) : (
                        <div className="prose prose-invert max-w-none">
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm, remarkBreaks]}
                            rehypePlugins={[rehypeHighlight, rehypeRaw]}
                            components={{
                              code: ({ node, inline, className, children, ...props }: any) => {
                                const match = /language-(\w+)/.exec(className || '');
                                return !inline ? (
                                  <div className="relative group/code">
                                    <div className="absolute right-2 top-2 opacity-0 group-hover/code:opacity-100 transition-opacity">
                                      <button
                                        onClick={() => handleCopy(String(children), `code-${index}`)}
                                        className="p-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
                                        title="Copy code"
                                      >
                                        {copiedId === `code-${index}` ? (
                                          <Check size={14} className="text-green-400" />
                                        ) : (
                                          <Copy size={14} />
                                        )}
                                      </button>
                                    </div>
                                    <code className={className} {...props}>
                                      {children}
                                    </code>
                                  </div>
                                ) : (
                                  <code className={`${className} bg-white/10 px-1.5 py-0.5 rounded`} {...props}>
                                    {children}
                                  </code>
                                );
                              },
                            }}
                          >
                            {message.text}
                          </ReactMarkdown>
                        </div>
                      )}
                      
                      <div className="flex items-center justify-between mt-2 pt-2 border-t border-white/10">
                        <span className="text-xs text-white/60">{message.timestamp}</span>
                        {message.sender === 'ai' && !message.isStreaming && (
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => handleCopy(message.text, message.id)}
                              className="p-1.5 hover:bg-white/10 rounded transition-colors opacity-0 group-hover:opacity-100"
                              title="Copy response"
                            >
                              {copiedId === message.id ? (
                                <Check size={14} className="text-green-400" />
                              ) : (
                                <Copy size={14} />
                              )}
                            </button>
                            <button
                              onClick={() => handleRegenerate(message.id)}
                              className="p-1.5 hover:bg-white/10 rounded transition-colors opacity-0 group-hover:opacity-100"
                              title="Regenerate response"
                            >
                              <RefreshCw size={14} />
                            </button>
                          </div>
                        )}
                        {message.sender === 'user' && (
                          <button
                            onClick={() => handleEdit(message.id)}
                            className="p-1.5 hover:bg-white/10 rounded transition-colors opacity-0 group-hover:opacity-100"
                            title="Edit message"
                          >
                            <Edit2 size={14} />
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Smart Suggestions - Only show for the latest AI message */}
                    {message.sender === 'ai' && message.suggestions && message.suggestions.length > 0 && !message.isStreaming && index === messages.length - 1 && (
                      <div className="flex flex-wrap gap-2 mt-3">
                        {message.suggestions.map((suggestion, idx) => (
                          <button
                            key={idx}
                            onClick={() => handleSuggestionClick(suggestion)}
                            className="px-3 py-1.5 text-xs rounded-lg border border-white/20 hover:border-[#00CEC9] hover:bg-white/5 transition-all group/suggest"
                          >
                            <span className="opacity-70 group-hover/suggest:opacity-100 transition-opacity">{suggestion}</span>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            {isProcessing && messages[messages.length - 1]?.isStreaming && (
              <div className="flex justify-center">
                <button
                  onClick={stopGeneration}
                  className="px-4 py-2 rounded-xl glass hover:bg-red-500/20 transition-colors flex items-center gap-2 text-red-400"
                >
                  <StopCircle size={16} />
                  Stop generating
                </button>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="p-4 border-t border-white/10 bg-white/5" ref={inputContainerRef}>
            <div className="flex items-end gap-3">
              <button 
                className="w-10 h-10 rounded-xl hover:bg-white/10 flex items-center justify-center transition-colors flex-shrink-0" 
                title="Attach file"
              >
                <Paperclip size={20} className="text-[#DDDDDD]" />
              </button>
              
              <div className="flex-1 relative">
                <textarea
                  ref={textareaRef}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={handleKeyPress}
                  disabled={isProcessing}
                  placeholder={editingId ? "Edit your message..." : "Message YourDaddy Assistant..."}
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 pr-12 outline-none focus:ring-2 focus:ring-[#00CEC9] transition-all disabled:opacity-50 resize-none min-h-[52px] max-h-[200px] scrollbar-custom"
                  rows={1}
                />
                <div className="absolute right-3 bottom-3 text-xs text-white/40">
                  {inputText.length > 0 && `${inputText.length} characters`}
                </div>
              </div>

              <button
                onClick={handleSend}
                disabled={isProcessing || !inputText.trim()}
                className="w-12 h-12 rounded-xl bg-gradient-to-br from-[#00CEC9] to-[#6C5CE7] flex items-center justify-center hover:scale-105 transition-transform disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex-shrink-0"
                title="Send message"
              >
                {isProcessing ? (
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                ) : (
                  <Send size={20} />
                )}
              </button>
            </div>
            
            <div className="mt-2 text-xs text-white/40 text-center">
              Press <kbd className="px-2 py-0.5 bg-white/10 rounded">Enter</kbd> to send • <kbd className="px-2 py-0.5 bg-white/10 rounded">Shift + Enter</kbd> for new line
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConversationSpace;
