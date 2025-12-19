import { useState, useEffect, useRef } from 'react';
import { Mic, Volume2, Settings as SettingsIcon, MicOff, Loader, ChevronDown, ChevronUp, Play, RefreshCw, MessageSquare } from 'lucide-react';
import { io, Socket } from 'socket.io-client';

type VoiceState = 'idle' | 'wake_listening' | 'command_listening' | 'processing' | 'speaking';

interface CommandHistoryItem {
  id: string;
  timestamp: number;
  userText: string;
  assistantResponse: string;
  confidence?: number;
}

const VoiceInterface = () => {
  const [voiceState, setVoiceState] = useState<VoiceState>('idle');
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [waveform, setWaveform] = useState<number[]>(Array(20).fill(0.2));
  const [transcript, setTranscript] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [response, setResponse] = useState('');
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [commandHistory, setCommandHistory] = useState<CommandHistoryItem[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [wakeWordEnabled, setWakeWordEnabled] = useState(true);
  const [voiceFeedbackEnabled, setVoiceFeedbackEnabled] = useState(true);
  const [micSensitivity, setMicSensitivity] = useState(75);
  const [voiceSpeed, setVoiceSpeed] = useState(1.0);
  const [selectedLanguage, setSelectedLanguage] = useState('en-US');
  const [audioLevel, setAudioLevel] = useState(0);
  const [confidence, setConfidence] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    // Initialize WebSocket connection
    const socketInstance = io();
    setSocket(socketInstance);

    socketInstance.on('connect', () => {
      setIsConnected(true);
      console.log('Voice interface connected to backend');
      fetchCommandHistory();
    });

    socketInstance.on('disconnect', () => {
      setIsConnected(false);
    });

    socketInstance.on('voice_status', (data) => {
      setIsListening(data.listening);
    });

    socketInstance.on('voice_transcript', (data) => {
      setTranscript(data.text);
      setInterimTranscript(''); // Clear interim when final arrives
      setVoiceState('processing');
      setConfidence(data.confidence || 0.8);
    });
    
    socketInstance.on('voice_partial_transcript', (data) => {
      setInterimTranscript(data.text);
    });

    socketInstance.on('voice_response', (data) => {
      setResponse(data.response);
      setIsProcessing(false);
      setVoiceState('speaking');
      
      // Add to history
      const newItem: CommandHistoryItem = {
        id: Date.now().toString(),
        timestamp: Date.now(),
        userText: transcript,
        assistantResponse: data.response,
        confidence: confidence
      };
      setCommandHistory(prev => [newItem, ...prev.slice(0, 9)]);
      
      // Return to idle after speaking
      setTimeout(() => {
        if (wakeWordEnabled) {
          setVoiceState('wake_listening');
        } else {
          setVoiceState('idle');
        }
      }, 2000);
    });

    socketInstance.on('voice_start_response', (data) => {
      if (data.success) {
        console.log('Voice listening started successfully');
      } else {
        console.error('Failed to start voice listening:', data.error);
        setIsListening(false);
      }
    });

    socketInstance.on('voice_stop_response', (data) => {
      if (data.success) {
        console.log('Voice listening stopped successfully');
      }
    });

    return () => {
      socketInstance.disconnect();
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  const fetchCommandHistory = async () => {
    try {
      const response = await fetch('/api/voice/history');
      if (response.ok) {
        const history = await response.json();
        setCommandHistory(history);
      }
    } catch (error) {
      console.error('Failed to fetch command history:', error);
    }
  };

  // Real-time audio level monitoring
  useEffect(() => {
    let animationId: number;
    
    if (analyserRef.current && isListening) {
      const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
      
      const updateWaveform = () => {
        if (!analyserRef.current || !isListening) {
          setWaveform(Array(20).fill(0.2));
          setAudioLevel(0);
          return;
        }
        
        analyserRef.current.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
        const normalizedLevel = average / 255;
        setAudioLevel(normalizedLevel);
        
        // Update waveform based on frequency data
        const waveData = Array(20).fill(0).map((_, i) => {
          const start = Math.floor((i / 20) * dataArray.length);
          const end = Math.floor(((i + 1) / 20) * dataArray.length);
          const slice = dataArray.slice(start, end);
          const avg = slice.reduce((a, b) => a + b, 0) / slice.length;
          // More visible range: 0.3 to 1.0
          return Math.max(0.3, Math.min(1.0, (avg / 255) * 1.2));
        });
        setWaveform(waveData);
        
        animationId = requestAnimationFrame(updateWaveform);
      };
      
      animationId = requestAnimationFrame(updateWaveform);
    } else {
      setWaveform(Array(20).fill(0.2));
      setAudioLevel(0);
    }
    
    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [isListening]);

  const toggleListening = async () => {
    if (!isConnected || !socket) {
      setTranscript('Not connected to server');
      return;
    }

    if (isListening) {
      // Stop listening
      socket.emit('stop_voice_listening');
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
      if (recognitionRef.current) {
        recognitionRef.current.stop();
        recognitionRef.current = null;
      }
      analyserRef.current = null;
      setIsListening(false);
      setVoiceState('idle');
      setInterimTranscript('');
    } else {
      // Start listening
      setTranscript('Initializing voice recognition...');
      setResponse('');
      socket.emit('start_voice_listening');
      
      try {
        // Also start browser-based recording as fallback
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;
        audioChunksRef.current = [];
        
        // Setup audio analysis
        audioContextRef.current = new AudioContext();
        const source = audioContextRef.current.createMediaStreamSource(stream);
        analyserRef.current = audioContextRef.current.createAnalyser();
        analyserRef.current.fftSize = 256;
        source.connect(analyserRef.current);

        mediaRecorder.ondataavailable = (event) => {
          audioChunksRef.current.push(event.data);
        };

        mediaRecorder.onstop = () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
          // Convert to base64 and send to server if needed
          const reader = new FileReader();
          reader.onloadend = () => {
            if (socket && reader.result) {
              const base64Audio = (reader.result as string).split(',')[1];
              socket.emit('voice_audio_data', { audio_data: base64Audio });
            }
          };
          reader.readAsDataURL(audioBlob);
        };

        mediaRecorder.start();
        setIsListening(true);
        setVoiceState(wakeWordEnabled ? 'wake_listening' : 'command_listening');
        setTranscript('');
        setInterimTranscript('');
        
        // Start Web Speech API for real-time transcription
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
          const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
          const recognition = new SpeechRecognition();
          recognitionRef.current = recognition;
          
          recognition.continuous = true;
          recognition.interimResults = true;
          recognition.lang = selectedLanguage;
          recognition.maxAlternatives = 1;
          
          recognition.onresult = (event: any) => {
            let interim = '';
            let final = '';
            
            for (let i = event.resultIndex; i < event.results.length; i++) {
              const transcript = event.results[i][0].transcript;
              if (event.results[i].isFinal) {
                final += transcript + ' ';
              } else {
                interim += transcript;
              }
            }
            
            if (interim) {
              setInterimTranscript(interim);
            }
            if (final) {
              setTranscript(prev => prev + final);
              setInterimTranscript('');
            }
          };
          
          recognition.onerror = (event: any) => {
            console.error('Speech recognition error:', event.error);
          };
          
          recognition.onend = () => {
            if (isListening) {
              // Restart if still listening
              try {
                recognition.start();
              } catch (e) {
                console.log('Recognition restart failed:', e);
              }
            }
          };
          
          try {
            recognition.start();
            console.log('✅ Live speech recognition started');
          } catch (e) {
            console.error('Failed to start speech recognition:', e);
          }
        }
        
        console.log('✅ Audio analysis setup complete - waveform should be active now');
      } catch (error) {
        console.error('Failed to access microphone:', error);
        setTranscript('Microphone access denied. Please enable microphone permissions.');
        setVoiceState('idle');
      }
    }
  };

  const speakText = (text: string) => {
    if (socket) {
      socket.emit('request_tts', { text });
    }
  };

  const getStateInfo = () => {
    switch (voiceState) {
      case 'wake_listening':
        return { text: "Say 'Hey Assistant' to start", color: 'from-blue-500 to-cyan-500', icon: Mic };
      case 'command_listening':
        return { text: 'Listening for command...', color: 'from-green-500 to-emerald-500', icon: Volume2 };
      case 'processing':
        return { text: 'Processing...', color: 'from-purple-500 to-pink-500', icon: Loader };
      case 'speaking':
        return { text: 'Speaking...', color: 'from-cyan-500 to-blue-500', icon: MessageSquare };
      default:
        return { text: 'Click to start', color: 'from-[#6C5CE7] to-[#00CEC9]', icon: Mic };
    }
  };

  const stateInfo = getStateInfo();
  const StateIcon = stateInfo.icon;

  return (
    <div className="min-h-screen py-8 animate-fade-in flex flex-col items-center justify-center relative">
      {/* Floating Action Buttons */}
      <div className="fixed top-8 right-8 z-50 flex gap-3">
        <button
          onClick={() => setShowHistory(!showHistory)}
          className="w-12 h-12 rounded-full bg-gradient-to-br from-[#00CEC9] to-[#6C5CE7] flex items-center justify-center shadow-lg hover:shadow-2xl hover:scale-110 transition-all duration-300 relative"
          title="Command History"
        >
          <MessageSquare size={20} className="text-white" />
          {commandHistory.length > 0 && (
            <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full text-xs flex items-center justify-center text-white font-bold">
              {commandHistory.length}
            </span>
          )}
        </button>
        
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="w-12 h-12 rounded-full bg-gradient-to-br from-[#6C5CE7] to-[#00CEC9] flex items-center justify-center shadow-lg hover:shadow-2xl hover:scale-110 transition-all duration-300"
          title="Voice Settings"
        >
          <SettingsIcon size={20} className={`text-white transition-transform duration-300 ${
            showSettings ? 'rotate-180' : 'rotate-0'
          }`} />
        </button>
      </div>

      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-2 text-gradient">Voice Interface</h1>
        <p className="text-[#DDDDDD]">Speak naturally, I'm listening</p>
      </div>

      <div className="relative mb-16">
        <div className="relative w-80 h-80 flex items-center justify-center">
          {voiceState !== 'idle' && (
            <>
              <div className={`absolute w-80 h-80 rounded-full border-2 bg-gradient-to-r ${stateInfo.color} opacity-10 animate-ping`}></div>
              <div className={`absolute w-96 h-96 rounded-full border bg-gradient-to-r ${stateInfo.color} opacity-5 animate-pulse`}></div>
            </>
          )}
          
          {/* Active listening indicator - bright ring when audio detected */}
          {isListening && audioLevel > 0.1 && (
            <div 
              className={`absolute rounded-full border-4 bg-gradient-to-r ${stateInfo.color} transition-all duration-100`}
              style={{
                width: `${240 + audioLevel * 100}px`,
                height: `${240 + audioLevel * 100}px`,
                opacity: 0.3 + audioLevel * 0.4
              }}
            ></div>
          )}

          <div
            className={`relative w-48 h-48 rounded-full flex items-center justify-center cursor-pointer transition-all duration-300 bg-gradient-to-br ${stateInfo.color} ${
              voiceState !== 'idle' ? 'scale-110 shadow-2xl shadow-[#00CEC9]/50' : 'hover:scale-105'
            } ${!isConnected ? 'opacity-50 cursor-not-allowed' : ''}`}
            onClick={toggleListening}
          >
            <div className="w-44 h-44 rounded-full bg-[#0A0E27] flex items-center justify-center flex-col gap-2">
              {voiceState === 'processing' ? (
                <Loader size={64} className="text-[#00CEC9] animate-spin" />
              ) : !isConnected ? (
                <MicOff size={64} className="text-gray-500" />
              ) : (
                <StateIcon size={64} className={`${voiceState !== 'idle' ? 'animate-pulse' : ''}`} style={{
                  color: voiceState === 'wake_listening' ? '#00CEC9' : 
                         voiceState === 'command_listening' ? '#10b981' :
                         voiceState === 'speaking' ? '#3b82f6' : '#6C5CE7'
                }} />
              )}
              {confidence > 0 && voiceState === 'processing' && (
                <div className="flex gap-1">
                  {Array(5).fill(0).map((_, i) => (
                    <div key={i} className={`w-2 h-2 rounded-full ${i < confidence * 5 ? 'bg-[#00CEC9]' : 'bg-gray-600'}`}></div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="absolute -bottom-16 left-1/2 -translate-x-1/2 w-full">
            <div className="flex items-end justify-center gap-1 h-20">
              {waveform.map((height, index) => (
                <div
                  key={index}
                  className={`w-2 rounded-full transition-all duration-75 bg-gradient-to-t ${stateInfo.color}`}
                  style={{ 
                    height: `${height * 100}%`,
                    opacity: voiceState !== 'idle' ? 1 : 0.3
                  }}
                ></div>
              ))}
            </div>
            {/* Microphone active indicator */}
            {isListening && audioLevel === 0 && (
              <div className="absolute -bottom-6 left-1/2 -translate-x-1/2 text-xs text-[#00CEC9] animate-pulse">
                🎤 Microphone active - speak now
              </div>
            )}
          </div>
        </div>
        
        {/* State text */}
        <div className="text-center mt-8">
          <p className="text-xl text-white font-medium">{stateInfo.text}</p>
          {isListening && (
            <div className="mt-2 flex items-center justify-center gap-2">
              <span className="text-sm text-[#DDDDDD]">Input Level:</span>
              <div className="w-32 h-2 bg-white/10 rounded-full overflow-hidden">
                <div 
                  className={`h-full bg-gradient-to-r ${stateInfo.color} transition-all duration-100`}
                  style={{ width: `${Math.max(5, audioLevel * 100)}%` }}
                ></div>
              </div>
              <span className="text-xs text-[#DDDDDD]">{Math.round(audioLevel * 100)}%</span>
            </div>
          )}
        </div>
      </div>

      {/* Live Transcription Display - Always visible when listening */}
      {isListening && (transcript || interimTranscript) && (
        <div className="glass-strong p-6 rounded-2xl mb-4 max-w-2xl w-full min-h-[100px]">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-[#DDDDDD]">Listening...</span>
          </div>
          <div className="text-xl">
            {transcript && <span className="text-white">{transcript}</span>}
            {interimTranscript && (
              <span className="text-[#00CEC9] opacity-70 italic">{interimTranscript}</span>
            )}
          </div>
        </div>
      )}
      
      {/* Final transcript after processing */}
      {!isListening && transcript && (
        <div className="glass-strong p-6 rounded-2xl mb-4 max-w-2xl w-full">
          <div className="text-sm text-[#DDDDDD] mb-2">You said:</div>
          <p className="text-xl text-[#00CEC9]">{transcript}</p>
        </div>
      )}

      {response && (
        <div className="glass-strong p-6 rounded-2xl mb-8 max-w-2xl w-full">
          <div className="flex items-center justify-between mb-2">
            <div className="text-sm text-[#DDDDDD]">Assistant:</div>
            <button
              onClick={() => speakText(response)}
              className="text-[#6C5CE7] hover:text-[#00CEC9] transition-colors"
              title="Speak response"
            >
              <Volume2 size={16} />
            </button>
          </div>
          <p className="text-lg text-white">{response}</p>
        </div>
      )}

      {!isConnected && (
        <div className="glass-strong p-4 rounded-xl mb-8 max-w-2xl w-full text-center">
          <p className="text-yellow-400">⚠️ Not connected to voice server</p>
        </div>
      )}

      {/* Enhanced Command History - Modal */}
      {showHistory && (
        <div className="fixed inset-0 z-40 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-fade-in" onClick={() => setShowHistory(false)}>
          <div className="glass-strong rounded-2xl overflow-hidden max-w-2xl w-full max-h-[80vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between p-6 border-b border-white/10 sticky top-0 bg-[#0A0E27]/95 backdrop-blur-sm z-10">
              <h2 className="text-xl font-bold">Recent Conversations</h2>
              <button
                onClick={() => setShowHistory(false)}
                className="text-[#DDDDDD] hover:text-white transition-colors p-2 hover:bg-white/10 rounded-lg"
              >
                ✕
              </button>
            </div>
            
            <div className="p-6 space-y-4">
              {commandHistory.length === 0 ? (
                <p className="text-[#DDDDDD] text-center py-8">No commands yet. Start speaking!</p>
              ) : (
                commandHistory.map((item) => (
                  <div key={item.id} className="space-y-2 p-4 rounded-xl bg-white/5 hover:bg-white/10 transition-colors">
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#00CEC9] to-[#6C5CE7] flex items-center justify-center flex-shrink-0 mt-1">
                        <Mic size={16} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs text-[#DDDDDD]/70">
                            {new Date(item.timestamp).toLocaleTimeString()}
                          </span>
                          {item.confidence && (
                            <span className="text-xs text-[#00CEC9]">
                              {Math.round(item.confidence * 100)}% confidence
                            </span>
                          )}
                        </div>
                        <p className="text-white font-medium mb-2">You: {item.userText}</p>
                        <p className="text-[#DDDDDD] text-sm">Assistant: {item.assistantResponse}</p>
                      </div>
                    </div>
                    <div className="flex gap-2 pl-11">
                      <button
                        onClick={() => speakText(item.assistantResponse)}
                        className="flex items-center gap-1 text-xs text-[#6C5CE7] hover:text-[#00CEC9] transition-colors"
                      >
                        <Play size={12} /> Replay
                      </button>
                      <button className="flex items-center gap-1 text-xs text-[#6C5CE7] hover:text-[#00CEC9] transition-colors">
                        <RefreshCw size={12} /> Repeat
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      {/* Enhanced Voice Settings - Modal */}
      {showSettings && (
        <div className="fixed inset-0 z-40 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-fade-in" onClick={() => setShowSettings(false)}>
          <div className="glass-strong rounded-2xl overflow-hidden max-w-md w-full max-h-[80vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between p-6 border-b border-white/10 sticky top-0 bg-[#0A0E27]/95 backdrop-blur-sm z-10">
              <h2 className="text-xl font-bold">Voice Settings</h2>
              <button
                onClick={() => setShowSettings(false)}
                className="text-[#DDDDDD] hover:text-white transition-colors p-2 hover:bg-white/10 rounded-lg"
              >
                ✕
              </button>
            </div>
            
            <div className="p-6 space-y-5">
              {/* Wake Word Detection */}
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-[#DDDDDD] block">Wake Word Detection</span>
                  <span className="text-xs text-[#DDDDDD]/70">Activate with "Hey Assistant"</span>
                </div>
                <div 
                  className={`w-12 h-6 rounded-full cursor-pointer relative transition-colors ${
                    wakeWordEnabled ? 'bg-gradient-to-r from-[#00CEC9] to-[#6C5CE7]' : 'bg-gray-600'
                  }`}
                  onClick={() => setWakeWordEnabled(!wakeWordEnabled)}
                >
                  <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${
                    wakeWordEnabled ? 'right-1' : 'left-1'
                  }`}></div>
                </div>
              </div>

              {/* Voice Feedback */}
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-[#DDDDDD] block">Voice Feedback</span>
                  <span className="text-xs text-[#DDDDDD]/70">Audio confirmation beeps</span>
                </div>
                <div 
                  className={`w-12 h-6 rounded-full cursor-pointer relative transition-colors ${
                    voiceFeedbackEnabled ? 'bg-gradient-to-r from-[#00CEC9] to-[#6C5CE7]' : 'bg-gray-600'
                  }`}
                  onClick={() => setVoiceFeedbackEnabled(!voiceFeedbackEnabled)}
                >
                  <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${
                    voiceFeedbackEnabled ? 'right-1' : 'left-1'
                  }`}></div>
                </div>
              </div>

              {/* Language Selector */}
              <div>
                <label className="text-[#DDDDDD] block mb-2">Language</label>
                <select 
                  value={selectedLanguage}
                  onChange={(e) => setSelectedLanguage(e.target.value)}
                  className="w-full bg-white/10 text-white rounded-lg px-4 py-2 border border-white/20 focus:border-[#00CEC9] focus:outline-none"
                >
                  <option value="en-US">English (US)</option>
                  <option value="en-GB">English (UK)</option>
                  <option value="hi-IN">Hindi</option>
                  <option value="es-ES">Spanish</option>
                  <option value="fr-FR">French</option>
                </select>
              </div>

              {/* Voice Speed */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[#DDDDDD]">Voice Speed</span>
                  <span className="text-[#00CEC9]">{voiceSpeed.toFixed(1)}x</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="2"
                  step="0.1"
                  value={voiceSpeed}
                  onChange={(e) => setVoiceSpeed(parseFloat(e.target.value))}
                  className="w-full accent-[#00CEC9]"
                />
                <div className="flex justify-between text-xs text-[#DDDDDD]/70 mt-1">
                  <span>Slower</span>
                  <span>Faster</span>
                </div>
              </div>

              {/* Microphone Sensitivity */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[#DDDDDD]">Microphone Sensitivity</span>
                  <span className="text-[#00CEC9]">{micSensitivity}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={micSensitivity}
                  onChange={(e) => setMicSensitivity(parseInt(e.target.value))}
                  className="w-full accent-[#00CEC9]"
                />
                {audioLevel > 0 && (
                  <div className="mt-2">
                    <span className="text-xs text-[#DDDDDD]/70">Current input level</span>
                    <div className="h-2 bg-white/10 rounded-full overflow-hidden mt-1">
                      <div 
                        className="h-full bg-gradient-to-r from-[#00CEC9] to-[#6C5CE7] transition-all duration-100"
                        style={{ width: `${audioLevel * 100}%` }}
                      ></div>
                    </div>
                  </div>
                )}
              </div>

              {/* Test Microphone Button */}
              <button
                onClick={toggleListening}
                className="w-full px-4 py-3 bg-gradient-to-r from-[#6C5CE7] to-[#00CEC9] rounded-lg text-white font-medium hover:shadow-lg transition-all"
              >
                {isListening ? 'Stop Testing' : 'Test Microphone'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VoiceInterface;
