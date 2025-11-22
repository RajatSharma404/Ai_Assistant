#!/usr/bin/env python3
"""
Enhanced WebSocket Handlers for Modern Web Backend
Adds tool calling, advanced streaming, and semantic features.

Integration:
Add this code to modern_web_backend.py in the WebSocket sections.
"""

import json
import logging
import time
from typing import Optional, Dict, Any
from flask_socketio import emit, request

logger = logging.getLogger(__name__)

# Global state (shared with modern_web_backend.py)
# These are populated in the main backend file
chat_sessions = {}
chat_session_lock = None
socketio = None


def create_enhanced_websocket_handlers(app, socketio_instance, chat_session_lock_instance):
    """
    Create enhanced WebSocket handlers.
    
    Call this in modern_web_backend.py after creating socketio instance:
        enhance_websocket_handlers(app, socketio, chat_session_lock)
    """
    global socketio, chat_session_lock
    socketio = socketio_instance
    chat_session_lock = chat_session_lock_instance
    
    # Register handlers
    setup_streaming_with_tools()
    setup_semantic_chat()
    setup_advanced_chat_features()


def setup_streaming_with_tools():
    """Enhanced streaming with tool calling support."""
    
    @socketio.on('chat_stream_with_tools')
    def handle_chat_stream_with_tools(data):
        """
        Stream chat response with tool calling support.
        Streams tokens and tool results in real-time.
        """
        try:
            message = data.get('message', '')
            session_id = data.get('session_id', request.sid)
            use_tools = data.get('use_tools', True)
            
            if not message:
                emit('error', {'error': 'No message provided'})
                return
            
            logger.info(f"🔧 Streaming with tools: {session_id}")
            
            # Get or create chat session with tools
            with chat_session_lock:
                if session_id not in chat_sessions:
                    try:
                        from modules.chat_with_tools import ChatWithToolCalling
                        chat_sessions[session_id] = ChatWithToolCalling()
                        chat_sessions[session_id].add_system_prompt(
                            "You are a helpful AI assistant with access to tools. "
                            "Use tools when needed to provide accurate information."
                        )
                    except ImportError:
                        emit('error', {'error': 'Chat with tools module not available'})
                        return
                
                chat = chat_sessions[session_id]
            
            # Stream response with tool calling
            start_time = time.time()
            tokens = 0
            full_response = ""
            
            try:
                for token in chat.stream_response(message, use_tools=use_tools):
                    tokens += 1
                    full_response += token
                    
                    # Emit token
                    emit('chat_token', {
                        'token': token,
                        'count': tokens,
                        'partial': full_response,
                        'with_tools': use_tools
                    })
                    
                    # Small delay
                    time.sleep(0.001)
            
            except Exception as stream_error:
                logger.error(f"Streaming error: {stream_error}")
                emit('error', {'error': f'Streaming failed: {str(stream_error)}'})
                return
            
            # Send completion with stats
            duration = time.time() - start_time
            stats = chat.get_stats()
            
            emit('chat_complete', {
                'tokens': tokens,
                'duration': round(duration, 2),
                'tokens_per_second': round(tokens / duration, 2) if duration > 0 else 0,
                'full_response': full_response,
                'session_id': session_id,
                'stats': stats,
                'tool_calls': stats.get('tool_calls_made', 0),
                'timestamp': time.time()
            })
            
            logger.info(f"✅ Stream complete: {tokens} tokens, {stats.get('tool_calls_made', 0)} tools")
        
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            emit('error', {'error': f'Failed: {str(e)}'})
    
    # Return handler reference
    return handle_chat_stream_with_tools


def setup_semantic_chat():
    """Semantic chat with caching and similarity."""
    
    @socketio.on('semantic_chat')
    def handle_semantic_chat(data):
        """
        Chat with semantic caching and response reuse.
        Returns cached responses for similar queries.
        """
        try:
            message = data.get('message', '')
            session_id = data.get('session_id', request.sid)
            use_cache = data.get('use_cache', True)
            
            if not message:
                emit('error', {'error': 'No message provided'})
                return
            
            # Check cache for similar responses
            if use_cache:
                try:
                    from modules.chat_with_tools import SemanticChatEnhancer
                    enhancer = SemanticChatEnhancer()
                    
                    similar = enhancer.get_similar_responses(message, threshold=0.8, limit=1)
                    
                    if similar:
                        emit('cached_response', {
                            'message': message,
                            'response': similar[0]['response'],
                            'from_cache': True,
                            'quality': similar[0]['quality'],
                            'session_id': session_id,
                            'timestamp': time.time()
                        })
                        logger.info(f"✅ Used cached response")
                        return
                except Exception as e:
                    logger.warning(f"Cache lookup failed: {e}")
            
            # If not cached, generate response
            with chat_session_lock:
                if session_id not in chat_sessions:
                    try:
                        from modules.chat_with_tools import ChatWithToolCalling
                        chat_sessions[session_id] = ChatWithToolCalling()
                    except ImportError:
                        emit('error', {'error': 'Chat module not available'})
                        return
                
                chat = chat_sessions[session_id]
            
            # Generate response
            response = chat.get_response(message, use_tools=True)
            
            # Cache the response
            try:
                from modules.chat_with_tools import SemanticChatEnhancer
                enhancer = SemanticChatEnhancer()
                enhancer.cache_response(message, response, quality=0.95)
            except Exception as e:
                logger.warning(f"Failed to cache response: {e}")
            
            # Emit response
            emit('semantic_response', {
                'message': message,
                'response': response,
                'from_cache': False,
                'session_id': session_id,
                'timestamp': time.time()
            })
        
        except Exception as e:
            logger.error(f"Semantic chat error: {e}")
            emit('error', {'error': str(e)})


def setup_advanced_chat_features():
    """Advanced features: regenerate, alternatives, continue."""
    
    @socketio.on('regenerate_response')
    def handle_regenerate(data):
        """Regenerate the last response."""
        try:
            session_id = data.get('session_id', request.sid)
            
            with chat_session_lock:
                if session_id not in chat_sessions:
                    emit('error', {'error': 'Session not found'})
                    return
                
                chat = chat_sessions[session_id]
            
            # Regenerate
            response = chat.chat.regenerate_response()
            
            emit('regenerated_response', {
                'response': response,
                'session_id': session_id,
                'timestamp': time.time()
            })
        
        except Exception as e:
            logger.error(f"Regeneration error: {e}")
            emit('error', {'error': str(e)})
    
    @socketio.on('get_alternatives')
    def handle_alternatives(data):
        """Get alternative responses."""
        try:
            session_id = data.get('session_id', request.sid)
            count = data.get('count', 3)
            
            with chat_session_lock:
                if session_id not in chat_sessions:
                    emit('error', {'error': 'Session not found'})
                    return
                
                chat = chat_sessions[session_id]
            
            # Get alternatives
            alternatives = chat.chat.get_alternatives(num_alternatives=count)
            
            emit('alternatives', {
                'alternatives': alternatives,
                'count': len(alternatives),
                'session_id': session_id,
                'timestamp': time.time()
            })
        
        except Exception as e:
            logger.error(f"Alternatives error: {e}")
            emit('error', {'error': str(e)})
    
    @socketio.on('continue_response')
    def handle_continue(data):
        """Continue the last response."""
        try:
            session_id = data.get('session_id', request.sid)
            
            with chat_session_lock:
                if session_id not in chat_sessions:
                    emit('error', {'error': 'Session not found'})
                    return
                
                chat = chat_sessions[session_id]
            
            # Continue generating
            history = chat.get_conversation_history()
            if not history or history[-1].get('role') != 'assistant':
                emit('error', {'error': 'No response to continue'})
                return
            
            # Add continuation prompt
            last_response = history[-1]['content']
            continuation_prompt = f"Continue the previous response:\n{last_response}\n\n"
            
            full_response = last_response
            for token in chat.stream_response(continuation_prompt):
                full_response += token
                emit('token', {'token': token})
            
            emit('continue_complete', {
                'response': full_response,
                'session_id': session_id,
                'timestamp': time.time()
            })
        
        except Exception as e:
            logger.error(f"Continue error: {e}")
            emit('error', {'error': str(e)})
    
    @socketio.on('edit_message')
    def handle_edit_message(data):
        """Edit a message in conversation."""
        try:
            session_id = data.get('session_id', request.sid)
            message_index = data.get('message_index')
            new_content = data.get('new_content')
            
            with chat_session_lock:
                if session_id not in chat_sessions:
                    emit('error', {'error': 'Session not found'})
                    return
                
                chat = chat_sessions[session_id]
            
            # Edit message
            chat.chat.edit_message(message_index, new_content)
            
            emit('message_edited', {
                'index': message_index,
                'content': new_content,
                'session_id': session_id,
                'timestamp': time.time()
            })
        
        except Exception as e:
            logger.error(f"Edit error: {e}")
            emit('error', {'error': str(e)})
    
    @socketio.on('search_history')
    def handle_search_history(data):
        """Search conversation history."""
        try:
            session_id = data.get('session_id', request.sid)
            query = data.get('query', '')
            limit = data.get('limit', 10)
            
            with chat_session_lock:
                if session_id not in chat_sessions:
                    emit('error', {'error': 'Session not found'})
                    return
                
                chat = chat_sessions[session_id]
            
            # Search
            results = chat.chat.search_history(query, limit=limit)
            
            emit('search_results', {
                'query': query,
                'results': results,
                'count': len(results),
                'session_id': session_id,
                'timestamp': time.time()
            })
        
        except Exception as e:
            logger.error(f"Search error: {e}")
            emit('error', {'error': str(e)})
    
    @socketio.on('export_conversation')
    def handle_export(data):
        """Export conversation in various formats."""
        try:
            session_id = data.get('session_id', request.sid)
            format_type = data.get('format', 'json')  # json, markdown, text
            
            with chat_session_lock:
                if session_id not in chat_sessions:
                    emit('error', {'error': 'Session not found'})
                    return
                
                chat = chat_sessions[session_id]
            
            # Export
            exported = chat.export_conversation(format=format_type)
            
            emit('conversation_exported', {
                'format': format_type,
                'content': exported,
                'session_id': session_id,
                'timestamp': time.time()
            })
        
        except Exception as e:
            logger.error(f"Export error: {e}")
            emit('error', {'error': str(e)})


# Integration code to add to modern_web_backend.py
INTEGRATION_NOTES = """
To integrate these handlers into modern_web_backend.py:

1. Add import at the top:
   from modules.websocket_handlers import create_enhanced_websocket_handlers

2. After creating socketio instance, call:
   create_enhanced_websocket_handlers(app, socketio, chat_session_lock)

3. The handlers will automatically register with socketio.

Available WebSocket Events:
- chat_stream_with_tools: Stream with tool calling
- semantic_chat: Chat with semantic caching
- regenerate_response: Regenerate last response
- get_alternatives: Get 3+ alternative responses
- continue_response: Continue last response
- edit_message: Edit message in history
- search_history: Search conversation history
- export_conversation: Export to JSON/Markdown
"""

if __name__ == "__main__":
    print(INTEGRATION_NOTES)
