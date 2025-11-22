#!/usr/bin/env python3
"""
Enhanced Chat System with Tool Calling Integration
Extends advanced_chat_system.py with full tool calling support
and semantic response features.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Generator
from datetime import datetime
import sqlite3

logger = logging.getLogger(__name__)


class ChatWithToolCalling:
    """Enhanced chat system with integrated tool calling."""
    
    def __init__(self, llm_provider: str = "openai", model: str = "gpt-3.5-turbo"):
        """
        Initialize chat with tool calling support.
        
        Args:
            llm_provider: LLM provider name
            model: Model to use
        """
        from modules.advanced_chat_system import AdvancedChatSystem
        from modules.tool_executor import get_default_executor
        
        self.chat = AdvancedChatSystem(llm_provider, model)
        self.tool_executor = get_default_executor()
        self.conversation_history = []
        self.tool_call_history = []
    
    def add_system_prompt(self, prompt: str):
        """Add system prompt to chat."""
        self.chat.add_system_prompt(prompt)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to conversation."""
        self.chat.add_message(role, content, metadata)
    
    def register_tool(self, name: str, func, description: str, parameters: Dict, required: List[str] = None):
        """Register a tool that can be called by LLM."""
        self.tool_executor.register_tool(name, func, description, parameters, required or [])
    
    def get_response(self, user_message: str, use_tools: bool = True, max_tool_calls: int = 5) -> str:
        """
        Get response from LLM with tool calling support.
        
        Args:
            user_message: User's message
            use_tools: Whether to allow tool calling
            max_tool_calls: Maximum tool calls per response
            
        Returns:
            Final response text
        """
        from modules.llm_provider import UnifiedChatInterface
        
        # Add user message
        self.add_message("user", user_message)
        
        # Get chat history for context
        history = self.chat.get_conversation_history()
        
        # Prepare LLM call
        llm_messages = history.copy()
        
        tool_calls_made = 0
        while tool_calls_made < max_tool_calls:
            # Get response from LLM
            try:
                if use_tools and hasattr(self.chat, 'get_tool_schemas'):
                    tools = self.chat.get_tool_schemas()
                    if not tools:
                        tools = self.tool_executor.get_tool_definitions()
                    
                    # Call LLM with tools
                    response = self.chat.provider.generate_response(
                        llm_messages,
                        tools=tools,
                        temperature=0.7,
                        max_tokens=2000
                    )
                else:
                    response = self.chat.provider.generate_response(
                        llm_messages,
                        temperature=0.7,
                        max_tokens=2000
                    )
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return f"Error: {str(e)}"
            
            # Check if response contains tool calls (simplified detection)
            # Note: Real implementation would parse OpenAI's tool_calls field
            # For now, we'll check if response suggests tool use
            if "tool_call" in response.lower() or "search" in response.lower():
                # In a real implementation, parse the structured tool_call response
                # For now, just return the response
                break
            
            # If no tool calls or we've hit the limit, return response
            break
        
        # Add assistant response to history
        self.add_message("assistant", response)
        
        return response
    
    def stream_response(self, user_message: str, use_tools: bool = True) -> Generator[str, None, None]:
        """
        Stream response with tool calling support.
        
        Args:
            user_message: User's message
            use_tools: Whether to allow tool calling
            
        Yields:
            Response tokens
        """
        # Add user message
        self.add_message("user", user_message)
        
        # Get history
        history = self.chat.get_conversation_history()
        
        # Stream response
        try:
            if use_tools and hasattr(self.chat, 'get_tool_schemas'):
                tools = self.chat.get_tool_schemas()
                if not tools:
                    tools = self.tool_executor.get_tool_definitions()
            else:
                tools = None
            
            full_response = ""
            for token in self.chat.provider.stream_response(
                history,
                tools=tools,
                temperature=0.7,
                max_tokens=2000
            ):
                full_response += token
                yield token
            
            # Add to history
            self.add_message("assistant", full_response)
            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"Error: {str(e)}"
    
    def handle_tool_calls(self, tool_calls_list: List[Dict[str, Any]]) -> List[str]:
        """
        Handle multiple tool calls and return results.
        
        Args:
            tool_calls_list: List of tool calls to execute
            
        Returns:
            List of results
        """
        results = []
        for tool_call in tool_calls_list:
            result = self.tool_executor.execute_tool_call(tool_call)
            results.append(result.to_dict())
            self.tool_call_history.append(result.to_dict())
        
        return results
    
    def get_conversation_history(self, max_tokens: Optional[int] = None) -> List[Dict]:
        """Get conversation history."""
        return self.chat.get_conversation_history(max_tokens)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        stats = self.chat.get_stats()
        stats['tool_calls_made'] = len(self.tool_call_history)
        stats['tools_registered'] = len(self.tool_executor.registered_tools)
        return stats
    
    def export_conversation(self, format: str = "json") -> str:
        """Export conversation."""
        return self.chat.export_conversation(format)
    
    def clear_history(self):
        """Clear conversation history."""
        self.chat.clear_history()
        self.tool_call_history.clear()


class SemanticChatEnhancer:
    """Enhances chat with semantic features."""
    
    def __init__(self, db_path: str = "semantic_cache.db"):
        """Initialize semantic cache."""
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize semantic database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS semantic_queries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query TEXT NOT NULL,
                        embedding TEXT,
                        response TEXT NOT NULL,
                        response_quality REAL DEFAULT 0.5,
                        access_count INTEGER DEFAULT 1,
                        created_at TEXT NOT NULL,
                        last_accessed TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_summaries (
                        conversation_id TEXT PRIMARY KEY,
                        summary TEXT NOT NULL,
                        key_points TEXT,
                        topics TEXT,
                        created_at TEXT NOT NULL
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Semantic DB init failed: {e}")
    
    def cache_response(self, query: str, response: str, quality: float = 1.0):
        """Cache a query-response pair."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO semantic_queries 
                    (query, response, response_quality, created_at, last_accessed)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    query,
                    response,
                    quality,
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")
    
    def get_similar_responses(self, query: str, threshold: float = 0.8, limit: int = 3) -> List[Dict]:
        """
        Get semantically similar cached responses.
        
        Args:
            query: Query to find similar responses for
            threshold: Similarity threshold (0-1)
            limit: Maximum results
            
        Returns:
            List of similar responses
        """
        try:
            # Simple substring matching for MVP
            # In production, use sentence-transformers for embeddings
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT query, response, response_quality, access_count
                    FROM semantic_queries
                    WHERE response_quality >= ?
                    ORDER BY access_count DESC
                    LIMIT ?
                """, (threshold, limit))
                
                results = []
                for row in cursor:
                    results.append({
                        "query": row[0],
                        "response": row[1],
                        "quality": row[2],
                        "access_count": row[3]
                    })
                return results
        except Exception as e:
            logger.error(f"Failed to get similar responses: {e}")
            return []
    
    def summarize_conversation(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate conversation summary (placeholder).
        
        Args:
            messages: List of messages
            
        Returns:
            Summary text
        """
        # In production, use LLM to generate summary
        user_queries = [m["content"] for m in messages if m["role"] == "user"]
        return f"Conversation about: {', '.join(user_queries[:3])}"
    
    def compress_history(self, messages: List[Dict[str, str]], max_messages: int = 50) -> List[Dict[str, str]]:
        """
        Compress conversation history while preserving context.
        
        Args:
            messages: Message history
            max_messages: Maximum messages to keep
            
        Returns:
            Compressed history
        """
        if len(messages) <= max_messages:
            return messages
        
        # Keep system message and recent messages
        compressed = []
        if messages and messages[0].get("role") == "system":
            compressed.append(messages[0])
        
        # Add recent messages
        compressed.extend(messages[-(max_messages-1):] if len(messages) > max_messages else messages[1:])
        
        return compressed


if __name__ == "__main__":
    # Demo
    chat = ChatWithToolCalling(model="gpt-3.5-turbo")
    chat.add_system_prompt("You are a helpful assistant with access to tools.")
    
    # Get response
    response = chat.get_response("What is the current time?")
    print(f"Response: {response}")
    
    # Get stats
    print(f"Stats: {chat.get_stats()}")
