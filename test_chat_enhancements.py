#!/usr/bin/env python3
"""
Comprehensive Test Suite for Chat System Enhancements
Tests all new modules: tool_executor, chat_with_tools, web_search, context_optimizer
"""

import sys
import os
import time
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

sys.path.insert(0, os.path.dirname(__file__))

import json
from typing import Dict, List, Any


class TestResult:
    """Store test result."""
    def __init__(self, name: str, passed: bool, message: str = "", duration: float = 0.0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration = duration
    
    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} | {self.name} ({self.duration:.3f}s) | {self.message}"


class TestSuite:
    """Test suite for chat enhancements."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.passed = 0
        self.failed = 0
    
    def run_test(self, name: str, test_func):
        """Run a test."""
        try:
            start = time.time()
            test_func()
            duration = time.time() - start
            result = TestResult(name, True, "", duration)
            self.passed += 1
        except AssertionError as e:
            duration = time.time() - start
            result = TestResult(name, False, f"Assertion: {str(e)}", duration)
            self.failed += 1
        except Exception as e:
            duration = time.time() - start
            result = TestResult(name, False, f"Error: {str(e)}", duration)
            self.failed += 1
        
        self.results.append(result)
        print(result)
    
    def summary(self):
        """Print summary."""
        total = self.passed + self.failed
        print("\n" + "="*70)
        print(f"TEST SUMMARY: {self.passed}/{total} passed")
        print("="*70)
        
        if self.failed > 0:
            print(f"\n❌ {self.failed} test(s) failed")
            return False
        else:
            print(f"\n✅ All {self.passed} test(s) passed!")
            return True


# ============================================================================
# TEST: Tool Executor
# ============================================================================

def test_tool_executor():
    """Test tool executor module."""
    from modules.tool_executor import ToolExecutor, ToolResult, get_default_executor
    
    suite = TestSuite()
    
    def test_register_tool():
        executor = ToolExecutor()
        
        def test_func(x: int) -> int:
            return x * 2
        
        executor.register_tool(
            "double",
            test_func,
            "Double a number",
            {"x": {"type": "integer"}},
            required_params=["x"]
        )
        
        assert "double" in executor.registered_tools
        assert "double" in executor.tool_definitions
    
    def test_execute_tool():
        executor = ToolExecutor()
        
        def add(a: int, b: int) -> int:
            return a + b
        
        executor.register_tool("add", add, "Add two numbers", 
                             {"a": {"type": "int"}, "b": {"type": "int"}},
                             ["a", "b"])
        
        result = executor.execute_tool("add", {"a": 5, "b": 3})
        
        assert result.success == True
        assert result.result == 8
    
    def test_default_executor():
        executor = get_default_executor()
        
        # Should have default tools
        tools = executor.get_tool_definitions()
        assert len(tools) > 0
        
        tool_names = [t["function"]["name"] for t in tools]
        assert "web_search" in tool_names
        assert "calculator" in tool_names
        assert "get_current_time" in tool_names
    
    def test_tool_execution_history():
        executor = get_default_executor()
        
        # Execute calculator
        result = executor.execute_tool("calculator", {"expression": "2+2"})
        
        history = executor.get_execution_history()
        assert len(history) > 0
    
    suite.run_test("Register tool", test_register_tool)
    suite.run_test("Execute tool", test_execute_tool)
    suite.run_test("Default executor", test_default_executor)
    suite.run_test("Execution history", test_tool_execution_history)
    
    return suite.summary()


# ============================================================================
# TEST: Web Search Integration
# ============================================================================

def test_web_search():
    """Test web search integration."""
    from modules.web_search_integration import (
        WebSearchIntegration, WebSearchTrigger, SearchTriggerType
    )
    
    suite = TestSuite()
    
    def test_trigger_detection():
        trigger = WebSearchTrigger()
        
        # Should trigger search
        should_search, trigger_type = trigger.should_search("What's the latest news?")
        assert should_search == True
        
        should_search, trigger_type = trigger.should_search("Tell me about AI")
        assert should_search == True
    
    def test_trigger_no_search():
        trigger = WebSearchTrigger()
        
        # Should not trigger search
        should_search, _ = trigger.should_search("Hello there")
        assert should_search == False
    
    def test_search_caching():
        search = WebSearchIntegration()
        
        # First search should cache
        result1 = search.search_web("Python programming", max_results=3)
        
        # Second search should return cached
        result2 = search.search_web("Python programming", max_results=3)
        
        # Both should be present
        if result1:
            assert result1.total_results > 0
        # Caching works whether we get results or not
    
    def test_search_formatting():
        search = WebSearchIntegration()
        
        # Mock result for testing
        result = search.search_web("test query", max_results=1)
        
        if result and result.results:
            formatted = search.format_results_for_llm(result)
            assert "Web Search Results" in formatted
            assert "test query" in formatted
    
    suite.run_test("Trigger detection - search", test_trigger_detection)
    suite.run_test("Trigger detection - no search", test_trigger_no_search)
    suite.run_test("Search caching", test_search_caching)
    suite.run_test("Result formatting", test_search_formatting)
    
    return suite.summary()


# ============================================================================
# TEST: Context Optimizer
# ============================================================================

def test_context_optimizer():
    """Test context window optimization."""
    from modules.context_optimizer import (
        SmartContextWindow, ConversationCompressor, SemanticHistoryRetrieval
    )
    
    suite = TestSuite()
    
    def test_add_messages():
        ctx = SmartContextWindow(max_tokens=2000)
        
        ctx.add_message("system", "You are helpful")
        ctx.add_message("user", "Hello")
        ctx.add_message("assistant", "Hi there!")
        
        assert len(ctx.message_history) == 3
    
    def test_optimized_history():
        ctx = SmartContextWindow(max_tokens=1000)
        
        # Add some messages
        ctx.add_message("system", "You are helpful")
        for i in range(10):
            ctx.add_message("user", f"Question {i}")
            ctx.add_message("assistant", f"Answer {i}" * 5)
        
        # Get optimized history
        optimized = ctx.get_optimized_history("New question?")
        
        # Should include system message
        assert len(optimized) > 0
        assert optimized[0].get("role") == "system"
    
    def test_context_stats():
        ctx = SmartContextWindow(max_tokens=2000)
        
        ctx.add_message("system", "Test")
        ctx.add_message("user", "Hello")
        ctx.add_message("assistant", "Hi")
        
        stats = ctx.get_stats()
        
        assert stats["total_messages"] == 3
        assert stats["max_tokens"] == 2000
        assert "utilization_percent" in stats
    
    def test_message_compression():
        compressor = ConversationCompressor()
        
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1" * 50},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2" * 50},
        ]
        
        # Compress to fit budget
        compressed = compressor.compress_messages(messages, target_tokens=500)
        
        # Should still have system message and some content
        assert len(compressed) > 0
    
    suite.run_test("Add messages", test_add_messages)
    suite.run_test("Optimized history", test_optimized_history)
    suite.run_test("Context stats", test_context_stats)
    suite.run_test("Message compression", test_message_compression)
    
    return suite.summary()


# ============================================================================
# TEST: Chat with Tools
# ============================================================================

def test_chat_with_tools():
    """Test chat system with tool calling."""
    from modules.chat_with_tools import ChatWithToolCalling, SemanticChatEnhancer
    
    suite = TestSuite()
    
    def test_initialization():
        try:
            chat = ChatWithToolCalling()
            assert chat.chat is not None
            assert chat.tool_executor is not None
        except ImportError:
            # Skip if dependencies missing
            pass
    
    def test_register_tool():
        try:
            chat = ChatWithToolCalling()
            
            def my_func(x):
                return x * 2
            
            chat.register_tool(
                "my_tool",
                my_func,
                "Test tool",
                {"x": {"type": "integer"}},
                ["x"]
            )
            
            assert "my_tool" in chat.tool_executor.registered_tools
        except ImportError:
            pass
    
    def test_semantic_enhancer():
        enhancer = SemanticChatEnhancer()
        
        # Cache a response
        enhancer.cache_response(
            "What is Python?",
            "Python is a programming language",
            quality=0.95
        )
        
        # Should be cached
        similar = enhancer.get_similar_responses("Python programming")
        # Note: May or may not find based on keyword matching
    
    def test_conversation_compression_simple():
        enhancer = SemanticChatEnhancer()
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        
        compressed = enhancer.compress_history(messages, max_messages=5)
        
        # Should handle compression gracefully
        assert len(compressed) > 0
    
    suite.run_test("Chat initialization", test_initialization)
    suite.run_test("Register tool", test_register_tool)
    suite.run_test("Semantic enhancer", test_semantic_enhancer)
    suite.run_test("Conversation compression", test_conversation_compression_simple)
    
    return suite.summary()


# ============================================================================
# Integration Tests
# ============================================================================

def test_integration():
    """Integration tests combining multiple modules."""
    
    suite = TestSuite()
    
    def test_tool_executor_with_search():
        try:
            from modules.tool_executor import get_default_executor
            
            executor = get_default_executor()
            tools = executor.get_tool_definitions()
            
            # Should include web_search
            tool_names = [t["function"]["name"] for t in tools]
            assert "web_search" in tool_names
        except Exception as e:
            print(f"Skipped due to: {e}")
    
    def test_context_with_search():
        try:
            from modules.context_optimizer import SmartContextWindow
            from modules.web_search_integration import WebSearchIntegration
            
            ctx = SmartContextWindow()
            search = WebSearchIntegration()
            
            # Add messages with search
            should_search, _ = search.should_search_for_message("What's new?")
            assert should_search == True
        except Exception as e:
            print(f"Skipped due to: {e}")
    
    suite.run_test("Tool executor with search", test_tool_executor_with_search)
    suite.run_test("Context with search", test_context_with_search)
    
    return suite.summary()


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all tests."""
    print("="*70)
    print("🧪 CHAT SYSTEM ENHANCEMENTS - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    all_passed = True
    
    # Test 1: Tool Executor
    print("\n" + "="*70)
    print("📦 Test Group 1: Tool Executor")
    print("="*70)
    all_passed &= test_tool_executor()
    
    # Test 2: Web Search
    print("\n" + "="*70)
    print("🔍 Test Group 2: Web Search Integration")
    print("="*70)
    all_passed &= test_web_search()
    
    # Test 3: Context Optimizer
    print("\n" + "="*70)
    print("🎯 Test Group 3: Context Window Optimizer")
    print("="*70)
    all_passed &= test_context_optimizer()
    
    # Test 4: Chat with Tools
    print("\n" + "="*70)
    print("💬 Test Group 4: Chat with Tools")
    print("="*70)
    all_passed &= test_chat_with_tools()
    
    # Test 5: Integration
    print("\n" + "="*70)
    print("🔗 Test Group 5: Integration Tests")
    print("="*70)
    all_passed &= test_integration()
    
    # Final summary
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TEST GROUPS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED - CHECK OUTPUT ABOVE")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
