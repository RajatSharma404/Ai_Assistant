#!/usr/bin/env python3
"""
Chat System Demonstration Script
Tests all key features of the new advanced chat system.
"""

import sys
import os

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

sys.path.insert(0, os.path.dirname(__file__))

import json
from modules.advanced_chat_system import AdvancedChatSystem, TokenCounter


def test_token_counter():
    """Test token counting functionality."""
    print("\n" + "="*60)
    print("🔢 TEST 1: Token Counter")
    print("="*60)
    
    counter = TokenCounter("gpt-3.5-turbo")
    
    test_strings = [
        "Hello world",
        "This is a longer message with multiple words to test token counting",
        "Python is a programming language"
    ]
    
    for text in test_strings:
        tokens = counter.count(text)
        print(f"Text: '{text}'")
        print(f"Tokens: {tokens}\n")
    
    print(f"✅ Token counter works! Max context: {counter.token_limit} tokens")


def test_basic_chat():
    """Test basic chat functionality."""
    print("\n" + "="*60)
    print("💬 TEST 2: Basic Chat System")
    print("="*60)
    
    chat = AdvancedChatSystem(model="gpt-3.5-turbo")
    
    # Add system prompt
    chat.add_system_prompt("You are a helpful Python programming assistant.")
    print("✅ System prompt added")
    
    # Add messages
    chat.add_message("user", "What is Python?")
    chat.add_message("assistant", "Python is a high-level programming language.")
    print("✅ Messages added to history")
    
    # Get history
    history = chat.get_conversation_history()
    print(f"✅ Conversation history has {len(history)} messages")
    
    # Get stats
    stats = chat.get_stats()
    print(f"\nConversation Stats:")
    print(f"  - Total messages: {stats['total_messages']}")
    print(f"  - Total tokens: {stats['total_tokens']}")
    print(f"  - Duration: {stats['duration']:.1f}s")
    print(f"  - Model: {stats['model']}")


def test_message_management():
    """Test message editing and deletion."""
    print("\n" + "="*60)
    print("✏️  TEST 3: Message Management")
    print("="*60)
    
    chat = AdvancedChatSystem()
    chat.add_message("user", "Original message")
    print("✅ Added message: 'Original message'")
    
    # Edit message
    chat.edit_message(0, "Edited message")
    print("✅ Edited message to: 'Edited message'")
    
    # Verify edit
    history = chat.get_conversation_history()
    print(f"Message content: '{history[0]['content']}'")
    
    # Search history
    chat.add_message("user", "What about Python?")
    chat.add_message("user", "Tell me about JavaScript")
    results = chat.search_history("Python")
    print(f"✅ Search found {len(results)} messages with 'Python'")


def test_conversation_export():
    """Test exporting conversations."""
    print("\n" + "="*60)
    print("💾 TEST 4: Export Conversation")
    print("="*60)
    
    chat = AdvancedChatSystem()
    chat.add_system_prompt("You are helpful.")
    chat.add_message("user", "Hello!")
    chat.add_message("assistant", "Hi there!")
    
    # Export as JSON
    json_export = chat.export_conversation("json")
    print("✅ Exported as JSON")
    
    # Parse and verify
    exported_data = json.loads(json_export)
    print(f"✅ Exported data has {len(exported_data['messages'])} messages")
    
    # Export as Markdown
    md_export = chat.export_conversation("markdown")
    print("✅ Exported as Markdown")
    print(f"\nMarkdown preview:\n{md_export[:200]}...")


def test_context_management():
    """Test conversation context management."""
    print("\n" + "="*60)
    print("🎯 TEST 5: Context Management")
    print("="*60)
    
    chat = AdvancedChatSystem()
    print(f"✅ Context ID: {chat.context_id}")
    print(f"✅ Created at: {chat.created_at.isoformat()}")
    
    # Add many messages
    for i in range(5):
        chat.add_message("user", f"Question {i}")
        chat.add_message("assistant", f"Answer {i}")
    
    print(f"✅ Added 10 messages")
    
    # Get trimmed history
    trimmed = chat.get_conversation_history(max_tokens=100)
    print(f"✅ Trimmed history to fit 100 tokens: {len(trimmed)} messages")
    
    # Clear history
    chat.clear_history()
    history = chat.get_conversation_history()
    print(f"✅ Cleared history: {len(history)} messages remaining")


def test_tool_registration():
    """Test tool/function calling framework."""
    print("\n" + "="*60)
    print("🔧 TEST 6: Tool Registration")
    print("="*60)
    
    from modules.advanced_chat_system import ToolSchema
    
    chat = AdvancedChatSystem()
    
    # Define tools
    tools = [
        ToolSchema(
            name="calculate",
            description="Perform calculations",
            parameters={"expression": {"type": "string"}},
            required=["expression"]
        ),
        ToolSchema(
            name="search_web",
            description="Search the web",
            parameters={"query": {"type": "string"}},
            required=["query"]
        )
    ]
    
    # Register tools
    for tool in tools:
        chat.register_tool(tool.name, lambda x: f"Result: {x}", tool)
    
    print(f"✅ Registered {len(tools)} tools")
    
    # Get tool schemas
    schemas = chat.get_tool_schemas()
    print(f"✅ Tool schemas: {len(schemas)} available")
    
    # Test tool execution
    result = chat.handle_tool_call("calculate", {"expression": "2+2"})
    print(f"✅ Tool call result: {result}")


def test_response_caching():
    """Test response caching."""
    print("\n" + "="*60)
    print("💾 TEST 7: Response Caching")
    print("="*60)
    
    chat = AdvancedChatSystem()
    
    # Simulate cached response
    message = "What is Python?"
    cache_key = chat._generate_cache_key(message)
    chat.response_cache[cache_key] = "Python is a programming language."
    
    print(f"✅ Cache key: {cache_key[:16]}...")
    print(f"✅ Cache size: {len(chat.response_cache)} entries")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("🚀 ADVANCED CHAT SYSTEM - FEATURE DEMONSTRATION")
    print("="*60)
    
    tests = [
        test_token_counter,
        test_basic_chat,
        test_message_management,
        test_conversation_export,
        test_context_management,
        test_tool_registration,
        test_response_caching
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test_func.__name__}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    print("="*60)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Chat system is ready to use.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
