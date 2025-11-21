#!/usr/bin/env python3
"""
Comprehensive Chat Testing Suite
Tests all chat functionality, commands, and conversation patterns for YourDaddy Assistant
"""

import sys
import os
import json
import requests
import time
import threading
import subprocess

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.conversational_ai import AdvancedConversationalAI
except ImportError:
    print("❌ Error: Could not import conversational AI module")
    sys.exit(1)

class ChatTestSuite:
    """Comprehensive chat testing suite"""
    
    def __init__(self):
        self.results = []
        self.test_automation_callback = self._create_test_automation_callback()
        
    def _create_test_automation_callback(self):
        """Create a test automation callback that simulates real automation tools"""
        def callback(action, param):
            print(f"🔧 Automation: {action}({param})")
            
            if action == 'open_application':
                return f"Opened {param} successfully"
            elif action == 'search_google':
                return f"Searching Google for '{param}'"
            elif action == 'play_music':
                if param == 'popular music':
                    return f"Playing popular music playlist"
                else:
                    return f"Playing '{param}'"
            elif action == 'close_application':
                return f"Closed {param}"
            elif action == 'set_volume':
                return f"Volume set to {param}%"
            else:
                return f"Executed {action} with parameter: {param}"
        return callback
    
    def test_conversational_responses(self):
        """Test the improved conversational AI responses"""
        print("\n" + "="*60)
        print("TESTING CONVERSATIONAL AI RESPONSES")
        print("="*60)
        
        # Initialize the conversational AI
        ai = AdvancedConversationalAI()
        
        # Test cases for various conversation patterns
        test_cases = [
            # Greetings
            ("hello", "greeting"),
            ("hi there", "greeting"),
            ("good morning", "greeting"),
            
            # Questions
            ("how are you?", "wellbeing"),
            ("what can you do?", "capabilities"),
            ("what's your name?", "identity"),
            
            # Commands
            ("play music", "music_command"),
            ("play something by coldplay", "music_command"),
            ("open notepad", "app_command"),
            ("search for python tutorials", "search_command"),
            
            # Casual conversation
            ("tell me a joke", "entertainment"),
            ("what's the weather like?", "information"),
            ("I'm feeling sad", "emotional_support"),
            ("that's awesome", "acknowledgment"),
            
            # Complex queries
            ("can you help me learn python programming?", "learning_assistance"),
            ("I need to organize my schedule", "productivity"),
            ("how do I fix my computer?", "technical_support"),
        ]
        
        results = []
        for query, expected_category in test_cases:
            print(f"\nTesting: '{query}'")
            print("-" * 40)
            
            try:
                response = ai.process_message(query)
                print(f"Response: {response}")
                
                # Check if response is appropriate (not too generic)
                is_appropriate = (
                    len(response) > 20 and  # Not too short
                    "I understand" not in response or len(response) > 50 and  # Not just generic understanding
                    response != "I'm here to help you with various tasks and conversations." and
                    "general assistance" not in response.lower()
                )
                
                results.append({
                    'query': query,
                    'response': response,
                    'appropriate': is_appropriate,
                    'category': expected_category
                })
                
                print(f"✓ Appropriate: {is_appropriate}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({
                    'query': query,
                    'response': f"Error: {e}",
                    'appropriate': False,
                    'category': expected_category
                })
        
        self.results.extend(results)
        return results
    
    def test_command_processing(self):
        """Test command processing capabilities"""
        print("\n" + "="*60)
        print("TESTING COMMAND PROCESSING")
        print("="*60)
        
        # Initialize conversational AI (which handles commands)
        ai = AdvancedConversationalAI()
        
        command_tests = [
            "play music by the beatles",
            "open calculator",
            "search google for machine learning",
            "set a timer for 5 minutes",
            "what's the time",
            "open file manager",
            "play video",
            "take a screenshot"
        ]
        
        for command in command_tests:
            print(f"\nTesting command: '{command}'")
            print("-" * 40)
            
            try:
                # Test through conversational AI
                response = ai.process_message(command)
                print(f"AI Response: {response}")
                
            except Exception as e:
                print(f"❌ Error processing command: {e}")
    
    def test_web_api(self):
        """Test the web API endpoints"""
        print("\n" + "="*60)
        print("TESTING WEB API")
        print("="*60)
        
        base_url = "http://localhost:5000"
        
        # Test health endpoint
        try:
            response = requests.get(f"{base_url}/api/health", timeout=5)
            print(f"Health endpoint: {response.status_code} - {response.json()}")
        except Exception as e:
            print(f"❌ Health endpoint failed: {e}")
        
        # Test command endpoint
        test_commands = [
            "hello",
            "what time is it",
            "open calculator"
        ]
        
        for cmd in test_commands:
            try:
                response = requests.post(
                    f"{base_url}/api/command", 
                    json={"command": cmd},
                    timeout=10
                )
                print(f"Command '{cmd}': {response.status_code} - {response.json()}")
            except Exception as e:
                print(f"❌ Command '{cmd}' failed: {e}")
    
    def simulate_interactive_chat(self):
        """Simulate the web chat interface"""
        print("\n" + "="*60)
        print("INTERACTIVE CHAT SIMULATION")
        print("="*60)
        print("This simulates how your web chat interface should work.")
        print("Type 'quit' to exit.")
        print("-"*60)
        
        # Initialize the conversational AI with automation
        ai = AdvancedConversationalAI(automation_callback=self.test_automation_callback)
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n🤖 Assistant: Goodbye! Have a great day! 👋")
                    break
                
                if not user_input:
                    continue
                
                # Process the message
                print("\n🤖 Assistant:", end=" ")
                response = ai.process_message(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n🤖 Assistant: Goodbye! Have a great day! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
    
    def display_supported_commands(self):
        """Display all supported commands"""
        print("\n" + "="*60)
        print("SUPPORTED COMMANDS REFERENCE")
        print("="*60)
        print("\n✅ Your assistant can now handle these commands:\n")
        
        commands = [
            ("Opening Apps", [
                "open chrome",
                "open calculator",
                "open notepad",
                "open spotify",
                "launch word"
            ]),
            ("Opening Websites", [
                "open google.com",
                "open youtube.com",
                "open github.com"
            ]),
            ("Closing Apps", [
                "close chrome",
                "close notepad",
                "quit spotify"
            ]),
            ("Google Search", [
                "google python tutorial",
                "search for best restaurants",
                "find weather forecast"
            ]),
            ("Playing Music", [
                "play believer",
                "play shape of you",
                "play music by coldplay"
            ]),
            ("Creating Documents", [
                "create a powerpoint",
                "make a ppt",
                "create a document",
                "open word document"
            ]),
            ("Volume Control", [
                "volume up",
                "volume down",
                "volume 50",
                "mute volume"
            ]),
            ("System Settings", [
                "open settings",
                "open wifi settings",
                "open bluetooth settings",
                "open display settings",
                "open sound settings"
            ]),
            ("Math & Info", [
                "what is 10 times 5",
                "calculate 100 plus 50",
                "what is pie value",
                "what time is it",
                "what date is today"
            ])
        ]
        
        for category, cmds in commands:
            print(f"\n📌 {category}:")
            for cmd in cmds:
                print(f"   • \"{cmd}\"")
        
        print("\n" + "=" * 60)
        print("🚀 Start the backend and try these commands in the chat!")
        print("   Run: python modern_web_backend.py")
        print("=" * 60)
    
    def debug_chat_responses(self):
        """Debug specific chat responses"""
        print("\n" + "="*60)
        print("DEBUG CHAT RESPONSES")
        print("="*60)
        
        # Initialize AI with automation callback
        ai = AdvancedConversationalAI(automation_callback=self.test_automation_callback)
        
        # Test different types of messages
        test_messages = [
            "hello",
            "how are you",
            "what can you do",
            "open chrome",
            "play music",
            "search for python tutorial",
            "what is 10 plus 5",
            "help me",
            "i need assistance",
            "do something"
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n{i}. User: {message}")
            try:
                response = ai.process_message(message)
                print(f"   AI: {response}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
            print("-" * 40)
    
    def generate_test_report(self):
        """Generate a comprehensive test report"""
        if not self.results:
            print("No test results to generate report from")
            return
        
        total_tests = len(self.results)
        appropriate_responses = sum(1 for r in self.results if r.get('appropriate', False))
        
        print(f"\n{'='*60}")
        print("TEST RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total_tests}")
        print(f"Appropriate Responses: {appropriate_responses}")
        print(f"Success Rate: {(appropriate_responses/total_tests)*100:.1f}%")
        
        # Save detailed report
        report_path = "test_report_chat_consolidated.json"
        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': total_tests,
            'appropriate_responses': appropriate_responses,
            'success_rate': (appropriate_responses/total_tests)*100,
            'detailed_results': self.results
        }
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            print(f"Detailed report saved to: {report_path}")
        except Exception as e:
            print(f"Failed to save report: {e}")
    
    def run_stub_tests(self):
        """Run stub tests with mocked functions for safe testing"""
        print("\n" + "="*60)
        print("STUB TESTS (SAFE MODE)")
        print("="*60)
        
        # Import conversational_ai to patch functions
        from modules import conversational_ai
        
        # Patch side-effectful functions for safe testing
        conversational_ai.webbrowser.open = lambda url: print(f"[webbrowser] {url}")
        
        def fake_popen(cmd, *_, **__):
            print(f"[Popen] {cmd}")
            class Dummy:
                def poll(self):
                    return 0
            return Dummy()
        
        def fake_run(cmd, *_, **__):
            print(f"[run] {cmd}")
            class Result:
                returncode = 0
            return Result()
        
        conversational_ai.subprocess.Popen = fake_popen
        conversational_ai.subprocess.run = fake_run
        
        def fake_callback(action, param):
            return f"[stub] {action} -> {param}"
        
        ai = AdvancedConversationalAI(automation_callback=fake_callback)
        commands = [
            "open chrome",
            "open youtube.com",
            "close chrome",
            "google best laptops",
            "play imagine dragons",
            "create a powerpoint",
            "volume 40",
            "open wifi settings",
            "lock the system",
        ]
        
        for cmd in commands:
            print(f"{cmd} => {ai.process_message(cmd)}")

def main():
    """Main test runner"""
    print("🤖 YourDaddy Assistant - Comprehensive Chat Test Suite")
    print("="*60)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("\nAvailable test modes:")
        print("1. full       - Run all tests")
        print("2. quick      - Run basic conversation tests")
        print("3. commands   - Test command processing")
        print("4. api        - Test web API")
        print("5. interactive- Interactive chat simulation")
        print("6. debug      - Debug chat responses")
        print("7. stub       - Safe stub testing")
        print("8. help       - Show supported commands")
        
        choice = input("\nSelect test mode (1-8): ").strip()
        mode_map = {
            '1': 'full', '2': 'quick', '3': 'commands', '4': 'api',
            '5': 'interactive', '6': 'debug', '7': 'stub', '8': 'help'
        }
        mode = mode_map.get(choice, 'full')
    
    test_suite = ChatTestSuite()
    
    if mode == 'full':
        print("Running comprehensive tests...")
        test_suite.test_conversational_responses()
        test_suite.test_command_processing()
        test_suite.test_web_api()
        test_suite.generate_test_report()
    elif mode == 'quick':
        test_suite.test_conversational_responses()
        test_suite.generate_test_report()
    elif mode == 'commands':
        test_suite.test_command_processing()
    elif mode == 'api':
        test_suite.test_web_api()
    elif mode == 'interactive':
        test_suite.simulate_interactive_chat()
    elif mode == 'debug':
        test_suite.debug_chat_responses()
    elif mode == 'stub':
        test_suite.run_stub_tests()
    elif mode == 'help':
        test_suite.display_supported_commands()
    else:
        print(f"Unknown mode: {mode}")
        return
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()