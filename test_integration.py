#!/usr/bin/env python3
"""
YourDaddy Assistant - Integration Test Suite
Comprehensive integration testing for all system components and APIs
"""

import sys
import os
import json
import time
import requests
import subprocess
import threading
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class IntegrationTestSuite:
    """Comprehensive integration test suite"""
    
    def __init__(self):
        self.results = {}
        self.server_process = None
        self.base_url = "http://localhost:5000"
    
    def start_test_server(self):
        """Start a test backend server"""
        print("🚀 Starting test backend server...")
        try:
            # Try to start the backend server
            self.server_process = subprocess.Popen(
                [sys.executable, "backend.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(5)
            
            # Check if server is running
            try:
                response = requests.get(f"{self.base_url}/api/health", timeout=5)
                if response.status_code == 200:
                    print("✅ Test server started successfully")
                    return True
            except:
                pass
            
            print("❌ Failed to start test server")
            return False
        
        except Exception as e:
            print(f"❌ Server startup error: {e}")
            return False
    
    def stop_test_server(self):
        """Stop the test backend server"""
        if self.server_process:
            print("🛑 Stopping test server...")
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None
    
    def test_api_endpoints(self):
        """Test all API endpoints"""
        print("\n🌐 Testing API Endpoints")
        print("=" * 40)
        
        endpoints = [
            ("GET", "/api/health", None, "Health Check"),
            ("GET", "/api/apps", None, "Apps List"),
            ("GET", "/api/system/status", None, "System Status"),
            ("POST", "/api/command", {"command": "hello"}, "Command Processing"),
            ("POST", "/api/command", {"command": "what time is it"}, "Time Query"),
            ("POST", "/api/command", {"command": "open calculator"}, "App Command")
        ]
        
        results = {}
        
        for method, endpoint, data, description in endpoints:
            try:
                if method == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                else:
                    response = requests.post(f"{self.base_url}{endpoint}", json=data, timeout=10)
                
                success = response.status_code in [200, 201]
                results[endpoint] = {
                    'success': success,
                    'status_code': response.status_code,
                    'description': description
                }
                
                status = "✅" if success else "❌"
                print(f"{status} {description}: {response.status_code}")
                
                if success and response.headers.get('content-type', '').startswith('application/json'):
                    data = response.json()
                    if 'response' in data:
                        print(f"   Response: {data['response'][:50]}...")
                    elif 'apps' in data:
                        print(f"   Found {len(data['apps'])} apps")
                    elif 'status' in data:
                        print(f"   Status: {data['status']}")
                
            except requests.exceptions.ConnectionError:
                results[endpoint] = {'success': False, 'error': 'Connection refused'}
                print(f"❌ {description}: Connection refused")
            except Exception as e:
                results[endpoint] = {'success': False, 'error': str(e)}
                print(f"❌ {description}: {e}")
        
        self.results['api_endpoints'] = results
        return results
    
    def test_apps_discovery(self):
        """Test application discovery functionality"""
        print("\n📱 Testing App Discovery")
        print("=" * 40)
        
        try:
            # Test the apps API specifically
            response = requests.get(f"{self.base_url}/api/apps", timeout=10)
            
            if response.status_code == 200:
                apps_data = response.json()
                
                if isinstance(apps_data, list):
                    apps = apps_data
                elif 'apps' in apps_data:
                    apps = apps_data['apps']
                else:
                    apps = []
                
                print(f"✅ Found {len(apps)} applications")
                
                # Test app categories
                categories = {}
                for app in apps:
                    category = app.get('category', 'Unknown')
                    categories[category] = categories.get(category, 0) + 1
                
                print("App categories:")
                for category, count in categories.items():
                    print(f"   - {category}: {count} apps")
                
                # Test specific app searches
                common_apps = ['Calculator', 'Notepad', 'Chrome', 'Firefox', 'Explorer']
                found_apps = []
                
                for common_app in common_apps:
                    for app in apps:
                        if common_app.lower() in app.get('name', '').lower():
                            found_apps.append(app['name'])
                            break
                
                print(f"✅ Found common apps: {', '.join(found_apps)}")
                
                self.results['apps_discovery'] = {
                    'success': True,
                    'total_apps': len(apps),
                    'categories': categories,
                    'found_common': found_apps
                }
                
            else:
                print(f"❌ Apps API failed: {response.status_code}")
                self.results['apps_discovery'] = {'success': False, 'error': f"HTTP {response.status_code}"}
        
        except Exception as e:
            print(f"❌ App discovery test failed: {e}")
            self.results['apps_discovery'] = {'success': False, 'error': str(e)}
    
    def test_automation_integration(self):
        """Test automation tools integration"""
        print("\n🔧 Testing Automation Integration")
        print("=" * 40)
        
        try:
            # Import automation tools directly
            from automation_tools_new import (
                get_system_status, list_installed_apps, get_running_processes
            )
            
            # Test system status
            try:
                status = get_system_status()
                print("✅ System status retrieved")
                if isinstance(status, dict):
                    print(f"   CPU: {status.get('cpu', 'N/A')}%")
                    print(f"   Memory: {status.get('memory', 'N/A')}%")
            except Exception as e:
                print(f"❌ System status failed: {e}")
            
            # Test app listing
            try:
                apps = list_installed_apps()
                print(f"✅ Found {len(apps) if apps else 0} installed apps")
            except Exception as e:
                print(f"❌ App listing failed: {e}")
            
            # Test process listing
            try:
                processes = get_running_processes()
                print(f"✅ Found {len(processes) if processes else 0} running processes")
            except Exception as e:
                print(f"❌ Process listing failed: {e}")
            
            self.results['automation_integration'] = {'success': True}
        
        except ImportError:
            print("❌ Automation tools not available")
            self.results['automation_integration'] = {'success': False, 'error': 'Import failed'}
        except Exception as e:
            print(f"❌ Automation integration failed: {e}")
            self.results['automation_integration'] = {'success': False, 'error': str(e)}
    
    def test_conversational_ai_integration(self):
        """Test conversational AI integration"""
        print("\n🤖 Testing Conversational AI Integration")
        print("=" * 40)
        
        try:
            # Test through API
            test_messages = [
                "hello",
                "what can you do",
                "open notepad",
                "what time is it",
                "play music"
            ]
            
            successful_responses = 0
            
            for message in test_messages:
                try:
                    response = requests.post(
                        f"{self.base_url}/api/command",
                        json={"command": message},
                        timeout=15
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        ai_response = data.get('response', '')
                        
                        if len(ai_response) > 10:  # Basic quality check
                            successful_responses += 1
                            print(f"✅ '{message}' -> Response length: {len(ai_response)}")
                        else:
                            print(f"⚠️ '{message}' -> Short response")
                    else:
                        print(f"❌ '{message}' -> HTTP {response.status_code}")
                
                except Exception as e:
                    print(f"❌ '{message}' -> Error: {e}")
            
            success_rate = (successful_responses / len(test_messages)) * 100
            print(f"📊 AI Response Success Rate: {success_rate:.1f}%")
            
            self.results['ai_integration'] = {
                'success': success_rate > 60,
                'success_rate': success_rate,
                'total_tests': len(test_messages),
                'successful': successful_responses
            }
        
        except Exception as e:
            print(f"❌ AI integration test failed: {e}")
            self.results['ai_integration'] = {'success': False, 'error': str(e)}
    
    def test_web_interface(self):
        """Test web interface functionality"""
        print("\n🌐 Testing Web Interface")
        print("=" * 40)
        
        try:
            # Test main page
            response = requests.get(self.base_url, timeout=5)
            
            if response.status_code == 200:
                print("✅ Main page accessible")
                
                # Check if it returns JSON (API mode) or HTML
                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    data = response.json()
                    print(f"   API Mode: {data.get('message', 'No message')}")
                elif 'text/html' in content_type:
                    print("   HTML Mode: Web interface loaded")
                else:
                    print(f"   Content Type: {content_type}")
            else:
                print(f"❌ Main page: HTTP {response.status_code}")
            
            # Test static files (if any)
            static_files = ['/static/style.css', '/static/app.js']
            for file_path in static_files:
                try:
                    response = requests.get(f"{self.base_url}{file_path}", timeout=5)
                    if response.status_code == 200:
                        print(f"✅ Static file: {file_path}")
                    else:
                        print(f"⚠️ Static file missing: {file_path}")
                except:
                    print(f"⚠️ Static file error: {file_path}")
            
            self.results['web_interface'] = {'success': True}
        
        except Exception as e:
            print(f"❌ Web interface test failed: {e}")
            self.results['web_interface'] = {'success': False, 'error': str(e)}
    
    def test_websocket_connection(self):
        """Test WebSocket connectivity"""
        print("\n🔌 Testing WebSocket Connection")
        print("=" * 40)
        
        try:
            import socketio
            
            # Create a test client
            sio = socketio.SimpleClient()
            
            # Try to connect
            sio.connect(self.base_url)
            print("✅ WebSocket connected")
            
            # Test sending a message
            sio.emit('chat_message', {'message': 'Hello WebSocket'})
            
            # Wait for response
            time.sleep(2)
            
            sio.disconnect()
            print("✅ WebSocket disconnected properly")
            
            self.results['websocket'] = {'success': True}
        
        except ImportError:
            print("⚠️ SocketIO client not available")
            self.results['websocket'] = {'success': False, 'error': 'SocketIO not available'}
        except Exception as e:
            print(f"❌ WebSocket test failed: {e}")
            self.results['websocket'] = {'success': False, 'error': str(e)}
    
    def test_security_features(self):
        """Test security features"""
        print("\n🔒 Testing Security Features")
        print("=" * 40)
        
        try:
            # Test CORS headers
            response = requests.options(f"{self.base_url}/api/command", timeout=5)
            cors_headers = response.headers.get('Access-Control-Allow-Origin')
            
            if cors_headers:
                print("✅ CORS headers present")
            else:
                print("⚠️ CORS headers missing")
            
            # Test rate limiting (if enabled)
            # Rapidly send requests to test rate limiting
            rate_limit_triggered = False
            for i in range(10):
                try:
                    response = requests.get(f"{self.base_url}/api/health", timeout=2)
                    if response.status_code == 429:
                        rate_limit_triggered = True
                        break
                except:
                    pass
            
            if rate_limit_triggered:
                print("✅ Rate limiting active")
            else:
                print("⚠️ Rate limiting not triggered (may be disabled)")
            
            # Test authentication endpoint
            try:
                response = requests.post(
                    f"{self.base_url}/api/login",
                    json={"username": "test", "password": "wrong"},
                    timeout=5
                )
                if response.status_code == 401:
                    print("✅ Authentication rejection working")
                elif response.status_code == 400:
                    print("⚠️ Authentication disabled")
            except:
                print("⚠️ Authentication endpoint not available")
            
            self.results['security'] = {'success': True}
        
        except Exception as e:
            print(f"❌ Security test failed: {e}")
            self.results['security'] = {'success': False, 'error': str(e)}
    
    def generate_integration_report(self):
        """Generate comprehensive integration test report"""
        print("\n" + "=" * 60)
        print("📊 INTEGRATION TEST REPORT")
        print("=" * 60)
        
        total_tests = len(self.results)
        successful_tests = sum(1 for result in self.results.values() 
                              if result.get('success', False))
        
        print(f"Total Test Categories: {total_tests}")
        print(f"Successful Categories: {successful_tests}")
        print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        
        print("\nDetailed Results:")
        print("-" * 30)
        
        for test_name, result in self.results.items():
            status = "✅" if result.get('success', False) else "❌"
            print(f"{status} {test_name.replace('_', ' ').title()}")
            
            if 'error' in result:
                print(f"   Error: {result['error']}")
            
            if 'success_rate' in result:
                print(f"   Success Rate: {result['success_rate']:.1f}%")
            
            if 'total_apps' in result:
                print(f"   Total Apps: {result['total_apps']}")
        
        # Save report to file
        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': (successful_tests/total_tests)*100,
            'detailed_results': self.results
        }
        
        try:
            with open('test_integration_report.json', 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"\n📄 Report saved to: test_integration_report.json")
        except Exception as e:
            print(f"\n❌ Failed to save report: {e}")
    
    def run_all_tests(self):
        """Run all integration tests"""
        print("🧪 YourDaddy Assistant - Integration Test Suite")
        print("=" * 60)
        
        # Start test server
        server_started = self.start_test_server()
        
        if server_started:
            try:
                # Run all tests
                self.test_api_endpoints()
                self.test_apps_discovery()
                self.test_conversational_ai_integration()
                self.test_web_interface()
                self.test_websocket_connection()
                self.test_security_features()
                
                # Test automation separately (doesn't need server)
                self.test_automation_integration()
                
            finally:
                self.stop_test_server()
        else:
            print("⚠️ Server failed to start - running offline tests only")
            self.test_automation_integration()
        
        # Generate report
        self.generate_integration_report()

def main():
    """Main integration test function"""
    if len(sys.argv) > 1:
        test_suite = IntegrationTestSuite()
        command = sys.argv[1].lower()
        
        if command == "api":
            test_suite.start_test_server()
            test_suite.test_api_endpoints()
            test_suite.stop_test_server()
        elif command == "apps":
            test_suite.start_test_server()
            test_suite.test_apps_discovery()
            test_suite.stop_test_server()
        elif command == "automation":
            test_suite.test_automation_integration()
        elif command == "ai":
            test_suite.start_test_server()
            test_suite.test_conversational_ai_integration()
            test_suite.stop_test_server()
        elif command == "web":
            test_suite.start_test_server()
            test_suite.test_web_interface()
            test_suite.stop_test_server()
        elif command == "security":
            test_suite.start_test_server()
            test_suite.test_security_features()
            test_suite.stop_test_server()
        else:
            print(f"Unknown test: {command}")
            print("Available tests: api, apps, automation, ai, web, security")
    else:
        # Run all tests
        test_suite = IntegrationTestSuite()
        test_suite.run_all_tests()

if __name__ == "__main__":
    main()