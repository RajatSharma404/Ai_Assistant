#!/usr/bin/env python3
"""Test WhatsApp parsing fix"""
import re

def test_app_name_extraction():
    """Test that app names like WhatsApp are correctly extracted"""
    
    test_cases = [
        ("open whatsapp", "whatsapp"),
        ("open WhatsApp", "whatsapp"),
        ("launch whatsapp", "whatsapp"),
        ("start whatsapp", "whatsapp"),
        ("open the whatsapp app", "whatsapp"),
        ("open facebook", "facebook"),
        ("open notepad", "notepad"),
        ("open the app store", "store"),  # 'app' should be removed when standalone
        ("open instagram", "instagram"),
    ]
    
    def extract_app_name(query: str) -> str:
        """Extract app name using the fixed logic"""
        app_name = query.lower()
        
        # Remove command words carefully
        for word in ['open ', 'launch ', 'start ', 'run ', ' the ', ' please']:
            app_name = app_name.replace(word, ' ')
        
        # Only remove 'app' if it's standalone, not part of a word
        app_name = re.sub(r'\bapp\b', '', app_name)
        app_name = re.sub(r'\bapplication\b', '', app_name)
        app_name = re.sub(r'\bprogram\b', '', app_name)
        return app_name.strip()
    
    print("Testing app name extraction...")
    print("=" * 60)
    
    all_passed = True
    for input_text, expected in test_cases:
        result = extract_app_name(input_text)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        
        if result != expected:
            all_passed = False
            
        print(f"{status} | Input: '{input_text}'")
        print(f"       Expected: '{expected}' | Got: '{result}'")
        print()
    
    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed!")
    
    return all_passed

if __name__ == "__main__":
    test_app_name_extraction()
