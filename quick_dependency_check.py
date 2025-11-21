#!/usr/bin/env python3
"""
Quick Dependency and Feature Analysis for AI Assistant
"""

import subprocess
import sys
import os
from pathlib import Path

def check_critical_dependencies():
    """Check for critical missing dependencies"""
    
    # List of critical packages and their purposes
    critical_packages = {
        'speech_recognition': 'Voice input functionality',
        'pyttsx3': 'Text-to-speech output',
        'opencv-python': 'Computer vision and image processing',
        'Pillow': 'Image processing and manipulation',
        'flask': 'Web interface backend',
        'flask-socketio': 'Real-time web communication',
        'requests': 'HTTP requests and API calls',
        'numpy': 'Numerical computations',
        'transformers': 'AI model integration',
        'openai': 'OpenAI API integration',
        'pyautogui': 'Computer automation',
        'selenium': 'Web automation',
        'beautifulsoup4': 'Web scraping',
        'edge-tts': 'Enhanced text-to-speech',
        'deep-translator': 'Translation services',
        'psutil': 'System monitoring',
        'pywin32': 'Windows system integration'
    }
    
    # Optional packages for advanced features
    optional_packages = {
        'torch': 'PyTorch for deep learning',
        'tensorflow': 'TensorFlow for machine learning',
        'scikit-learn': 'Machine learning algorithms',
        'matplotlib': 'Data visualization',
        'pandas': 'Data analysis',
        'langchain': 'LLM framework',
        'chromadb': 'Vector database',
        'faiss-cpu': 'Similarity search',
        'whisper': 'Advanced speech recognition',
        'customtkinter': 'Modern GUI framework',
        'sounddevice': 'Audio processing',
        'librosa': 'Audio analysis',
        'spacy': 'Natural language processing',
        'vosk': 'Offline speech recognition'
    }
    
    print("🔍 CRITICAL DEPENDENCY CHECK")
    print("=" * 50)
    
    # Check installed packages
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True, check=True)
        installed = {line.split()[0].lower().replace('_', '-') for line in result.stdout.strip().split('\n')[2:] if line.strip()}
    except:
        print("❌ Could not check installed packages")
        return
    
    # Check critical packages
    missing_critical = []
    available_critical = []
    
    print("\n📦 CRITICAL PACKAGES:")
    for package, purpose in critical_packages.items():
        package_name = package.lower().replace('_', '-')
        if package_name in installed:
            print(f"   ✅ {package:<20} - {purpose}")
            available_critical.append(package)
        else:
            print(f"   ❌ {package:<20} - {purpose}")
            missing_critical.append(package)
    
    # Check optional packages
    missing_optional = []
    available_optional = []
    
    print(f"\n🎯 OPTIONAL PACKAGES:")
    for package, purpose in optional_packages.items():
        package_name = package.lower().replace('_', '-')
        if package_name in installed:
            print(f"   ✅ {package:<20} - {purpose}")
            available_optional.append(package)
        else:
            print(f"   ⚠️  {package:<20} - {purpose}")
            missing_optional.append(package)
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"   Critical Available:  {len(available_critical)}/{len(critical_packages)}")
    print(f"   Optional Available:  {len(available_optional)}/{len(optional_packages)}")
    print(f"   Missing Critical:    {len(missing_critical)}")
    print(f"   Missing Optional:    {len(missing_optional)}")
    
    # Installation recommendations
    if missing_critical:
        print(f"\n🚨 CRITICAL MISSING PACKAGES:")
        print(f"   Install with: pip install " + " ".join(missing_critical))
    
    if missing_optional:
        print(f"\n💡 OPTIONAL ENHANCEMENTS:")
        print(f"   # For AI/ML features:")
        ml_packages = [p for p in missing_optional if p in ['torch', 'tensorflow', 'scikit-learn', 'transformers']]
        if ml_packages:
            print(f"   pip install " + " ".join(ml_packages))
        
        print(f"   # For audio/speech features:")
        audio_packages = [p for p in missing_optional if p in ['whisper', 'sounddevice', 'librosa', 'vosk']]
        if audio_packages:
            print(f"   pip install " + " ".join(audio_packages))
        
        print(f"   # For advanced UI:")
        ui_packages = [p for p in missing_optional if p in ['customtkinter', 'matplotlib']]
        if ui_packages:
            print(f"   pip install " + " ".join(ui_packages))
    
    return {
        'critical_available': len(available_critical),
        'critical_total': len(critical_packages),
        'optional_available': len(available_optional),
        'optional_total': len(optional_packages),
        'missing_critical': missing_critical,
        'missing_optional': missing_optional
    }

def check_feature_status():
    """Check the status of major features"""
    
    features = {
        'Voice Recognition': {
            'files': ['modules/advanced_speech_recognizer.py'],
            'packages': ['speech_recognition', 'pyaudio'],
            'status': 'unknown'
        },
        'Text-to-Speech': {
            'files': ['modules/advanced_voice.py', 'modules/neural_voice_engine.py'],
            'packages': ['pyttsx3', 'edge-tts'],
            'status': 'unknown'
        },
        'Web Interface': {
            'files': ['backend.py', 'modern_web_backend.py'],
            'packages': ['flask', 'flask-socketio'],
            'status': 'unknown'
        },
        'GUI Interface': {
            'files': ['yourdaddy_app.py'],
            'packages': ['tkinter'],
            'status': 'unknown'
        },
        'Computer Vision': {
            'files': ['modules/document_ocr.py', 'modules/multimodal.py'],
            'packages': ['opencv-python', 'Pillow'],
            'status': 'unknown'
        },
        'Automation': {
            'files': ['modules/smart_automation.py', 'automation_tools_new.py'],
            'packages': ['pyautogui', 'selenium'],
            'status': 'unknown'
        },
        'LLM Integration': {
            'files': ['modules/llm_provider.py', 'modules/network_aware_llm.py'],
            'packages': ['openai', 'transformers'],
            'status': 'unknown'
        },
        'Offline Mode': {
            'files': ['modules/offline_llm_provider.py', 'modules/offline_mode.py'],
            'packages': ['transformers', 'torch'],
            'status': 'unknown'
        }
    }
    
    project_root = Path('.')
    
    print(f"\n🎯 FEATURE STATUS CHECK")
    print("=" * 50)
    
    # Check installed packages
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True, check=True)
        installed = {line.split()[0].lower().replace('_', '-') for line in result.stdout.strip().split('\n')[2:] if line.strip()}
    except:
        print("❌ Could not check installed packages")
        return
    
    for feature_name, feature_info in features.items():
        # Check if files exist
        files_exist = all((project_root / file).exists() for file in feature_info['files'])
        
        # Check if packages are installed
        packages_installed = all(pkg.lower().replace('_', '-') in installed for pkg in feature_info['packages'])
        
        if files_exist and packages_installed:
            status = "✅ Ready"
        elif files_exist:
            status = "⚠️ Files exist, missing packages"
        elif packages_installed:
            status = "⚠️ Packages installed, missing files"
        else:
            status = "❌ Not available"
        
        print(f"   {feature_name:<20} - {status}")
        
        # Show details for problematic features
        if not files_exist or not packages_installed:
            if not files_exist:
                missing_files = [f for f in feature_info['files'] if not (project_root / f).exists()]
                print(f"      Missing files: {', '.join(missing_files)}")
            if not packages_installed:
                missing_packages = [pkg for pkg in feature_info['packages'] 
                                  if pkg.lower().replace('_', '-') not in installed]
                print(f"      Missing packages: {', '.join(missing_packages)}")

def main():
    """Run the analysis"""
    print("🤖 AI ASSISTANT - QUICK DEPENDENCY & FEATURE ANALYSIS")
    print("=" * 70)
    
    # Check dependencies
    dep_result = check_critical_dependencies()
    
    # Check features
    check_feature_status()
    
    # Final recommendations
    print(f"\n💡 QUICK FIX RECOMMENDATIONS:")
    if dep_result and dep_result['missing_critical']:
        print(f"   1. 🚨 Install critical packages first:")
        print(f"      pip install " + " ".join(dep_result['missing_critical']))
    
    print(f"   2. 🔧 Test basic functionality:")
    print(f"      python test_llm_connection.py")
    print(f"      python app.py check")
    
    print(f"   3. 🌐 Start the application:")
    print(f"      python app.py web      # Web interface")
    print(f"      python app.py gui      # Desktop interface")
    
    print("=" * 70)

if __name__ == "__main__":
    main()