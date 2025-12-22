"""
Test Enhanced App Discovery - Start Menu & Programs & Features
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_app_discovery():
    """Test the enhanced app discovery implementation"""
    
    print("=" * 70)
    print("🧪 TESTING ENHANCED APP DISCOVERY")
    print("=" * 70)
    print()
    
    try:
        # Import the app discovery module
        from ai_assistant.modules.app_discovery import app_discovery
        
        print("✅ Module imported successfully\n")
        
        # Test 1: Check if cache exists
        print("📦 Checking existing cache...")
        existing_apps = app_discovery.get_all_apps()
        print(f"   Current cache: {len(existing_apps)} apps\n")
        
        # Test 2: Run full scan
        print("🔍 Running full system scan...\n")
        discovered_apps = app_discovery.scan_installed_applications()
        
        # Test 3: Display results
        print("\n" + "=" * 70)
        print("📊 DISCOVERY RESULTS")
        print("=" * 70)
        print(f"Total apps discovered: {len(discovered_apps)}")
        
        # Sample apps
        print("\n📱 Sample discovered apps (first 20):")
        print("-" * 70)
        for i, (name, path) in enumerate(sorted(discovered_apps.items())[:20], 1):
            # Truncate long paths
            display_path = path if len(path) < 50 else path[:47] + "..."
            print(f"{i:2}. {name[:35]:<35} → {display_path}")
        
        # Test 4: Category breakdown
        print("\n📂 App Categories:")
        print("-" * 70)
        
        categories = {
            'Registry Apps': sum(1 for p in discovered_apps.values() if p.endswith('.exe') and 'Program Files' in p),
            'Start Menu Apps': sum(1 for p in discovered_apps.values() if p.endswith('.lnk')),
            'Store Apps': sum(1 for p in discovered_apps.values() if ':' in p and not '\\' in p),
            'System Utils': sum(1 for n, p in discovered_apps.items() if n in ['notepad', 'calculator', 'paint', 'cmd']),
            'Other': 0
        }
        categories['Other'] = len(discovered_apps) - sum(categories.values())
        
        for category, count in categories.items():
            percentage = (count / len(discovered_apps) * 100) if discovered_apps else 0
            print(f"  {category:<20}: {count:4} apps ({percentage:5.1f}%)")
        
        # Test 5: Search functionality
        print("\n🔎 Testing search functionality...")
        test_searches = ['chrome', 'notepad', 'word', 'calculator']
        
        for query in test_searches:
            matches = app_discovery.search_apps(query, limit=3)
            if matches:
                print(f"\n  Search '{query}':")
                for name, path, score in matches:
                    print(f"    ✓ {name} (score: {score})")
            else:
                print(f"\n  Search '{query}': No matches")
        
        # Test 6: Verify cache was saved
        print("\n💾 Verifying cache...")
        cache_file = app_discovery.apps_cache_file
        if os.path.exists(cache_file):
            file_size = os.path.getsize(cache_file)
            print(f"   ✅ Cache saved: {cache_file}")
            print(f"   📦 File size: {file_size:,} bytes")
        else:
            print(f"   ⚠️  Cache not found: {cache_file}")
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_app_discovery()
    sys.exit(0 if success else 1)
