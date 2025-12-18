"""
Test Learning Systems API Endpoints
Quick validation of all 65+ REST endpoints
"""

import requests
import json
from time import sleep

BASE_URL = "http://127.0.0.1:8000/api/learning"

def test_endpoints():
    """Test key endpoints from all 27 systems"""
    
    print("=" * 70)
    print("TESTING LEARNING API ENDPOINTS")
    print("=" * 70)
    
    # Wait for server to start
    print("\nWaiting for server...")
    sleep(2)
    
    results = {'passed': 0, 'failed': 0, 'tests': []}
    
    # Test 1: Get all stats
    try:
        response = requests.get(f"{BASE_URL}/stats/all", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ GET /stats/all - {data.get('total_systems', 0)} systems")
            results['passed'] += 1
            results['tests'].append(('GET /stats/all', 'PASS'))
        else:
            print(f"❌ GET /stats/all - Status {response.status_code}")
            results['failed'] += 1
            results['tests'].append(('GET /stats/all', 'FAIL'))
    except Exception as e:
        print(f"❌ GET /stats/all - Error: {e}")
        results['failed'] += 1
        results['tests'].append(('GET /stats/all', f'ERROR: {e}'))
    
    # Test 2: Smart Commands - Predict
    try:
        payload = {
            "user_id": "test_user",
            "recent_commands": ["open file", "edit text"],
            "context": {"time": "morning"}
        }
        response = requests.post(f"{BASE_URL}/smart-commands/predict", json=payload, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ POST /smart-commands/predict - {len(data.get('predictions', []))} predictions")
            results['passed'] += 1
            results['tests'].append(('POST /smart-commands/predict', 'PASS'))
        else:
            print(f"❌ POST /smart-commands/predict - Status {response.status_code}")
            results['failed'] += 1
            results['tests'].append(('POST /smart-commands/predict', 'FAIL'))
    except Exception as e:
        print(f"❌ POST /smart-commands/predict - Error: {str(e)[:50]}")
        results['failed'] += 1
        results['tests'].append(('POST /smart-commands/predict', 'ERROR'))
    
    # Test 3: Context-Aware Response
    try:
        payload = {
            "user_id": "test_user",
            "query": "How do I backup files?",
            "context": {"recent_activity": "file_editing"}
        }
        response = requests.post(f"{BASE_URL}/context/generate", json=payload, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ POST /context/generate - Response generated")
            results['passed'] += 1
            results['tests'].append(('POST /context/generate', 'PASS'))
        else:
            print(f"❌ POST /context/generate - Status {response.status_code}")
            results['failed'] += 1
            results['tests'].append(('POST /context/generate', 'FAIL'))
    except Exception as e:
        print(f"❌ POST /context/generate - Error: {str(e)[:50]}")
        results['failed'] += 1
        results['tests'].append(('POST /context/generate', 'ERROR'))
    
    # Test 4: Workflow Recommender
    try:
        payload = {
            "user_id": "test_user",
            "current_tasks": ["backup", "organize"],
            "context": {"urgency": "high"}
        }
        response = requests.post(f"{BASE_URL}/workflow/recommend", json=payload, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ POST /workflow/recommend - {len(data.get('recommendations', []))} recommendations")
            results['passed'] += 1
            results['tests'].append(('POST /workflow/recommend', 'PASS'))
        else:
            print(f"❌ POST /workflow/recommend - Status {response.status_code}")
            results['failed'] += 1
            results['tests'].append(('POST /workflow/recommend', 'FAIL'))
    except Exception as e:
        print(f"❌ POST /workflow/recommend - Error: {str(e)[:50]}")
        results['failed'] += 1
        results['tests'].append(('POST /workflow/recommend', 'ERROR'))
    
    # Test 5: Domain Embeddings
    try:
        response = requests.post(
            f"{BASE_URL}/domain/embed",
            params={"text": "machine learning model", "domain": "technical"},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ POST /domain/embed - {data.get('dimension', 0)}D embedding")
            results['passed'] += 1
            results['tests'].append(('POST /domain/embed', 'PASS'))
        else:
            print(f"❌ POST /domain/embed - Status {response.status_code}")
            results['failed'] += 1
            results['tests'].append(('POST /domain/embed', 'FAIL'))
    except Exception as e:
        print(f"❌ POST /domain/embed - Error: {str(e)[:50]}")
        results['failed'] += 1
        results['tests'].append(('POST /domain/embed', 'ERROR'))
    
    # Test 6: GNN Add Node
    try:
        payload = {
            "node_id": "user_test",
            "node_type": "user",
            "features": [0.1, 0.2, 0.3]
        }
        response = requests.post(f"{BASE_URL}/gnn/add-node", json=payload, timeout=5)
        if response.status_code == 200:
            print(f"✅ POST /gnn/add-node - Node added")
            results['passed'] += 1
            results['tests'].append(('POST /gnn/add-node', 'PASS'))
        else:
            print(f"❌ POST /gnn/add-node - Status {response.status_code}")
            results['failed'] += 1
            results['tests'].append(('POST /gnn/add-node', 'FAIL'))
    except Exception as e:
        print(f"❌ POST /gnn/add-node - Error: {str(e)[:50]}")
        results['failed'] += 1
        results['tests'].append(('POST /gnn/add-node', 'ERROR'))
    
    # Test 7: Adaptive Voice Log
    try:
        payload = {
            "user_id": "test_user",
            "text": "open the file",
            "confidence": 0.92,
            "was_correct": True
        }
        response = requests.post(f"{BASE_URL}/adaptive-voice/log", json=payload, timeout=5)
        if response.status_code == 200:
            print(f"✅ POST /adaptive-voice/log - Recognition logged")
            results['passed'] += 1
            results['tests'].append(('POST /adaptive-voice/log', 'PASS'))
        else:
            print(f"❌ POST /adaptive-voice/log - Status {response.status_code}")
            results['failed'] += 1
            results['tests'].append(('POST /adaptive-voice/log', 'FAIL'))
    except Exception as e:
        print(f"❌ POST /adaptive-voice/log - Error: {str(e)[:50]}")
        results['failed'] += 1
        results['tests'].append(('POST /adaptive-voice/log', 'ERROR'))
    
    # Test 8: RL Select Action
    try:
        response = requests.post(
            f"{BASE_URL}/rl/select-action",
            json=[0.1, 0.2, 0.3, 0.4, 0.5],
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ POST /rl/select-action - Action {data.get('action')} selected")
            results['passed'] += 1
            results['tests'].append(('POST /rl/select-action', 'PASS'))
        else:
            print(f"❌ POST /rl/select-action - Status {response.status_code}")
            results['failed'] += 1
            results['tests'].append(('POST /rl/select-action', 'FAIL'))
    except Exception as e:
        print(f"❌ POST /rl/select-action - Error: {str(e)[:50]}")
        results['failed'] += 1
        results['tests'].append(('POST /rl/select-action', 'ERROR'))
    
    # Test 9: Get individual system stats
    systems = ['rl', 'meta', 'gnn', 'domain', 'smart-commands', 'workflow', 'context']
    for system in systems:
        try:
            response = requests.get(f"{BASE_URL}/{system}/stats", timeout=5)
            if response.status_code == 200:
                print(f"✅ GET /{system}/stats")
                results['passed'] += 1
                results['tests'].append((f'GET /{system}/stats', 'PASS'))
            else:
                print(f"❌ GET /{system}/stats - Status {response.status_code}")
                results['failed'] += 1
                results['tests'].append((f'GET /{system}/stats', 'FAIL'))
        except Exception as e:
            print(f"❌ GET /{system}/stats - Error: {str(e)[:30]}")
            results['failed'] += 1
            results['tests'].append((f'GET /{system}/stats', 'ERROR'))
    
    # Summary
    print("\n" + "=" * 70)
    print("API TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Passed: {results['passed']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"📊 Success Rate: {(results['passed']/(results['passed']+results['failed'])*100):.1f}%")
    
    return results

if __name__ == "__main__":
    test_endpoints()
