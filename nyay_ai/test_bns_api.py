#!/usr/bin/env python3
"""
Test script for BNS-Constitution API endpoints

Usage:
    python test_bns_api.py
"""

import requests
import json
from tabulate import tabulate
import sys

BASE_URL = "http://localhost:5001/api/bns-constitution"

def test_all_sections():
    """Test getting all BNS sections"""
    print("\n" + "="*60)
    print("TEST 1: Get All BNS Sections")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/all")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Total sections: {data['total_sections']}")
            print(f"   Sections: {[s['bns_id'] for s in data['sections'][:5]]}...")
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_specific_section():
    """Test getting specific section"""
    print("\n" + "="*60)
    print("TEST 2: Get Specific BNS Section (Murder - 101)")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/section/101")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found: {data['title']}")
            print(f"   Punishment: {data['punishment']['imprisonment']}")
            print(f"   Constitution Articles: {len(data['constitution_articles'])} found")
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_victim_rights():
    """Test victim rights endpoint"""
    print("\n" + "="*60)
    print("TEST 3: Get Victim Rights (Rape - 64)")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/victim-rights/64")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found rights for: {data['title']}")
            print(f"   Total rights: {len(data['victim_rights'])}")
            
            # Display first 3 rights
            print("\n   Sample Rights:")
            for i, right in enumerate(data['victim_rights'][:3], 1):
                print(f"   {i}. {right}")
            
            print(f"\n   Support Organizations: {len(data['support_organizations'])}")
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_remedies():
    """Test remedies endpoint"""
    print("\n" + "="*60)
    print("TEST 4: Get Remedies (Dowry Death - 84)")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/remedies/84")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found remedies for: {data['title']}")
            print(f"   Total remedies: {len(data['remedies_and_paths'])}")
            
            # Display remedies
            for remedy in data['remedies_and_paths']:
                print(f"\n   Path {remedy['path_number']}: {remedy['name']}")
                print(f"   Expected time: {remedy.get('time_expected', 'N/A')}")
                print(f"   Steps: {len(remedy.get('steps', []))} steps")
            
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_search():
    """Test search endpoint"""
    print("\n" + "="*60)
    print("TEST 5: Search (keyword='harassment')")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/search", params={'keyword': 'harassment'})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {data['results_count']} results")
            
            for result in data['results']:
                print(f"   • BNS {result['bns_id']}: {result['title']}")
            
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_support_services():
    """Test support services endpoint"""
    print("\n" + "="*60)
    print("TEST 6: Get Support Services & Helplines")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/support-services")
        if response.status_code == 200:
            data = response.json()
            services = data['services']
            
            print(f"✅ Found {len(services)} support services")
            
            # Create table
            table_data = []
            for service in services[:5]:
                table_data.append([
                    service['name'],
                    service['number'],
                    service.get('availability', 'N/A')
                ])
            
            print("\n   Top 5 Services:")
            print(tabulate(
                table_data,
                headers=['Name', 'Number', 'Availability'],
                tablefmt='grid'
            ))
            
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_constitution_articles():
    """Test constitution articles endpoint"""
    print("\n" + "="*60)
    print("TEST 7: Get Constitutional Framework")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/constitution-articles")
        if response.status_code == 200:
            data = response.json()
            
            fundamental_rights = data.get('fundamental_rights', [])
            directive_principles = data.get('directive_principles', [])
            
            print(f"✅ Constitutional framework loaded")
            print(f"   Fundamental Rights: {len(fundamental_rights)} articles")
            print(f"   Directive Principles: {len(directive_principles)} articles")
            
            print("\n   Sample Fundamental Rights:")
            for article in fundamental_rights[:3]:
                print(f"   • Article {article['article']}: {article['title']}")
            
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + " "*10 + "BNS-CONSTITUTION API TEST SUITE" + " "*16 + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    tests = [
        test_all_sections,
        test_specific_section,
        test_victim_rights,
        test_remedies,
        test_search,
        test_support_services,
        test_constitution_articles
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except KeyboardInterrupt:
            print("\n\n⚠️  Tests interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total-passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total-passed} test(s) failed")
        return 1


if __name__ == '__main__':
    print("\n⏳ Testing BNS-Constitution API endpoints...")
    print("   Make sure the app is running: python app.py")
    
    try:
        # Quick connectivity check
        response = requests.get(f"{BASE_URL}/all", timeout=3)
        if response.status_code != 200:
            print("\n❌ API not responding correctly")
            print("   Make sure: python app.py is running on http://localhost:5001")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API")
        print("   Make sure: python app.py is running on http://localhost:5001")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Connection error: {e}")
        sys.exit(1)
    
    # Run tests
    exit_code = run_all_tests()
    sys.exit(exit_code)
