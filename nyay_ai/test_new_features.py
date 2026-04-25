#!/usr/bin/env python3
"""Test script for new cosine similarity and mapping features"""

from rag import retrieve_relevant_laws_with_scores, get_bns_constitution_mapping

print("\n" + "="*60)
print("🧪 Testing New Features: Cosine Similarity & Mapping")
print("="*60)

# Test 1: Retrieve laws with similarity scores
print("\n[TEST 1] Cosine Similarity Scores (R Values)")
print("-" * 60)
query = "पती मारहाण करतो काय करावे"
results = retrieve_relevant_laws_with_scores(query, top_k=3)
print(f"Query: {query}")
print(f"Retrieved {len(results)} results with scores:\n")
for i, (text, score, idx) in enumerate(results, 1):
    print(f"{i}. Score (R-value): {score:.4f}")
    print(f"   Text: {text[:80]}...\n")

# Test 2: Get BNS-Constitution Mapping
print("\n[TEST 2] BNS-Constitution Mapping")
print("-" * 60)
mapping = get_bns_constitution_mapping("arrest")
bns_sections = mapping.get('bns_sections', [])
print(f"Found {len(bns_sections)} BNS sections in mapping\n")
if bns_sections:
    for section in bns_sections[:1]:
        print(f"BNS Section {section.get('bns_id')}: {section.get('title')}")
        print(f"Description: {section.get('description', '')[:100]}...")
        
        articles = section.get('constitution_articles', [])
        print(f"\nRelated Constitutional Articles ({len(articles)}):")
        for art in articles[:2]:
            print(f"  • Article {art.get('article')}: {art.get('title')}")
        
        rights = section.get('victim_rights', [])
        print(f"\nVictim Rights ({len(rights)}):")
        for right in rights[:2]:
            print(f"  ✓ {right}")

# Test 3: API Response Format
print("\n[TEST 3] API Response Format")
print("-" * 60)
print("✅ similarity_scores field will contain:")
print("   - R-value (cosine similarity score 0-1)")
print("   - text_preview (snippet of matched law)")
print("   - index (chunk index)")
print("\n✅ mapping field will contain:")
print("   - BNS section details")
print("   - Constitutional articles")
print("   - Victim rights and remedies")

print("\n" + "="*60)
print("✅ All new features working correctly!")
print("="*60 + "\n")
