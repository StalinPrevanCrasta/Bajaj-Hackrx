#!/usr/bin/env python3
"""
Test script to demonstrate JSON output functionality
"""

import sys
import os
import json
from pathlib import Path

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def test_json_output():
    """Test the JSON output functionality"""
    print("🧪 Testing JSON Output Functionality")
    print("=" * 50)
    
    try:
        # Import the optimized RAG system
        from src.optimized_rag_system import OptimizedPolicyExpertRAG
        
        # Initialize the system
        print("🚀 Initializing optimized RAG system...")
        rag = OptimizedPolicyExpertRAG()
        
        # Test queries
        test_queries = [
            "What is covered under mental illness treatment?",
            "How do I make a cashless claim?"
        ]
        
        print("\n📋 Testing Individual JSON Responses:")
        print("-" * 40)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: {query}")
            
            # Get JSON response
            json_response = rag.fast_query(query, return_json=True)
            
            print("📊 JSON Output:")
            print(json.dumps(json_response, indent=2, default=str))
            
            # Show key fields
            print(f"✅ Status: {json_response.get('status', 'unknown')}")
            print(f"⏱️  Response Time: {json_response.get('response_time', 0):.2f}s")
            print(f"🎯 Confidence: {json_response.get('confidence', 0):.1f}")
            print(f"📄 Sources: {len(json_response.get('sources', []))}")
        
        print("\n📦 Testing Batch JSON Response:")
        print("-" * 40)
        
        batch_result = rag.batch_query(test_queries, return_json=True)
        print("📊 Batch JSON Output:")
        print(json.dumps(batch_result, indent=2, default=str))
        
        print(f"\n✅ All JSON tests completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during JSON testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_json_structure():
    """Show the expected JSON structure"""
    print("\n📋 Expected JSON Response Structure:")
    print("=" * 50)
    
    example_response = {
        "status": "success",
        "answer": "Mental illness treatment is covered under the policy with specific terms and conditions.",
        "query": "What is covered under mental illness treatment?",
        "sources": ["ICIHLIP22012V012223.txt"],
        "confidence": 0.9,
        "response_time": 2.45,
        "cached": False
    }
    
    print("🔧 Single Query Response:")
    print(json.dumps(example_response, indent=2))
    
    example_batch = {
        "status": "success",
        "total_queries": 2,
        "total_time": 5.23,
        "average_time": 2.61,
        "results": [
            {
                "query_index": 1,
                "question": "What is covered under mental illness treatment?",
                "response": example_response
            }
        ],
        "timestamp": 1706300000.0
    }
    
    print("\n🔧 Batch Response:")
    print(json.dumps(example_batch, indent=2))

if __name__ == "__main__":
    # Show expected structure first
    demonstrate_json_structure()
    
    # Run actual tests
    success = test_json_output()
    
    if success:
        print("\n🎉 JSON output functionality is working correctly!")
    else:
        print("\n⚠️ JSON output testing encountered issues.")
