#!/usr/bin/env python3
"""
Test script for the Product Authentication & Review Analysis API
"""

import requests
import base64
import json
import time
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30

def test_health_endpoints():
    """Test health check endpoints"""
    print("🔍 Testing health endpoints...")
    
    try:
        # Test basic health check
        response = requests.get(f"{API_BASE_URL}/", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        print("✅ Basic health check passed")
        
        # Test detailed health check
        response = requests.get(f"{API_BASE_URL}/health", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        health_data = response.json()
        print(f"✅ Detailed health check passed: {health_data}")
        
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_fake_review_detection():
    """Test fake review detection endpoint"""
    print("🕵️ Testing fake review detection...")
    
    try:
        test_reviews = [
            "This product is absolutely amazing! Best purchase ever!",
            "Decent quality for the price, works as expected.",
            "BEST PRODUCT EVER!!! BUY NOW!!! 5 STARS!!!",
            "Good build quality, arrived on time, satisfied with purchase."
        ]
        
        response = requests.post(
            f"{API_BASE_URL}/detect-fake-reviews",
            json={"reviews": test_reviews},
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 200
        result = response.json()
        
        print(f"✅ Fake review detection passed:")
        print(f"   Total reviews: {result['total_reviews']}")
        print(f"   Fake reviews detected: {result['fake_reviews_detected']}")
        print(f"   Fake percentage: {result['fake_review_percentage']:.1f}%")
        
        return True
    except Exception as e:
        print(f"❌ Fake review detection failed: {e}")
        return False

def test_review_analysis():
    """Test review analysis endpoint"""
    print("📝 Testing review analysis...")
    
    try:
        test_reviews = [
            "Great product, excellent quality and fast shipping!",
            "Poor build quality, broke after one week of use.",
            "Amazing comfort and style, worth the price.",
            "Disappointed with the customer service experience.",
            "Perfect fit, exactly what I was looking for!"
        ]
        
        response = requests.post(
            f"{API_BASE_URL}/analyze-reviews",
            json={
                "reviews": test_reviews,
                "product_id": "TEST_PRODUCT_123"
            },
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 200
        result = response.json()
        
        print(f"✅ Review analysis passed:")
        print(f"   Product ID: {result['product_id']}")
        print(f"   Total reviews: {result['total_reviews']}")
        print(f"   Positive aspects: {result['positive_aspects']}")
        print(f"   Negative aspects: {result['negative_aspects']}")
        print(f"   Sentiment distribution: {result['sentiment_distribution']}")
        
        return True
    except Exception as e:
        print(f"❌ Review analysis failed: {e}")
        return False

def create_test_image_base64():
    """Create a simple test image in base64 format"""
    # Create a simple 100x100 red square image
    from PIL import Image
    import io
    
    # Create a red square
    img = Image.new('RGB', (100, 100), color='red')
    
    # Convert to base64
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    
    return img_base64

def test_counterfeit_detection():
    """Test counterfeit detection endpoint"""
    print("🔍 Testing counterfeit detection...")
    
    try:
        # Create a test image
        test_image_b64 = create_test_image_base64()
        
        response = requests.post(
            f"{API_BASE_URL}/detect-counterfeit",
            json={
                "image_base64": test_image_b64,
                "product_description": "Nike Air Max shoes"
            },
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 200
        result = response.json()
        
        print(f"✅ Counterfeit detection passed:")
        print(f"   Is counterfeit: {result['is_counterfeit']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Brand detected: {result['brand_detected']}")
        print(f"   Logo count: {result['logo_count']}")
        print(f"   Detected issues: {result['detected_issues']}")
        
        return True
    except Exception as e:
        print(f"❌ Counterfeit detection failed: {e}")
        return False

def test_batch_analysis():
    """Test batch analysis endpoint"""
    print("🔄 Testing batch analysis...")
    
    try:
        test_image_b64 = create_test_image_base64()
        test_reviews = [
            "Great product, excellent quality!",
            "Poor build quality, broke quickly.",
            "Amazing comfort and style!"
        ]
        
        response = requests.post(
            f"{API_BASE_URL}/batch-analysis",
            json={
                "image_base64": test_image_b64,
                "product_description": "Nike Air Max shoes",
                "reviews": test_reviews,
                "product_id": "TEST_BATCH_123"
            },
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 200
        result = response.json()
        
        print(f"✅ Batch analysis passed:")
        print(f"   Counterfeit analysis: {result['counterfeit_analysis']['is_counterfeit']}")
        print(f"   Review analysis: {result['review_analysis']['total_reviews']} reviews")
        print(f"   Fake review analysis: {result['fake_review_analysis']['fake_reviews_detected']} fake reviews")
        
        return True
    except Exception as e:
        print(f"❌ Batch analysis failed: {e}")
        return False

def test_api_documentation():
    """Test API documentation endpoints"""
    print("📚 Testing API documentation...")
    
    try:
        # Test OpenAPI JSON
        response = requests.get(f"{API_BASE_URL}/openapi.json", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        print("✅ OpenAPI JSON endpoint accessible")
        
        # Test Swagger UI
        response = requests.get(f"{API_BASE_URL}/docs", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        print("✅ Swagger UI accessible")
        
        # Test ReDoc
        response = requests.get(f"{API_BASE_URL}/redoc", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        print("✅ ReDoc accessible")
        
        return True
    except Exception as e:
        print(f"❌ API documentation test failed: {e}")
        return False

def run_all_tests():
    """Run all API tests"""
    print("🚀 Starting API tests...")
    print("=" * 60)
    
    tests = [
        ("Health Endpoints", test_health_endpoints),
        ("API Documentation", test_api_documentation),
        ("Fake Review Detection", test_fake_review_detection),
        ("Review Analysis", test_review_analysis),
        ("Counterfeit Detection", test_counterfeit_detection),
        ("Batch Analysis", test_batch_analysis),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 40)
        start_time = time.time()
        
        try:
            success = test_func()
            end_time = time.time()
            duration = end_time - start_time
            results[test_name] = {
                "success": success,
                "duration": duration
            }
            print(f"⏱️  Duration: {duration:.2f}s")
        except Exception as e:
            results[test_name] = {
                "success": False,
                "duration": 0,
                "error": str(e)
            }
            print(f"❌ Unexpected error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(tests)
    passed_tests = sum(1 for r in results.values() if r["success"])
    
    for test_name, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        duration = result["duration"]
        print(f"{status} {test_name:<25} ({duration:.2f}s)")
        
        if not result["success"] and "error" in result:
            print(f"    Error: {result['error']}")
    
    print("-" * 60)
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    import sys
    
    # Check if PIL is available for image tests
    try:
        from PIL import Image
    except ImportError:
        print("⚠️  PIL (Pillow) not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        from PIL import Image
    
    print("🧪 Product Authentication API Test Suite")
    print(f"🌐 Testing API at: {API_BASE_URL}")
    
    # Wait for API to be ready
    print("⏳ Waiting for API to be ready...")
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_BASE_URL}/", timeout=5)
            if response.status_code == 200:
                print("✅ API is ready!")
                break
        except:
            if i == max_retries - 1:
                print("❌ API is not responding. Make sure it's running.")
                sys.exit(1)
            print(f"⏳ Retrying... ({i+1}/{max_retries})")
            time.sleep(2)
    
    # Run tests
    success = run_all_tests()
    
    if not success:
        sys.exit(1) 