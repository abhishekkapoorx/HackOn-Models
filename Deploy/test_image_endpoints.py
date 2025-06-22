#!/usr/bin/env python3
"""
Test script for demonstrating different image input methods in the API
"""

import requests
import base64
import json
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_base64_endpoint():
    """Test the original base64 endpoint"""
    print("Testing /detect-counterfeit with base64 image...")
    
    # Example image path (adjust as needed)
    image_path = "../public/image1.jpg"
    
    if not Path(image_path).exists():
        print(f"Image file not found: {image_path}")
        return
    
    # Encode image to base64
    image_b64 = encode_image_to_base64(image_path)
    
    # Prepare request
    payload = {
        "image_base64": image_b64,
        "product_description": "Nike shoes for authenticity check"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/detect-counterfeit", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def test_file_upload_endpoint():
    """Test the file upload endpoint"""
    print("\nTesting /upload-image-counterfeit with file upload...")
    
    image_path = "../public/image1.jpg"
    
    if not Path(image_path).exists():
        print(f"Image file not found: {image_path}")
        return
    
    try:
        with open(image_path, "rb") as image_file:
            files = {"file": ("image1.jpg", image_file, "image/jpeg")}
            data = {"product_description": "Nike shoes for authenticity check"}
            
            response = requests.post(f"{BASE_URL}/upload-image-counterfeit", files=files, data=data)
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def test_unified_endpoint_with_file():
    """Test the unified endpoint with file upload"""
    print("\nTesting /detect-counterfeit-unified with file upload...")
    
    image_path = "../public/image1.jpg"
    
    if not Path(image_path).exists():
        print(f"Image file not found: {image_path}")
        return
    
    try:
        with open(image_path, "rb") as image_file:
            files = {"file": ("image1.jpg", image_file, "image/jpeg")}
            data = {"product_description": "Nike shoes for authenticity check"}
            
            response = requests.post(f"{BASE_URL}/detect-counterfeit-unified", files=files, data=data)
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def test_unified_endpoint_with_base64():
    """Test the unified endpoint with base64 image"""
    print("\nTesting /detect-counterfeit-unified with base64 image...")
    
    image_path = "../public/image1.jpg"
    
    if not Path(image_path).exists():
        print(f"Image file not found: {image_path}")
        return
    
    try:
        image_b64 = encode_image_to_base64(image_path)
        
        data = {
            "image_base64": image_b64,
            "product_description": "Nike shoes for authenticity check"
        }
        
        response = requests.post(f"{BASE_URL}/detect-counterfeit-unified", data=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def test_batch_analysis_with_file():
    """Test the enhanced batch analysis with file upload"""
    print("\nTesting /batch-analysis-enhanced with file upload...")
    
    image_path = "../public/image1.jpg"
    
    if not Path(image_path).exists():
        print(f"Image file not found: {image_path}")
        return
    
    # Sample reviews
    sample_reviews = [
        "Great product, excellent quality!",
        "Poor quality, looks fake",
        "Amazing shoes, very comfortable",
        "Not as described, disappointed"
    ]
    
    try:
        with open(image_path, "rb") as image_file:
            files = {"file": ("image1.jpg", image_file, "image/jpeg")}
            data = {
                "product_description": "Nike Air Max shoes",
                "product_id": "NIKE-001",
                "reviews": json.dumps(sample_reviews)
            }
            
            response = requests.post(f"{BASE_URL}/batch-analysis-enhanced", files=files, data=data)
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def test_health_check():
    """Test the health check endpoint"""
    print("\nTesting /health endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def test_single_review_analysis():
    """Test the single review analysis endpoint (fake detection + sentiment only)"""
    print("\nTesting /analyze-single-review endpoint...")
    
    sample_review = "This product is absolutely amazing! Great quality and fast shipping. Highly recommended!"
    
    payload = {
        "review": sample_review,
        "product_id": "PROD-001"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/analyze-single-review", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def test_batch_review_analysis():
    """Test the batch review analysis endpoint (full aspect analysis)"""
    print("\nTesting /analyze-reviews endpoint (batch with aspects)...")
    
    sample_reviews = [
        "Great product, excellent quality!",
        "Poor quality, looks fake",
        "Amazing shoes, very comfortable",
        "Not as described, disappointed",
        "Fast shipping and good customer service",
        "The material feels cheap and flimsy"
    ]
    
    payload = {
        "reviews": sample_reviews,
        "product_id": "PROD-001"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/analyze-reviews", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def test_fake_review_detection_only():
    """Test fake review detection endpoint"""
    print("\nTesting /detect-fake-reviews endpoint...")
    
    sample_reviews = [
        "This is the best product ever! Amazing quality!",
        "Terrible product, waste of money",
        "Good value for money, recommended",
        "Perfect perfect perfect! Buy now!"
    ]
    
    payload = {
        "reviews": sample_reviews
    }
    
    try:
        response = requests.post(f"{BASE_URL}/detect-fake-reviews", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def test_single_return_analysis():
    """Test the single return analysis endpoint"""
    print("\nTesting /analyze-single-return endpoint...")
    
    sample_return = "The product quality is poor and it broke after 2 days. Very disappointed with this purchase."
    
    payload = {
        "return_reason": sample_return,
        "product_id": "PROD-001",
        "return_id": "RET-12345"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/analyze-single-return", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def test_batch_return_analysis():
    """Test the batch return analysis endpoint"""
    print("\nTesting /analyze-returns-batch endpoint...")
    
    sample_returns = [
        "Poor quality material, broke after one week",
        "Wrong size, too small even though I ordered large",
        "Item arrived damaged during shipping",
        "Not as described in the listing, completely different product",
        "Great product but wrong color was sent",
        "Delayed shipping, arrived 2 weeks late",
        "Cheap plastic, not worth the money",
        "Size runs small, doesn't fit properly"
    ]
    
    payload = {
        "returns": sample_returns,
        "product_id": "PROD-001"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/analyze-returns-batch", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("=== API Image Input Methods Test ===")
    
    # Test health first
    test_health_check()
    
    # Test different image input methods
    test_base64_endpoint()
    test_file_upload_endpoint()
    test_unified_endpoint_with_file()
    test_unified_endpoint_with_base64()
    test_batch_analysis_with_file()
    
    # Test review analysis methods
    print("\n=== Review Analysis Tests ===")
    test_single_review_analysis()
    test_batch_review_analysis()
    test_fake_review_detection_only()
    
    # Test return analysis methods
    print("\n=== Return Analysis Tests ===")
    test_single_return_analysis()
    test_batch_return_analysis()
    
    print("\n=== Test Complete ===") 