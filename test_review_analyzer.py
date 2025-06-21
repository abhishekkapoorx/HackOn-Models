#!/usr/bin/env python3
"""
Test script for the Review Analyzer Lambda function
"""

import sys
import json
import time
from aws_lambda_review_analyzer import lambda_handler

def test_lambda_function():
    """Test the Lambda function with various inputs"""
    
    print("Testing Review Analyzer Lambda Function")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            "name": "Positive Review",
            "text": "This product is absolutely amazing! The quality is outstanding and it works perfectly. I would definitely recommend it to anyone. 5 stars!"
        },
        {
            "name": "Negative Review", 
            "text": "This product is terrible. It broke after just one day of use. Very disappointed with the quality. Would not recommend."
        },
        {
            "name": "Neutral Review",
            "text": "The product is okay. It works as expected but nothing special. Average quality for the price."
        },
        {
            "name": "Potential Fake Review",
            "text": "Amazing product! Best ever! Super great! Buy now! 5 stars! Highly recommend! Perfect! Excellent! Outstanding! Wonderful!"
        },
        {
            "name": "Detailed Authentic Review",
            "text": "I bought this product last month after reading several reviews. The packaging was nice and the product arrived on time. After using it for a few weeks, I can say it meets my expectations. The build quality is decent, though not exceptional. It does what it's supposed to do, but I've seen better products in this price range. Overall, it's a reasonable purchase if you're looking for something functional."
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 30)
        print(f"Input: {test_case['text'][:100]}{'...' if len(test_case['text']) > 100 else ''}")
        
        # Create event
        event = {"text": test_case['text']}
        
        # Measure execution time
        start_time = time.time()
        
        try:
            # Call Lambda function
            result = lambda_handler(event, None)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"Execution Time: {execution_time:.2f} seconds")
            
            # Parse response
            if result['statusCode'] == 200:
                response_body = json.loads(result['body'])
                
                print(f"Status: {response_body['status']}")
                print(f"Fake Probability: {response_body['fake_probability']:.4f}")
                print("Sentiment Probabilities:")
                
                for sentiment, score in response_body['sentiment_probabilities'].items():
                    print(f"  {sentiment.capitalize()}: {score:.4f}")
                
                # Interpretation
                fake_prob = response_body['fake_probability']
                if fake_prob > 0.7:
                    fake_status = "Likely FAKE"
                elif fake_prob > 0.3:
                    fake_status = "Uncertain"
                else:
                    fake_status = "Likely AUTHENTIC"
                
                print(f"Interpretation: {fake_status}")
                
            else:
                print(f"Error: {result}")
                
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print()

def test_error_cases():
    """Test error handling"""
    
    print("Testing Error Cases")
    print("=" * 30)
    
    # Empty text
    print("Test: Empty text")
    result = lambda_handler({"text": ""}, None)
    print(f"Result: {result['statusCode']} - {json.loads(result['body'])['message']}")
    
    # Missing text field
    print("\nTest: Missing text field")
    result = lambda_handler({}, None)
    print(f"Result: {result['statusCode']} - {json.loads(result['body'])['message']}")
    
    # Very long text
    print("\nTest: Very long text")
    long_text = "This is a test. " * 1000  # Very long text
    result = lambda_handler({"text": long_text}, None)
    if result['statusCode'] == 200:
        response_body = json.loads(result['body'])
        print(f"Result: Successfully processed long text")
        print(f"Fake Probability: {response_body['fake_probability']:.4f}")
    else:
        print(f"Error: {result}")

def benchmark_performance():
    """Benchmark the performance of the function"""
    
    print("Performance Benchmark")
    print("=" * 30)
    
    test_text = "This product is really good. I like it a lot and would recommend it to others."
    num_tests = 5
    total_time = 0
    
    print(f"Running {num_tests} iterations...")
    
    for i in range(num_tests):
        start_time = time.time()
        result = lambda_handler({"text": test_text}, None)
        end_time = time.time()
        
        iteration_time = end_time - start_time
        total_time += iteration_time
        
        print(f"Iteration {i+1}: {iteration_time:.2f}s")
    
    avg_time = total_time / num_tests
    print(f"\nAverage execution time: {avg_time:.2f} seconds")

if __name__ == "__main__":
    try:
        # Run all tests
        test_lambda_function()
        test_error_cases()
        benchmark_performance()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        sys.exit(1) 