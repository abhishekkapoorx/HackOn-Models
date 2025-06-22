#!/usr/bin/env python3
"""
Test script for enhanced counterfeit detection system
"""

import asyncio
import os
import sys
from enhanced_counterfeit_detector_1 import (
    EnhancedCounterfeitDetector, 
    analyze_product_image,
    load_image
)

async def test_enhanced_detection():
    """Test the enhanced counterfeit detection system"""
    
    print("Enhanced Counterfeit Detection System Test")
    print("=" * 50)
    
    # Test images and descriptions
    test_cases = [
        {
            "image_path": "./archive/selected_for_segmentation/2607.jpg",
            "description": "Nike sports shoes",
            "expected_brand": "nike"
        },
        {
            "image_path": "./archive/selected_for_segmentation/2605.jpg", 
            "description": "Adidas running shoes",
            "expected_brand": "adidas"
        },
        {
            "image_path": "./public/image1.jpg",
            "description": "Brand logo product",
            "expected_brand": "unknown"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {test_case['description']}")
        print("-" * 30)
        
        image_path = test_case["image_path"]
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            continue
        
        try:
            # Load image first to check
            image = load_image(image_path)
            if image is None:
                print(f"❌ Failed to load image: {image_path}")
                continue
            
            print(f"✅ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
            
            # Run enhanced analysis
            print("🔍 Starting enhanced analysis...")
            result = await analyze_product_image(
                image_path, 
                test_case["description"],
                f"output/test_case_{i+1}"
            )
            
            if result:
                print("📊 Analysis Results:")
                print(f"   Brand Detected: {result.brand_detected}")
                print(f"   Counterfeit: {'YES' if result.is_counterfeit else 'NO'}")
                print(f"   Confidence: {result.confidence:.3f}")
                print(f"   Input Logos: {len(result.input_logos)}")
                print(f"   Reference Images: {result.analysis_details['num_reference_images']}")
                
                if result.logo_similarities:
                    print(f"   Max Logo Similarity: {max(result.logo_similarities):.3f}")
                    print(f"   Avg Logo Similarity: {result.analysis_details['avg_logo_similarity']:.3f}")
                
                if result.distortion_scores:
                    print(f"   Avg Distortion Score: {result.analysis_details['avg_distortion_score']:.3f}")
                
                if result.detected_issues:
                    print("⚠️  Issues Detected:")
                    for issue in result.detected_issues[:3]:  # Show first 3 issues
                        print(f"     - {issue}")
                    if len(result.detected_issues) > 3:
                        print(f"     ... and {len(result.detected_issues) - 3} more")
                else:
                    print("✅ No issues detected")
                
                print(f"📁 Output saved to: output/test_case_{i+1}/")
                
            else:
                print("❌ Analysis failed")
                
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()

async def test_individual_components():
    """Test individual components of the system"""
    
    print("\n" + "=" * 50)
    print("Component Testing")
    print("=" * 50)
    
    # Test YOLO model loading
    print("\n🔧 Testing YOLO Model...")
    try:
        from enhanced_counterfeit_detector_1 import EnhancedLogoDetector
        detector = EnhancedLogoDetector("m1.pt")
        if detector.model is not None:
            print("✅ YOLO model loaded successfully")
        else:
            print("❌ YOLO model failed to load")
    except Exception as e:
        print(f"❌ YOLO model error: {e}")
    
    # Test brand detection
    print("\n🔧 Testing Brand Detection...")
    try:
        from enhanced_counterfeit_detector_1 import BrandDetector
        brand_detector = BrandDetector()
        if brand_detector.model is not None:
            print("✅ Brand detection model loaded successfully")
        else:
            print("❌ Brand detection model failed to load")
    except Exception as e:
        print(f"❌ Brand detection error: {e}")
    
    # Test web search
    print("\n🔧 Testing Web Search...")
    try:
        from enhanced_counterfeit_detector_1 import WebImageSearcher
        searcher = WebImageSearcher()
        
        # Test search for a known brand
        urls = await searcher.search_brand_images("nike", "shoes")
        print(f"✅ Found {len(urls)} URLs for Nike shoes")
        
        if urls:
            print("   Sample URLs:")
            for i, url in enumerate(urls[:2]):
                print(f"     {i+1}. {url[:80]}...")
        
    except Exception as e:
        print(f"❌ Web search error: {e}")

async def test_logo_comparison():
    """Test logo comparison functionality"""
    
    print("\n🔧 Testing Logo Comparison...")
    
    test_image = "./archive/selected_for_segmentation/2607.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    try:
        from enhanced_counterfeit_detector_1 import EnhancedLogoDetector, LogoComparator
        
        # Load image
        image = load_image(test_image)
        if image is None:
            print("❌ Failed to load test image")
            return
        
        # Detect logos
        logo_detector = EnhancedLogoDetector("m1.pt")
        logos = logo_detector.detect_logos(image)
        
        print(f"✅ Detected {len(logos)} logos in test image")
        
        for i, logo in enumerate(logos):
            print(f"   Logo {i+1}: bbox={logo.bbox}, confidence={logo.confidence:.3f}")
            print(f"            distortion={logo.distortion_score:.3f}, quality={logo.quality_score:.3f}")
        
        # Test comparison with itself (should be high similarity)
        if len(logos) >= 2:
            comparator = LogoComparator()
            comparison = comparator.compare_logos([logos[0]], [logos[1]])
            print(f"✅ Logo comparison test: similarity={comparison['max_similarity']:.3f}")
        
    except Exception as e:
        print(f"❌ Logo comparison error: {e}")

def print_system_info():
    """Print system information"""
    print("System Information")
    print("=" * 50)
    
    import torch
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print()

async def main():
    """Main test function"""
    print_system_info()
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Run tests
    await test_individual_components()
    await test_logo_comparison()
    await test_enhanced_detection()
    
    print("\n" + "=" * 50)
    print("Testing Complete!")
    print("Check the 'output' directory for results.")

if __name__ == "__main__":
    asyncio.run(main()) 