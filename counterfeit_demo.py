#!/usr/bin/env python3
"""
Demo script for AI-based Product Counterfeit Detection System
"""

import os
from product_image_check import ProductCounterfeitDetector, load_image, save_result_summary

def main():
    print("🔍 AI-Based Product Counterfeit Detection System")
    print("=" * 60)
    
    # Initialize the detector
    detector = ProductCounterfeitDetector()
    
    # Example 1: Single image detection
    print("\n📸 Example 1: Single Image Detection")
    print("-" * 40)
    
    image_path = "public/image1.jpg"
    reference_url = "https://example.com/original_product.jpg"  # Replace with actual URL
    
    if os.path.exists(image_path):
        print(f"Analyzing image: {image_path}")
        
        # Load image
        image = load_image(image_path)
        if image is not None:
            # Detect counterfeit
            result = detector.detect_counterfeit(image, reference_url)
            
            # Print results
            print(f"\n✅ Analysis Complete!")
            print(f"Counterfeit: {'🔴 YES' if result.is_counterfeit else '🟢 NO'}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Logo Anomaly Score: {result.logo_anomaly_score:.3f}")
            print(f"Image Similarity Score: {result.image_similarity_score:.3f}")
            
            if result.detected_issues:
                print("\n🚨 Issues Detected:")
                for issue in result.detected_issues:
                    print(f"  • {issue}")
            else:
                print("\n✅ No issues detected")
            
            # Visualize results
            print("\n📊 Generating visualization...")
            detector.visualize_results(image, result, "detection_result.png")
            print("Visualization saved as 'detection_result.png'")
        else:
            print(f"❌ Failed to load image: {image_path}")
    else:
        print(f"❌ Image not found: {image_path}")
    
    # Example 2: Batch detection
    print("\n\n📸 Example 2: Batch Detection")
    print("-" * 40)
    
    image_paths = ["public/image1.jpg", "public/image2.jpg"]
    reference_urls = [
        "https://example.com/original1.jpg",
        "https://example.com/original2.jpg"
    ]
    
    # Filter existing images
    existing_images = [path for path in image_paths if os.path.exists(path)]
    
    if existing_images:
        print(f"Processing {len(existing_images)} images...")
        
        # Perform batch detection
        batch_results = detector.batch_detect(existing_images, reference_urls[:len(existing_images)])
        
        # Save results
        save_result_summary(batch_results, "batch_detection_results.txt")
        print("Batch results saved to 'batch_detection_results.txt'")
        
        # Print summary
        counterfeit_count = sum(1 for result in batch_results if result.is_counterfeit)
        print(f"\n📊 Batch Summary:")
        print(f"Total images: {len(batch_results)}")
        print(f"Counterfeit detected: {counterfeit_count}")
        print(f"Authentic: {len(batch_results) - counterfeit_count}")
    else:
        print("❌ No images found for batch processing")
    
    print("\n🎉 Demo completed!")

if __name__ == "__main__":
    main() 