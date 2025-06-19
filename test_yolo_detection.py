#!/usr/bin/env python3
"""
Test script for YOLO-based logo detection using ultralytics
"""

import cv2
import numpy as np
import torch
from product_image_check import ProductCounterfeitDetector, load_image
import os

def test_yolo_detection():
    """Test YOLO-based logo detection on sample images"""
    
    print("Testing YOLO-based Logo Detection (ultralytics)")
    print("=" * 50)
    
    # Initialize detector with YOLO model
    try:
        detector = ProductCounterfeitDetector("m1.pt")
        print("✓ YOLO model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
        return
    
    # Test image paths
    test_images = [
        "./archive/selected_for_segmentation/1528.jpg",
        "./archive/selected_for_segmentation/1529.jpg",
        "./public/image1.jpg",
        "./public/image2.jpg"
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\nProcessing: {image_path}")
            
            # Load image
            image = load_image(image_path)
            if image is None:
                print(f"  ✗ Failed to load image: {image_path}")
                continue
            
            print(f"  ✓ Image loaded successfully ({image.shape[1]}x{image.shape[0]})")
            
            # Detect logos using YOLO
            try:
                logo_regions = detector.logo_detector.detect_logo_regions(image)
                print(f"  ✓ Logos detected: {len(logo_regions)}")
                
                for i, logo in enumerate(logo_regions):
                    bbox = logo['bbox']
                    confidence = logo['confidence']
                    print(f"    Logo {i+1}: bbox={bbox}, confidence={confidence:.3f}")
                
                # Run full counterfeit detection
                result = detector.detect_counterfeit(image)
                
                print(f"  ✓ Counterfeit: {'YES' if result.is_counterfeit else 'NO'}")
                print(f"  ✓ Confidence: {result.confidence:.3f}")
                print(f"  ✓ Logo Anomaly Score: {result.logo_anomaly_score:.3f}")
                
                if result.detected_issues:
                    print("  ⚠ Issues:")
                    for issue in result.detected_issues:
                        print(f"    - {issue}")
                else:
                    print("  ✓ No issues detected")
                    
            except Exception as e:
                print(f"  ✗ Error during detection: {e}")
        else:
            print(f"\n✗ Image not found: {image_path}")
    
    print("\n" + "=" * 50)
    print("YOLO detection test completed!")

def test_yolo_model_info():
    """Test YOLO model information"""
    print("\nYOLO Model Information")
    print("=" * 30)
    
    try:
        from ultralytics import YOLO
        model = YOLO("m1.pt")
        
        print(f"Model path: {model.ckpt_path}")
        print(f"Model type: {type(model).__name__}")
        
        # Get model info
        if hasattr(model, 'model'):
            print(f"Architecture: {type(model.model).__name__}")
        
        print("✓ Model loaded successfully")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")

if __name__ == "__main__":
    test_yolo_model_info()
    test_yolo_detection() 