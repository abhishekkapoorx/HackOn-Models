import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from transformers import CLIPProcessor, CLIPModel, AutoFeatureExtractor, AutoModelForImageClassification
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dotenv import load_dotenv
import logging
from ultralytics import YOLO

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Result of counterfeit detection analysis"""
    is_counterfeit: bool
    confidence: float
    logo_anomaly_score: float
    image_similarity_score: float
    detected_issues: List[str]
    logo_regions: List[Dict[str, Any]]
    comparison_details: Dict[str, Any]

class LogoDetector:
    """Detects and analyzes logos in product images using YOLO"""
    
    def __init__(self, model_path: str = "m1.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        
        # Load YOLO model using ultralytics
        try:
            self.model = YOLO(model_path)
            logger.info(f"YOLO model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
        
    def detect_logo_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect logo regions using YOLO model"""
        if self.model is None:
            logger.warning("YOLO model not loaded, returning empty results")
            return []
        
        # Convert BGR to RGB for YOLO
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run YOLO inference using ultralytics
        results = self.model(rgb_image, verbose=False)
        
        logo_regions = []
        confidence_threshold = 0.55
        
        # Process YOLO results
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                if confidence < confidence_threshold:
                    continue
                
                # Convert to integer coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                
                logo_regions.append({
                    'bbox': (x1, y1, w, h),
                    'area': w * h,
                    'confidence': float(confidence),
                    'type': 'yolo_detection',
                    'class_id': class_id
                })
        
        logger.info(f"Detected {len(logo_regions)} logo regions using YOLO (confidence > {confidence_threshold})")
        return logo_regions
    
    def analyze_logo_distortion(self, image: np.ndarray, logo_regions: List[Dict[str, Any]]) -> float:
        """Analyze logo distortion and return anomaly score"""
        if not logo_regions:
            return 0.0
        
        distortion_scores = []
        
        for region in logo_regions:
            x, y, w, h = region['bbox']
            logo_roi = image[y:y+h, x:x+w]
            
            if logo_roi.size == 0:
                continue
            
            # Convert to grayscale for analysis
            gray_roi = cv2.cvtColor(logo_roi, cv2.COLOR_BGR2GRAY)
            
            # Analyze symmetry
            symmetry_score = self._calculate_symmetry(gray_roi)
            
            # Analyze edge consistency
            edge_consistency = self._analyze_edge_consistency(gray_roi)
            
            # Analyze color consistency
            color_consistency = self._analyze_color_consistency(logo_roi)
            
            # Combined distortion score
            distortion_score = (1 - symmetry_score) * 0.4 + (1 - edge_consistency) * 0.3 + (1 - color_consistency) * 0.3
            distortion_scores.append(distortion_score)
        
        return float(np.mean(distortion_scores)) if distortion_scores else 0.0
    
    def _calculate_symmetry(self, gray_image: np.ndarray) -> float:
        """Calculate symmetry score of the image"""
        height, width = gray_image.shape
        
        # Vertical symmetry
        left_half = gray_image[:, :width//2]
        right_half = cv2.flip(gray_image[:, width//2:], 1)
        
        # Ensure same size
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        vertical_symmetry = 1 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255
        
        # Horizontal symmetry
        top_half = gray_image[:height//2, :]
        bottom_half = cv2.flip(gray_image[height//2:, :], 0)
        
        min_height = min(top_half.shape[0], bottom_half.shape[0])
        top_half = top_half[:min_height, :]
        bottom_half = bottom_half[:min_height, :]
        
        horizontal_symmetry = 1 - np.mean(np.abs(top_half.astype(float) - bottom_half.astype(float))) / 255
        
        return float((vertical_symmetry + horizontal_symmetry) / 2)
    
    def _analyze_edge_consistency(self, gray_image: np.ndarray) -> float:
        """Analyze edge consistency in the image"""
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate edge direction consistency
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_direction = np.arctan2(sobely, sobelx)
        
        # Calculate direction consistency
        direction_std = np.std(gradient_direction[gradient_magnitude > np.mean(gradient_magnitude)])
        direction_consistency = 1 / (1 + direction_std)
        
        return float((edge_density + direction_consistency) / 2)
    
    def _analyze_color_consistency(self, color_image: np.ndarray) -> float:
        """Analyze color consistency in the image"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
        
        # Calculate color variance
        hsv_std = np.std(hsv, axis=(0, 1))
        lab_std = np.std(lab, axis=(0, 1))
        
        # Normalize and combine
        color_variance = (np.mean(hsv_std) + np.mean(lab_std)) / 2
        color_consistency = 1 / (1 + color_variance / 50)
        
        return float(color_consistency)

class ImageComparator:
    """Compares product images with original references"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = self.clip_model.to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load a pre-trained model for feature extraction
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
        self.feature_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50").to(self.device)
        
    def download_reference_image(self, url: str) -> Optional[np.ndarray]:
        """Download reference image from URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content))
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Failed to download reference image: {e}")
            return None
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract deep features from image"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Extract features using CLIP
        inputs = self.clip_processor(
            text=None,
            images=pil_image,
            return_tensors="pt"
        )
        
        # Extract pixel_values and move to device
        pixel_values = torch.tensor(inputs["pixel_values"], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            features = self.clip_model.get_image_features(pixel_values=pixel_values)
        
        return features.cpu().numpy().flatten()
    
    def compare_images(self, image1: np.ndarray, image2: np.ndarray) -> Dict[str, float]:
        """Compare two images and return similarity metrics"""
        # Extract features
        features1 = self.extract_features(image1)
        features2 = self.extract_features(image2)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]
        
        # Calculate structural similarity
        structural_sim = self._calculate_structural_similarity(image1, image2)
        
        # Calculate color histogram similarity
        color_sim = self._calculate_color_similarity(image1, image2)
        
        # Calculate texture similarity
        texture_sim = self._calculate_texture_similarity(image1, image2)
        
        return {
            'cosine_similarity': float(similarity),
            'structural_similarity': structural_sim,
            'color_similarity': color_sim,
            'texture_similarity': texture_sim,
            'overall_similarity': (similarity + structural_sim + color_sim + texture_sim) / 4
        }
    
    def _calculate_structural_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate structural similarity index"""
        # Resize images to same size
        height, width = min(img1.shape[:2]), min(img2.shape[:2])
        img1_resized = cv2.resize(img1, (width, height))
        img2_resized = cv2.resize(img2, (width, height))
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM
        from skimage.metrics import structural_similarity as ssim
        try:
            ssim_score = ssim(gray1, gray2)
            return max(0, ssim_score)  # Ensure non-negative
        except:
            return 0.0
    
    def _calculate_color_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate color histogram similarity"""
        # Calculate histograms for each channel
        hist1 = []
        hist2 = []
        
        for i in range(3):  # BGR channels
            hist1.append(cv2.calcHist([img1], [i], None, [256], [0, 256]).flatten())
            hist2.append(cv2.calcHist([img2], [i], None, [256], [0, 256]).flatten())
        
        # Calculate correlation for each channel
        correlations = []
        for h1, h2 in zip(hist1, hist2):
            correlation = np.corrcoef(h1, h2)[0, 1]
            if not np.isnan(correlation):
                correlations.append(correlation)
        
        return float(np.mean(correlations)) if correlations else 0.0
    
    def _calculate_texture_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate texture similarity using GLCM features"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Calculate GLCM features (simplified)
        # Use edge density as a texture measure
        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)
        
        edge_density1 = np.sum(edges1 > 0) / edges1.size
        edge_density2 = np.sum(edges2 > 0) / edges2.size
        
        # Calculate similarity based on edge density difference
        density_diff = abs(edge_density1 - edge_density2)
        texture_similarity = 1 / (1 + density_diff * 10)
        
        return float(texture_similarity)

class ProductCounterfeitDetector:
    """Main class for product counterfeit detection"""
    
    def __init__(self, yolo_model_path: str = "m1.pt"):
        self.logo_detector = LogoDetector(yolo_model_path)
        self.image_comparator = ImageComparator()
        self.thresholds = {
            'logo_distortion': 0.3,  # Higher = more suspicious
            'image_similarity': 0.7,  # Lower = more suspicious
            'overall_confidence': 0.6
        }
    
    def detect_counterfeit(self, 
                          product_image: np.ndarray, 
                          reference_image_url: Optional[str] = None,
                          reference_image: Optional[np.ndarray] = None) -> DetectionResult:
        """
        Detect counterfeit products by analyzing logo distortion and image similarity
        
        Args:
            product_image: Image of the product to check
            reference_image_url: URL of the original product image
            reference_image: Direct reference image array
            
        Returns:
            DetectionResult with analysis details
        """
        detected_issues = []
        
        # Step 1: Logo Analysis using YOLO
        logger.info("Analyzing logo distortion using YOLO...")
        logo_regions = self.logo_detector.detect_logo_regions(product_image)
        logo_anomaly_score = self.logo_detector.analyze_logo_distortion(product_image, logo_regions)
        
        if logo_anomaly_score > self.thresholds['logo_distortion']:
            detected_issues.append(f"Logo distortion detected (score: {logo_anomaly_score:.3f})")
        
        # Step 2: Image Comparison
        image_similarity_score = 1.0
        comparison_details = {}
        
        if reference_image_url or reference_image is not None:
            logger.info("Comparing with reference image...")
            
            if reference_image is None and reference_image_url:
                reference_image = self.image_comparator.download_reference_image(reference_image_url)
            
            if reference_image is not None:
                comparison_details = self.image_comparator.compare_images(product_image, reference_image)
                image_similarity_score = comparison_details['overall_similarity']
                
                if image_similarity_score < self.thresholds['image_similarity']:
                    detected_issues.append(f"Low similarity with reference (score: {image_similarity_score:.3f})")
            else:
                detected_issues.append("Failed to download reference image")
        
        # Step 3: Determine if counterfeit
        is_counterfeit = len(detected_issues) > 0
        confidence = self._calculate_confidence(logo_anomaly_score, image_similarity_score, len(detected_issues))
        
        return DetectionResult(
            is_counterfeit=is_counterfeit,
            confidence=confidence,
            logo_anomaly_score=logo_anomaly_score,
            image_similarity_score=image_similarity_score,
            detected_issues=detected_issues,
            logo_regions=logo_regions,
            comparison_details=comparison_details
        )
    
    def _calculate_confidence(self, logo_score: float, similarity_score: float, issue_count: int) -> float:
        """Calculate overall confidence in the detection"""
        # Weighted combination of factors
        logo_weight = 0.4
        similarity_weight = 0.4
        issue_weight = 0.2
        
        logo_confidence = 1 - logo_score  # Lower distortion = higher confidence
        similarity_confidence = similarity_score
        issue_confidence = 1 / (1 + issue_count)  # Fewer issues = higher confidence
        
        confidence = (logo_confidence * logo_weight + 
                     similarity_confidence * similarity_weight + 
                     issue_confidence * issue_weight)
        
        return min(1.0, max(0.0, confidence))
    
    def visualize_results(self, image: np.ndarray, result: DetectionResult, save_path: Optional[str] = None):
        """Visualize detection results on the image"""
        # Convert BGR to RGB for matplotlib
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image with logo regions
        axes[0].imshow(rgb_image)
        axes[0].set_title('Product Image with YOLO Logo Detection')
        
        for region in result.logo_regions:
            x, y, w, h = region['bbox']
            confidence = region.get('confidence', 0.0)
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
            axes[0].text(x, y-5, f'Logo: {confidence:.2f}', color='red', fontsize=8, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        axes[0].axis('off')
        
        # Results summary
        axes[1].text(0.1, 0.9, f"Counterfeit: {'YES' if result.is_counterfeit else 'NO'}", 
                    fontsize=16, fontweight='bold', 
                    color='red' if result.is_counterfeit else 'green')
        axes[1].text(0.1, 0.8, f"Confidence: {result.confidence:.3f}", fontsize=12)
        axes[1].text(0.1, 0.7, f"Logo Anomaly: {result.logo_anomaly_score:.3f}", fontsize=12)
        axes[1].text(0.1, 0.6, f"Image Similarity: {result.image_similarity_score:.3f}", fontsize=12)
        axes[1].text(0.1, 0.5, f"Logos Detected: {len(result.logo_regions)}", fontsize=12)
        
        if result.detected_issues:
            axes[1].text(0.1, 0.4, "Issues Detected:", fontsize=12, fontweight='bold')
            for i, issue in enumerate(result.detected_issues):
                axes[1].text(0.1, 0.35 - i*0.05, f"• {issue}", fontsize=10, color='red')
        
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def batch_detect(self, image_paths: List[str], reference_urls: Optional[List[str]] = None) -> List[DetectionResult]:
        """Perform batch detection on multiple images"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                continue
            
            # Get reference URL if provided
            reference_url = reference_urls[i] if reference_urls and i < len(reference_urls) else None
            
            # Detect counterfeit
            result = self.detect_counterfeit(image, reference_url)
            results.append(result)
        
        return results

# Utility functions for easy usage
def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load image from file path"""
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return None
    return image

def save_result_summary(results: List[DetectionResult], output_path: str):
    """Save detection results summary to file"""
    with open(output_path, 'w') as f:
        f.write("Product Counterfeit Detection Results (YOLO-based)\n")
        f.write("=" * 50 + "\n\n")
        
        for i, result in enumerate(results):
            f.write(f"Image {i+1}:\n")
            f.write(f"  Counterfeit: {'YES' if result.is_counterfeit else 'NO'}\n")
            f.write(f"  Confidence: {result.confidence:.3f}\n")
            f.write(f"  Logo Anomaly Score: {result.logo_anomaly_score:.3f}\n")
            f.write(f"  Image Similarity Score: {result.image_similarity_score:.3f}\n")
            f.write(f"  Logos Detected: {len(result.logo_regions)}\n")
            
            if result.logo_regions:
                f.write("  Logo Details:\n")
                for j, logo in enumerate(result.logo_regions):
                    f.write(f"    Logo {j+1}: bbox={logo['bbox']}, confidence={logo['confidence']:.3f}\n")
            
            if result.detected_issues:
                f.write("  Issues:\n")
                for issue in result.detected_issues:
                    f.write(f"    - {issue}\n")
            
            f.write("\n")

# Example usage
if __name__ == "__main__":
    # Initialize detector with YOLO model
    detector = ProductCounterfeitDetector("m1.pt")
    
    # Example 1: Single image detection
    image_path = "./archive/selected_for_segmentation/2607.jpg"
    reference_url = "https://th.bing.com/th/id/OIP.hlS2sncgJNmpza2QcRwJ8QHaLH?w=204&h=306&c=7&r=0&o=7&dpr=1.1&pid=1.7&rm=3"
    
    if os.path.exists(image_path):
        image = load_image(image_path)
        if image is not None:
            result = detector.detect_counterfeit(image, reference_url)
            
            print(f"Counterfeit: {'YES' if result.is_counterfeit else 'NO'}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Logo Anomaly Score: {result.logo_anomaly_score:.3f}")
            print(f"Image Similarity Score: {result.image_similarity_score:.3f}")
            print(f"Logos Detected: {len(result.logo_regions)}")
            
            if result.logo_regions:
                print("Logo Details:")
                for i, logo in enumerate(result.logo_regions):
                    print(f"  Logo {i+1}: bbox={logo['bbox']}, confidence={logo['confidence']:.3f}")
            
            if result.detected_issues:
                print("Issues Detected:")
                for issue in result.detected_issues:
                    print(f"  - {issue}")
            
            # Visualize results
            detector.visualize_results(image, result, "detection_result.png")
    
    # Example 2: Batch detection
    image_paths = ["./archive/selected_for_segmentation/2607.jpg", "./archive/selected_for_segmentation/2605.jpg"]
    reference_urls = ["https://th.bing.com/th/id/OIP.hlS2sncgJNmpza2QcRwJ8QHaLH?w=204&h=306&c=7&r=0&o=7&dpr=1.1&pid=1.7&rm=3"]
    
    batch_results = detector.batch_detect(image_paths, reference_urls)
    save_result_summary(batch_results, "batch_detection_results.txt")
