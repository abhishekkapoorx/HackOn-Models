#!/usr/bin/env python3
"""
Enhanced Counterfeit Detection System with Web Search and Logo Comparison
Fixed version for FastAPI deployment
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import hashlib
import json
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from transformers import CLIPProcessor, CLIPModel, AutoFeatureExtractor, AutoModelForImageClassification, BlipProcessor, BlipForConditionalGeneration
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dotenv import load_dotenv
import logging
from ultralytics import YOLO
import time
from urllib.parse import quote_plus
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LogoInfo:
    """Information about detected logo"""
    bbox: Tuple[int, int, int, int]
    confidence: float
    area: int
    features: Optional[np.ndarray] = None
    distortion_score: float = 0.0
    quality_score: float = 0.0

@dataclass
class BrandImageInfo:
    """Information about brand reference image"""
    url: str
    image: Optional[np.ndarray] = None
    logos: Optional[List[LogoInfo]] = None
    similarity_score: float = 0.0
    download_success: bool = False

@dataclass
class CounterfeitAnalysisResult:
    """Comprehensive counterfeit analysis result"""
    is_counterfeit: bool
    confidence: float
    brand_detected: str
    input_logos: List[LogoInfo]
    reference_images: List[BrandImageInfo]
    logo_similarities: List[float]
    distortion_scores: List[float]
    detected_issues: List[str]
    analysis_details: Dict[str, Any]

class BrandDetector:
    """Detects brand from product image using image captioning"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = self.model.to(self.device)  # Fixed: proper assignment
            logger.info("Brand detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load brand detection model: {e}")
            self.processor = None
            self.model = None
    
    def detect_brand(self, image: np.ndarray) -> str:
        """Detect brand from image using image captioning"""
        if self.model is None:
            return "unknown"
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Generate caption
            inputs = self.processor(pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Fixed: proper device transfer
            
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=50)
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Extract brand names from caption
            brand = self._extract_brand_from_caption(caption)
            logger.info(f"Detected brand: {brand} from caption: {caption}")
            
            return brand
            
        except Exception as e:
            logger.error(f"Error in brand detection: {e}")
            return "unknown"
    
    def _extract_brand_from_caption(self, caption: str) -> str:
        """Extract brand name from caption"""
        # Common brand names to look for
        brands = [
            'nike', 'adidas', 'puma', 'reebok', 'converse', 'vans',
            'gucci', 'louis vuitton', 'chanel', 'prada', 'versace',
            'rolex', 'omega', 'cartier', 'patek philippe',
            'apple', 'samsung', 'sony', 'dell', 'hp',
            'coca cola', 'pepsi', 'starbucks', 'mcdonalds',
            'mercedes', 'bmw', 'audi', 'toyota', 'honda'
        ]
        
        caption_lower = caption.lower()
        
        for brand in brands:
            if brand in caption_lower:
                return brand
        
        # Try to extract any capitalized words that might be brands
        words = caption.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                return word.lower()
        
        return "unknown"

class WebImageSearcher:
    """Searches web for brand reference images using Google Custom Search"""
    
    def __init__(self):
        self.max_images = 5
        self.timeout = 10
        self.api_key = os.getenv('GAPIS_API_KEY')
        self.search_engine_id = os.getenv('GCSE_ID')
        
        if not self.api_key or not self.search_engine_id:
            logger.warning("Google Custom Search API key or Search Engine ID not found in environment variables")
    
    async def search_brand_images(self, brand: str, product_type: str = "product", description: str = "") -> List[str]:
        """Search for brand images on the web"""
        if brand == "unknown":
            return []
        
        if not self.api_key or not self.search_engine_id:
            logger.error("Google Custom Search API credentials not configured")
            return []
        
        # Extract keywords from description for better search
        desc_keywords = self._extract_keywords_from_description(description)
        
        # Build search queries incorporating description
        search_queries = []
        
        if desc_keywords:
            # Primary queries with description keywords
            search_queries.extend([
                f"{brand} {desc_keywords} original authentic",
                f"{brand} {desc_keywords} official",
                f"authentic {brand} {desc_keywords}",
                f"original {brand} {desc_keywords} {product_type}"
            ])
        
        # Fallback queries without description
        search_queries.extend([
            f"{brand} {product_type} original authentic",
            f"{brand} logo official",
            f"{brand} authentic {product_type}",
            f"original {brand} {product_type}"
        ])
        
        all_urls = []
        
        for query in search_queries:
            urls = await self._search_google_images(query)
            all_urls.extend(urls)
            if len(all_urls) >= self.max_images:
                break
        
        # Remove duplicates and limit results
        unique_urls = list(dict.fromkeys(all_urls))[:self.max_images]
        
        logger.info(f"Found {len(unique_urls)} reference images for brand: {brand}")
        return unique_urls
    
    async def _search_google_images(self, query: str) -> List[str]:
        """Search Google Custom Search for images"""
        try:
            search_url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'searchType': 'image',
                'num': 7,
                'imgType': 'photo',
                'safe': 'active'
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        urls = self._extract_image_urls_from_response(data)
                        return urls
                    else:
                        logger.error(f"Google Custom Search API returned status {response.status}")
                        
        except Exception as e:
            logger.error(f"Error searching Google images: {e}")
        
        return []
    
    def _extract_image_urls_from_response(self, response_data: dict) -> List[str]:
        """Extract image URLs from Google Custom Search API response"""
        urls = []
        
        if 'items' in response_data:
            for item in response_data['items']:
                if 'link' in item:
                    image_url = item['link']
                    
                    # Basic validation
                    if self._is_valid_image_url(image_url):
                        urls.append(image_url)
        
        return urls
    
    def _is_valid_image_url(self, url: str) -> bool:
        """Validate if URL is likely a valid image"""
        # Check if URL ends with image extension
        if not re.search(r'\.(jpg|jpeg|png|gif|webp)(\?.*)?$', url, re.IGNORECASE):
            return False
        
        # Skip very long URLs
        if len(url) > 500:
            return False
        
        # Skip certain domains that might not have useful images
        blocked_domains = ['googleapis.com', 'google.com', 'gstatic.com']
        if any(domain in url for domain in blocked_domains):
            return False
        
        return True
    
    def _extract_keywords_from_description(self, description: str) -> str:
        """Extract relevant keywords from product description"""
        if not description:
            return ""
        
        # Common product-related keywords to keep
        relevant_keywords = []
        
        # Color keywords
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'gray', 'grey', 'brown', 
                 'pink', 'purple', 'orange', 'navy', 'beige', 'tan', 'maroon', 'gold', 'silver']
        
        # Product type keywords
        product_types = ['shoe', 'shoes', 'sneaker', 'sneakers', 'boot', 'boots', 'sandal', 'sandals',
                        'shirt', 't-shirt', 'tshirt', 'polo', 'hoodie', 'jacket', 'coat', 'pants',
                        'jeans', 'shorts', 'dress', 'skirt', 'bag', 'purse', 'backpack', 'wallet',
                        'watch', 'sunglasses', 'hat', 'cap', 'belt', 'scarf', 'gloves']
        
        # Material keywords
        materials = ['leather', 'canvas', 'denim', 'cotton', 'wool', 'silk', 'polyester', 'nylon']
        
        # Style keywords
        styles = ['casual', 'formal', 'sport', 'athletic', 'running', 'basketball', 'tennis',
                 'vintage', 'classic', 'modern', 'retro', 'slim', 'regular', 'loose']
        
        all_keywords = colors + product_types + materials + styles
        
        # Extract keywords from description
        description_lower = description.lower()
        words = re.findall(r'\b\w+\b', description_lower)
        
        for word in words:
            if word in all_keywords and word not in relevant_keywords:
                relevant_keywords.append(word)
        
        # Limit to most relevant keywords (max 3)
        return ' '.join(relevant_keywords[:3])
    
    def download_images(self, urls: List[str]) -> List[BrandImageInfo]:
        """Download reference images"""
        brand_images = []
        
        for url in urls:
            brand_info = BrandImageInfo(url=url)
            
            try:
                response = requests.get(url, timeout=self.timeout, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                response.raise_for_status()
                
                image = Image.open(BytesIO(response.content))
                image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                brand_info.image = image_array
                brand_info.download_success = True
                
                logger.info(f"Successfully downloaded image from: {url}")
                
            except Exception as e:
                logger.error(f"Failed to download image from {url}: {e}")
                brand_info.download_success = False
            
            brand_images.append(brand_info)
        
        return brand_images

class EnhancedLogoDetector:
    """Enhanced logo detector with feature extraction"""
    
    def __init__(self, model_path: str = "./models/my_model.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.confidence_threshold = 0.50
        
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            logger.info(f"YOLO model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
        
        # Load CLIP for feature extraction
        try:
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = self.clip_model.to(self.device)  # Fixed: proper assignment
            logger.info("CLIP model loaded for logo feature extraction")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self.clip_model = None
    
    def detect_logos(self, image: np.ndarray) -> List[LogoInfo]:
        """Detect logos and extract features"""
        if self.model is None:
            logger.warning("YOLO model not loaded")
            return []
        
        # Convert BGR to RGB for YOLO
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run YOLO inference
        results = self.model(rgb_image, verbose=False)
        
        logos = []
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                if confidence < self.confidence_threshold:
                    continue
                
                # Convert to integer coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                
                # Extract logo region
                logo_roi = image[y1:y2, x1:x2]
                
                # Create LogoInfo
                logo_info = LogoInfo(
                    bbox=(x1, y1, w, h),
                    confidence=float(confidence),
                    area=w * h
                )
                
                # Extract features
                if self.clip_model is not None and logo_roi.size > 0:
                    logo_info.features = self._extract_logo_features(logo_roi)
                
                # Calculate quality scores
                logo_info.distortion_score = self._calculate_distortion_score(logo_roi)
                logo_info.quality_score = self._calculate_quality_score(logo_roi)
                
                logos.append(logo_info)
        
        logger.info(f"Detected {len(logos)} logos with confidence > {self.confidence_threshold}")
        return logos
    
    def _extract_logo_features(self, logo_image: np.ndarray) -> np.ndarray:
        """Extract features from logo using CLIP"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(logo_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Process with CLIP
            inputs = self.clip_processor(images=pil_image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)  # Fixed: proper device transfer
            
            with torch.no_grad():
                features = self.clip_model.get_image_features(pixel_values=pixel_values)
            
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error extracting logo features: {e}")
            return np.array([])
    
    def _calculate_distortion_score(self, logo_image: np.ndarray) -> float:
        """Calculate logo distortion score"""
        if logo_image.size == 0:
            return 1.0
        
        try:
            gray = cv2.cvtColor(logo_image, cv2.COLOR_BGR2GRAY)
            
            # Calculate symmetry
            height, width = gray.shape
            left_half = gray[:, :width//2]
            right_half = cv2.flip(gray[:, width//2:], 1)
            
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            symmetry_score = 1 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255
            
            # Calculate edge consistency
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Combined distortion score (lower is better)
            distortion = 1 - (symmetry_score * 0.7 + edge_density * 0.3)
            
            return float(distortion)
            
        except Exception as e:
            logger.error(f"Error calculating distortion score: {e}")
            return 1.0
    
    def _calculate_quality_score(self, logo_image: np.ndarray) -> float:
        """Calculate logo quality score"""
        if logo_image.size == 0:
            return 0.0
        
        try:
            gray = cv2.cvtColor(logo_image, cv2.COLOR_BGR2GRAY)
            
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = min(laplacian_var / 1000, 1.0)  # Normalize
            
            # Calculate contrast
            contrast = gray.std() / 255
            
            # Calculate brightness consistency
            brightness_std = np.std(gray) / 255
            brightness_consistency = 1 - brightness_std
            
            # Combined quality score
            quality = (sharpness * 0.4 + contrast * 0.3 + brightness_consistency * 0.3)
            
            return float(quality)
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0

class LogoComparator:
    """Compares logos between input and reference images"""
    
    def __init__(self):
        self.similarity_threshold = 0.5
        self.distortion_threshold = 0.5
    
    def compare_logos(self, input_logos: List[LogoInfo], reference_logos: List[LogoInfo]) -> Dict[str, Any]:
        """Compare logos between input and reference images"""
        if not input_logos or not reference_logos:
            return {
                'max_similarity': 0.0,
                'avg_similarity': 0.0,
                'similarity_scores': [],
                'best_matches': [],
                'distortion_analysis': {
                    'distortion_issues': [],
                    'avg_input_distortion': 0.0,
                    'avg_reference_distortion': 0.0,
                    'distortion_difference': 0.0
                }
            }
        
        similarity_matrix = []
        best_matches = []
        
        for i, input_logo in enumerate(input_logos):
            logo_similarities = []
            
            for j, ref_logo in enumerate(reference_logos):
                similarity = self._calculate_logo_similarity(input_logo, ref_logo)
                logo_similarities.append(similarity)
            
            similarity_matrix.append(logo_similarities)
            
            # Find best match for this input logo
            best_ref_idx = np.argmax(logo_similarities)
            best_similarity = logo_similarities[best_ref_idx]
            
            best_matches.append({
                'input_logo_idx': i,
                'reference_logo_idx': best_ref_idx,
                'similarity': best_similarity,
                'input_distortion': input_logo.distortion_score,
                'reference_distortion': reference_logos[best_ref_idx].distortion_score
            })
        
        # Calculate overall metrics
        all_similarities = [match['similarity'] for match in best_matches]
        max_similarity = max(all_similarities) if all_similarities else 0.0
        avg_similarity = np.mean(all_similarities) if all_similarities else 0.0
        
        # Distortion analysis
        distortion_analysis = self._analyze_distortions(input_logos, reference_logos, best_matches)
        
        return {
            'max_similarity': float(max_similarity),
            'avg_similarity': float(avg_similarity),
            'similarity_scores': all_similarities,
            'best_matches': best_matches,
            'distortion_analysis': distortion_analysis,
            'similarity_matrix': similarity_matrix
        }
    
    def _calculate_logo_similarity(self, logo1: LogoInfo, logo2: LogoInfo) -> float:
        """Calculate similarity between two logos"""
        if logo1.features is None or logo2.features is None or len(logo1.features) == 0 or len(logo2.features) == 0:
            # Fallback to geometric similarity
            return self._geometric_similarity(logo1, logo2)
        
        try:
            # Feature-based similarity using cosine similarity
            # Fixed: ensure features are 2D arrays
            feat1 = logo1.features.reshape(1, -1)
            feat2 = logo2.features.reshape(1, -1)
            similarity = cosine_similarity(feat1, feat2)[0][0]
            
            # Adjust based on size similarity
            size_similarity = self._size_similarity(logo1, logo2)
            
            # Combined similarity
            combined_similarity = similarity * 0.8 + size_similarity * 0.2
            
            return float(combined_similarity)
            
        except Exception as e:
            logger.error(f"Error calculating logo similarity: {e}")
            return self._geometric_similarity(logo1, logo2)
    
    def _geometric_similarity(self, logo1: LogoInfo, logo2: LogoInfo) -> float:
        """Calculate geometric similarity between logos"""
        # Size similarity
        size_sim = self._size_similarity(logo1, logo2)
        
        # Aspect ratio similarity
        aspect1 = logo1.bbox[2] / logo1.bbox[3] if logo1.bbox[3] > 0 else 1.0
        aspect2 = logo2.bbox[2] / logo2.bbox[3] if logo2.bbox[3] > 0 else 1.0
        
        aspect_sim = 1 - abs(aspect1 - aspect2) / max(aspect1, aspect2)
        
        return (size_sim + aspect_sim) / 2
    
    def _size_similarity(self, logo1: LogoInfo, logo2: LogoInfo) -> float:
        """Calculate size similarity between logos"""
        area1 = logo1.area
        area2 = logo2.area
        
        if area1 == 0 or area2 == 0:
            return 0.0
        
        size_ratio = min(area1, area2) / max(area1, area2)
        return size_ratio
    
    def _analyze_distortions(self, input_logos: List[LogoInfo], reference_logos: List[LogoInfo], matches: List[Dict]) -> Dict[str, Any]:
        """Analyze distortions in matched logos"""
        distortion_issues = []
        
        for match in matches:
            input_distortion = match['input_distortion']
            ref_distortion = match['reference_distortion']
            
            distortion_diff = input_distortion - ref_distortion
            
            if input_distortion > self.distortion_threshold:
                distortion_issues.append(f"High distortion in input logo (score: {input_distortion:.3f})")
            
            if distortion_diff > 0.2:
                distortion_issues.append(f"Significantly higher distortion than reference (diff: {distortion_diff:.3f})")
        
        avg_input_distortion = np.mean([logo.distortion_score for logo in input_logos])
        avg_ref_distortion = np.mean([logo.distortion_score for logo in reference_logos])
        
        return {
            'distortion_issues': distortion_issues,
            'avg_input_distortion': float(avg_input_distortion),
            'avg_reference_distortion': float(avg_ref_distortion),
            'distortion_difference': float(avg_input_distortion - avg_ref_distortion)
        }

class EnhancedCounterfeitDetector:
    """Main enhanced counterfeit detection system"""
    
    def __init__(self, yolo_model_path: str = "./models/my_model.pt"):
        self.brand_detector = BrandDetector()
        self.web_searcher = WebImageSearcher()
        self.logo_detector = EnhancedLogoDetector(yolo_model_path)
        self.logo_comparator = LogoComparator()
        
        # Detection thresholds
        self.thresholds = {
            'logo_similarity': 0.5,
            'distortion_score': 0.5,
            'overall_confidence': 0.5
        }
    
    async def analyze_counterfeit(self, image: np.ndarray, product_description: str = "") -> CounterfeitAnalysisResult:
        """Comprehensive counterfeit analysis"""
        logger.info("Starting enhanced counterfeit analysis...")
        
        # Step 1: Detect brand
        brand = self.brand_detector.detect_brand(image)
        if product_description:
            # Try to extract brand from description as well
            desc_brand = self._extract_brand_from_description(product_description)
            if desc_brand != "unknown":
                brand = desc_brand
        
        logger.info(f"Detected brand: {brand}")
        
        # Step 2: Detect logos in input image
        input_logos = self.logo_detector.detect_logos(image)
        logger.info(f"Detected {len(input_logos)} logos in input image")
        
        # Step 3: Search for reference images
        reference_urls = await self.web_searcher.search_brand_images(brand, "product", product_description)
        reference_images = self.web_searcher.download_images(reference_urls)
        
        # Step 4: Detect logos in reference images
        for ref_image_info in reference_images:
            if ref_image_info.download_success and ref_image_info.image is not None:
                ref_image_info.logos = self.logo_detector.detect_logos(ref_image_info.image)
        
        # Step 5: Compare logos
        logo_similarities = []
        distortion_scores = []
        detected_issues = []
        
        comparison_results = []
        
        for ref_image_info in reference_images:
            if ref_image_info.logos:
                comparison = self.logo_comparator.compare_logos(input_logos, ref_image_info.logos)
                comparison_results.append(comparison)
                
                logo_similarities.append(comparison['max_similarity'])
                
                # Check for issues
                if comparison['max_similarity'] < self.thresholds['logo_similarity']:
                    detected_issues.append(f"Low logo similarity with reference (score: {comparison['max_similarity']:.3f})")
                
                # Distortion analysis
                distortion_analysis = comparison.get('distortion_analysis', {})
                avg_input_distortion = distortion_analysis.get('avg_input_distortion', 0.0)
                distortion_scores.append(avg_input_distortion)
                
                distortion_issues = distortion_analysis.get('distortion_issues', [])
                if distortion_issues:
                    detected_issues.extend(distortion_issues)
        
        # Step 6: Overall analysis
        is_counterfeit, confidence = self._determine_counterfeit_status(
            input_logos, logo_similarities, distortion_scores, detected_issues
        )
        
        # Create comprehensive result
        result = CounterfeitAnalysisResult(
            is_counterfeit=is_counterfeit,
            confidence=confidence,
            brand_detected=brand,
            input_logos=input_logos,
            reference_images=reference_images,
            logo_similarities=logo_similarities,
            distortion_scores=distortion_scores,
            detected_issues=detected_issues,
            analysis_details={
                'comparison_results': comparison_results,
                'num_reference_images': len([img for img in reference_images if img.download_success]),
                'avg_logo_similarity': np.mean(logo_similarities) if logo_similarities else 0.0,
                'avg_distortion_score': np.mean(distortion_scores) if distortion_scores else 0.0
            }
        )
        
        logger.info(f"Analysis complete. Counterfeit: {is_counterfeit}, Confidence: {confidence:.3f}")
        return result
    
    def _extract_brand_from_description(self, description: str) -> str:
        """Extract brand from product description"""
        brands = [
            'nike', 'adidas', 'puma', 'reebok', 'converse', 'vans',
            'gucci', 'louis vuitton', 'chanel', 'prada', 'versace',
            'rolex', 'omega', 'cartier', 'patek philippe',
            'apple', 'samsung', 'sony', 'dell', 'hp'
        ]
        
        description_lower = description.lower()
        
        for brand in brands:
            if brand in description_lower:
                return brand
        
        return "unknown"
    
    def _determine_counterfeit_status(self, input_logos: List[LogoInfo], similarities: List[float], 
                                   distortions: List[float], issues: List[str]) -> Tuple[bool, float]:
        """Determine if product is counterfeit based on analysis"""
        
        # No logos detected
        if not input_logos:
            return True, 0.8  # Suspicious if no logos on branded product
        
        # No reference data
        if not similarities:
            return False, 0.3  # Can't determine without reference
        
        # Calculate scores
        max_similarity = max(similarities) if similarities else 0.0
        avg_similarity = np.mean(similarities) if similarities else 0.0
        avg_distortion = np.mean(distortions) if distortions else 0.0
        
        # Decision logic
        counterfeit_indicators = 0
        confidence_factors = []
        
        # Low similarity indicator
        if max_similarity < self.thresholds['logo_similarity']:
            counterfeit_indicators += 1
            confidence_factors.append(1 - max_similarity)
        
        # High distortion indicator
        if avg_distortion > self.thresholds['distortion_score']:
            counterfeit_indicators += 1
            confidence_factors.append(avg_distortion)
        
        # Multiple issues indicator
        if len(issues) > 2:
            counterfeit_indicators += 1
            confidence_factors.append(len(issues) / 10)
        
        # Determine result
        is_counterfeit = counterfeit_indicators >= 2
        
        # Calculate confidence
        if confidence_factors:
            base_confidence = np.mean(confidence_factors)
            confidence = min(0.95, max(0.1, base_confidence))
        else:
            confidence = 0.9 if not is_counterfeit else 0.5
        
        return is_counterfeit, float(confidence)
    
    def visualize_analysis(self, image: np.ndarray, result: CounterfeitAnalysisResult, save_path: str = ""):
        """Visualize comprehensive analysis results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image with logos
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(rgb_image)
        axes[0, 0].set_title(f'Input Image - Brand: {result.brand_detected}')
        
        for i, logo in enumerate(result.input_logos):
            x, y, w, h = logo.bbox
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
            axes[0, 0].add_patch(rect)
            axes[0, 0].text(x, y-5, f'Logo {i+1}: {logo.confidence:.2f}', 
                           color='red', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        axes[0, 0].axis('off')
        
        # Reference images
        ref_count = 0
        for ref_img in result.reference_images:
            if ref_img.download_success and ref_img.image is not None and ref_count < 2:
                row = 0 if ref_count == 0 else 1
                col = 1 if ref_count == 0 else 0
                
                ref_rgb = cv2.cvtColor(ref_img.image, cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(ref_rgb)
                axes[row, col].set_title(f'Reference {ref_count + 1}')
                
                if ref_img.logos:
                    for logo in ref_img.logos:
                        x, y, w, h = logo.bbox
                        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none')
                        axes[row, col].add_patch(rect)
                
                axes[row, col].axis('off')
                ref_count += 1
        
        # Analysis results
        axes[0, 2].text(0.1, 0.9, f"Counterfeit: {'YES' if result.is_counterfeit else 'NO'}", 
                       fontsize=16, fontweight='bold', 
                       color='red' if result.is_counterfeit else 'green')
        axes[0, 2].text(0.1, 0.8, f"Confidence: {result.confidence:.3f}", fontsize=12)
        axes[0, 2].text(0.1, 0.7, f"Brand: {result.brand_detected}", fontsize=12)
        axes[0, 2].text(0.1, 0.6, f"Input Logos: {len(result.input_logos)}", fontsize=12)
        axes[0, 2].text(0.1, 0.5, f"Reference Images: {result.analysis_details['num_reference_images']}", fontsize=12)
        
        if result.logo_similarities:
            axes[0, 2].text(0.1, 0.4, f"Max Similarity: {max(result.logo_similarities):.3f}", fontsize=12)
            axes[0, 2].text(0.1, 0.3, f"Avg Similarity: {result.analysis_details['avg_logo_similarity']:.3f}", fontsize=12)
        
        if result.distortion_scores:
            axes[0, 2].text(0.1, 0.2, f"Avg Distortion: {result.analysis_details['avg_distortion_score']:.3f}", fontsize=12)
        
        axes[0, 2].axis('off')
        
        # Issues
        axes[1, 2].text(0.1, 0.9, "Detected Issues:", fontsize=12, fontweight='bold')
        if result.detected_issues:
            for i, issue in enumerate(result.detected_issues[:8]):  # Show max 8 issues
                axes[1, 2].text(0.1, 0.8 - i*0.08, f"• {issue}", fontsize=9, color='red')
        else:
            axes[1, 2].text(0.1, 0.8, "No issues detected", fontsize=10, color='green')
        
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_analysis_report(self, result: CounterfeitAnalysisResult, output_path: str):
        """Save detailed analysis report"""
        report = {
            'analysis_summary': {
                'is_counterfeit': result.is_counterfeit,
                'confidence': result.confidence,
                'brand_detected': result.brand_detected,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'logo_analysis': {
                'input_logos_count': len(result.input_logos),
                'input_logos': [asdict(logo) for logo in result.input_logos],
                'logo_similarities': result.logo_similarities,
                'distortion_scores': result.distortion_scores
            },
            'reference_analysis': {
                'reference_images_found': len(result.reference_images),
                'successful_downloads': len([img for img in result.reference_images if img.download_success]),
                'reference_urls': [img.url for img in result.reference_images]
            },
            'detected_issues': result.detected_issues,
            'analysis_details': result.analysis_details
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Analysis report saved to: {output_path}")

# Utility functions
def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load image from file path"""
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return None
    return image

async def analyze_product_image(image_path: str, description: str = "", output_dir: str = "output"):
    """Analyze a product image for counterfeiting"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = load_image(image_path)
    if image is None:
        return None
    
    # Initialize detector
    detector = EnhancedCounterfeitDetector("./models/my_model.pt")
    
    # Perform analysis
    result = await detector.analyze_counterfeit(image, description)
    
    # Generate outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save visualization
    viz_path = os.path.join(output_dir, f"{base_name}_analysis.png")
    detector.visualize_analysis(image, result, viz_path)
    
    # Save report
    report_path = os.path.join(output_dir, f"{base_name}_report.json")
    detector.save_analysis_report(result, report_path)
    
    return result 