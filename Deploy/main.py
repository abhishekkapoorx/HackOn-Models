#!/usr/bin/env python3
"""
FastAPI Backend for Product Authentication and Review Analysis
Integrates counterfeit detection, review processing, and fake review detection
"""

# Load environment variables from .env file first
from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import logging
from typing import List, Optional, Dict, Any
from io import BytesIO
import base64
import json
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, SecretStr
import uvicorn
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyabsa import AspectTermExtraction as ATEPC
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import aiofiles

def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# Import our custom modules
import sys
sys.path.append('..')
from enhanced_counterfeit_detector import EnhancedCounterfeitDetector, CounterfeitAnalysisResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Product Authentication & Review Analysis API",
    description="AI-powered product authenticity verification and review analysis service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class CounterfeitDetectionRequest(BaseModel):
    image_base64: Optional[str] = None
    product_description: str = Field(default="", description="Product description for better analysis")

class CounterfeitDetectionResponse(BaseModel):
    is_counterfeit: bool
    confidence: float
    brand_detected: str
    detected_issues: List[str]
    logo_count: int
    analysis_summary: Dict[str, Any]

class ReviewAnalysisRequest(BaseModel):
    reviews: List[str] = Field(..., description="List of review texts to analyze")
    product_id: str = Field(..., description="Product ID for the reviews")

class ReviewAnalysisResponse(BaseModel):
    product_id: str
    total_reviews: int
    positive_aspects: List[str]
    negative_aspects: List[str]
    suggestions: Dict[str, List[str]]
    sentiment_distribution: Dict[str, int]

class FakeReviewDetectionRequest(BaseModel):
    reviews: List[str] = Field(..., description="List of review texts to check for authenticity")

class FakeReviewDetectionResponse(BaseModel):
    total_reviews: int
    fake_reviews_detected: int
    fake_review_percentage: float
    review_authenticity: List[Dict[str, Any]]

class BatchAnalysisRequest(BaseModel):
    image_base64: Optional[str] = None
    product_description: str = ""
    reviews: List[str] = []
    product_id: str

class BatchAnalysisResponse(BaseModel):
    counterfeit_analysis: CounterfeitDetectionResponse
    review_analysis: ReviewAnalysisResponse
    fake_review_analysis: FakeReviewDetectionResponse

class SingleReviewAnalysisRequest(BaseModel):
    review: str = Field(..., description="Single review text to analyze")
    product_id: str = Field(..., description="Product ID for the review")

class SingleReviewAnalysisResponse(BaseModel):
    product_id: str
    review_text: str
    is_fake: bool
    fake_probability: float
    sentiment: str
    sentiment_confidence: float

class SingleReturnAnalysisRequest(BaseModel):
    return_reason: str = Field(..., description="Single return reason text to analyze")
    product_id: str = Field(..., description="Product ID for the return")
    return_id: Optional[str] = Field(None, description="Return ID for tracking")

class SingleReturnAnalysisResponse(BaseModel):
    product_id: str
    return_id: Optional[str]
    return_text: str
    is_fake: bool
    fake_probability: float
    sentiment: str
    sentiment_confidence: float
    return_category: str  # e.g., "Quality", "Size", "Shipping", "Other"

class BatchReturnAnalysisRequest(BaseModel):
    returns: List[str] = Field(..., description="List of return reason texts to analyze")
    product_id: str = Field(..., description="Product ID for the returns")

class BatchReturnAnalysisResponse(BaseModel):
    product_id: str
    total_returns: int
    positive_aspects: List[str]
    negative_aspects: List[str]
    return_categories: Dict[str, int]  # Categories like Quality, Size, Shipping issues
    suggestions: Dict[str, List[str]]
    sentiment_distribution: Dict[str, int]
    fake_returns_detected: int
    fake_return_percentage: float

# Global variables for models
counterfeit_detector = None
fake_review_tokenizer = None
fake_review_model = None
aspect_extractor = None
llm = None
device = None

@app.on_event("startup")
async def startup_event():
    """Initialize ML models on startup"""
    global counterfeit_detector, fake_review_tokenizer, fake_review_model, aspect_extractor, llm, device
    
    logger.info("Initializing models...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Initialize counterfeit detector
        counterfeit_detector = EnhancedCounterfeitDetector("./models/my_model.pt")
        logger.info("Counterfeit detector initialized")
        
        # Initialize fake review detection model
        fake_review_tokenizer = AutoTokenizer.from_pretrained("SravaniNirati/bert_fake_review_detection")
        fake_review_model = AutoModelForSequenceClassification.from_pretrained(
            "SravaniNirati/bert_fake_review_detection"
        ).to(device)
        logger.info("Fake review detection model initialized")
        
        # Initialize aspect extractor
        aspect_extractor = ATEPC.AspectExtractor(
            checkpoint="english",
            auto_device=True
        )
        logger.info("Aspect extractor initialized")
        
        # Initialize LLM for suggestions
        groq_api_key = os.getenv('GROQ_API_KEY')
        if groq_api_key:
            llm = ChatGroq(model="llama3-70b-8192", api_key=SecretStr(groq_api_key))
            logger.info("LLM initialized")
        else:
            logger.warning("GROQ_API_KEY not found. Review suggestions will be limited.")
            
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 image to OpenCV format"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        pil_image = Image.open(BytesIO(image_data))
        
        # Convert to OpenCV format (BGR)
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return cv_image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def detect_fake_reviews(reviews: List[str]) -> List[Dict[str, Any]]:
    """Detect fake reviews in batch"""
    if not fake_review_model or not fake_review_tokenizer:
        raise HTTPException(status_code=500, detail="Fake review detection model not initialized")
    
    results = []
    batch_size = 32
    
    for i in range(0, len(reviews), batch_size):
        batch_reviews = reviews[i:i+batch_size]
        
        try:
            # Tokenize batch
            inputs = fake_review_tokenizer(
                batch_reviews, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = fake_review_model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
                
            for j, review in enumerate(batch_reviews):
                fake_prob = predictions[j][0].item()  # Probability of being fake
                is_fake = fake_prob > 0.5
                
                results.append({
                    "review": review[:100] + "..." if len(review) > 100 else review,
                    "is_fake": is_fake,
                    "fake_probability": float(fake_prob),  # Convert to Python float
                    "confidence": float(max(fake_prob, 1 - fake_prob))  # Convert to Python float
                })
                
        except Exception as e:
            logger.error(f"Error processing batch {i}: {e}")
            # Add error results for this batch
            for review in batch_reviews:
                results.append({
                    "review": review[:100] + "..." if len(review) > 100 else review,
                    "is_fake": False,
                    "fake_probability": 0.5,  # Already Python float
                    "confidence": 0.0,  # Already Python float
                    "error": str(e)
                })
    
    return results

def process_reviews_for_aspects(reviews: List[str], product_id: str) -> Dict[str, Any]:
    """Process reviews to extract aspects and generate suggestions"""
    if not aspect_extractor:
        raise HTTPException(status_code=500, detail="Aspect extractor not initialized")
    
    try:
        # Extract aspects and sentiment
        extracted = aspect_extractor.extract_aspect(
            inference_source=reviews,
            pred_sentiment=True,
            print_result=False
        )
        
        # Aggregate aspects by sentiment
        from collections import defaultdict, Counter
        aspect_summary = defaultdict(list)
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        
        # Add null check for extracted results
        if extracted:
            for entry in extracted:
                if isinstance(entry, dict) and 'aspect' in entry and 'sentiment' in entry:
                    for asp, sent in zip(entry['aspect'], entry['sentiment']):
                        aspect_summary[sent].append(asp)
                        if sent in sentiment_counts:
                            sentiment_counts[sent] += 1
        
        # Get top aspects
        def top_aspects(aspect_list, max_count=10):
            if not aspect_list:
                return []
            return [item for item, _ in Counter(aspect_list).most_common(max_count)]
        
        positive_aspects = top_aspects(aspect_summary.get("Positive", []))
        negative_aspects = top_aspects(aspect_summary.get("Negative", []))
        
        # Generate suggestions using LLM if available
        suggestions = {"improvements": [], "strengths": []}
        if llm and (positive_aspects or negative_aspects):
            try:
                prompt = PromptTemplate(
                    input_variables=["positives", "negatives"],
                    template="""Based on these product review aspects:

Positive aspects: {positives}
Negative aspects: {negatives}

Give sellers actionable advice in two categories:
1. Improvements they should make (based on negative aspects)
2. Strengths they should highlight (based on positive aspects)

Return only a JSON object with 'improvements' and 'strengths' keys, each containing an array of strings.
"""
                )
                
                chain = prompt | llm
                response = chain.invoke({
                    "positives": ", ".join(positive_aspects),
                    "negatives": ", ".join(negative_aspects)
                })
                
                # Parse LLM response with proper type checking
                try:
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    # Ensure response_text is a string
                    if isinstance(response_text, list):
                        response_text = " ".join(str(item) for item in response_text)
                    else:
                        response_text = str(response_text)
                    
                    # Extract JSON from response
                    import re
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        suggestions = json.loads(json_match.group())
                except:
                    logger.warning("Could not parse LLM response for suggestions")
                    
            except Exception as e:
                logger.error(f"Error generating suggestions: {e}")
        
        return {
            "positive_aspects": positive_aspects,
            "negative_aspects": negative_aspects,
            "suggestions": suggestions,
            "sentiment_distribution": {k: int(v) for k, v in sentiment_counts.items()}  # Convert to Python int
        }
        
    except Exception as e:
        logger.error(f"Error processing reviews: {e}")
        return {
            "positive_aspects": [],
            "negative_aspects": [],
            "suggestions": {"improvements": [], "strengths": []},
            "sentiment_distribution": {"Positive": 0, "Negative": 0, "Neutral": 0}  # Already Python ints
        }

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Product Authentication & Review Analysis API", "status": "running"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models": {
            "counterfeit_detector": counterfeit_detector is not None,
            "fake_review_model": fake_review_model is not None,
            "aspect_extractor": aspect_extractor is not None,
            "llm": llm is not None
        },
        "device": str(device) if device else "unknown"
    }

@app.post("/detect-counterfeit", response_model=CounterfeitDetectionResponse)
async def detect_counterfeit(request: CounterfeitDetectionRequest):
    """Detect if a product is counterfeit based on image analysis"""
    if not counterfeit_detector:
        raise HTTPException(status_code=500, detail="Counterfeit detector not initialized")
    
    if not request.image_base64:
        raise HTTPException(status_code=400, detail="Image data is required")
    
    try:
        # Decode image
        image = decode_base64_image(request.image_base64)
         
        # Perform analysis
        result = await counterfeit_detector.analyze_counterfeit(
            image, 
            request.product_description
        )
        
        return CounterfeitDetectionResponse(
            is_counterfeit=result.is_counterfeit,
            confidence=float(result.confidence),  # Convert to Python float
            brand_detected=result.brand_detected,
            detected_issues=result.detected_issues,
            logo_count=int(len(result.input_logos)),  # Convert to Python int
            analysis_summary=convert_numpy_types({
                "logo_similarities": result.logo_similarities,
                "distortion_scores": result.distortion_scores,
                "reference_images_found": len([img for img in result.reference_images if img.download_success]),
                "analysis_details": result.analysis_details
            })
        )
        
    except Exception as e:
        logger.error(f"Error in counterfeit detection: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/upload-image-counterfeit")
async def upload_image_counterfeit(
    file: UploadFile = File(...),
    product_description: str = Form("")
):
    """Upload image file for counterfeit detection"""
    if not counterfeit_detector:
        raise HTTPException(status_code=500, detail="Counterfeit detector not initialized")
    
    # Validate file type with null check
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and decode image
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Perform analysis
        result = await counterfeit_detector.analyze_counterfeit(image, product_description)
        
        return CounterfeitDetectionResponse(
            is_counterfeit=result.is_counterfeit,
            confidence=float(result.confidence),  # Convert to Python float
            brand_detected=result.brand_detected,
            detected_issues=result.detected_issues,
            logo_count=int(len(result.input_logos)),  # Convert to Python int
            analysis_summary=convert_numpy_types({
                "logo_similarities": result.logo_similarities,
                "distortion_scores": result.distortion_scores,
                "reference_images_found": len([img for img in result.reference_images if img.download_success]),
                "analysis_details": result.analysis_details
            })
        )
        
    except Exception as e:
        logger.error(f"Error processing uploaded image: {e}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.post("/analyze-reviews", response_model=ReviewAnalysisResponse)
async def analyze_reviews(request: ReviewAnalysisRequest):
    """Analyze product reviews to extract aspects and generate suggestions"""
    if not request.reviews:
        raise HTTPException(status_code=400, detail="Reviews list cannot be empty")
    
    try:
        analysis_result = process_reviews_for_aspects(request.reviews, request.product_id)
        
        return ReviewAnalysisResponse(
            product_id=request.product_id,
            total_reviews=int(len(request.reviews)),  # Convert to Python int
            positive_aspects=analysis_result["positive_aspects"],
            negative_aspects=analysis_result["negative_aspects"],
            suggestions=convert_numpy_types(analysis_result["suggestions"]),  # Convert numpy types
            sentiment_distribution=convert_numpy_types(analysis_result["sentiment_distribution"])  # Convert numpy types
        )
        
    except Exception as e:
        logger.error(f"Error analyzing reviews: {e}")
        raise HTTPException(status_code=500, detail=f"Review analysis failed: {str(e)}")

@app.post("/detect-fake-reviews", response_model=FakeReviewDetectionResponse)
async def detect_fake_reviews_endpoint(request: FakeReviewDetectionRequest):
    """Detect fake reviews in a batch of review texts"""
    if not request.reviews:
        raise HTTPException(status_code=400, detail="Reviews list cannot be empty")
    
    try:
        detection_results = detect_fake_reviews(request.reviews)
        
        fake_count = sum(1 for result in detection_results if result.get("is_fake", False))
        
        return FakeReviewDetectionResponse(
            total_reviews=int(len(request.reviews)),  # Convert to Python int
            fake_reviews_detected=int(fake_count),  # Convert to Python int
            fake_review_percentage=float((fake_count / len(request.reviews)) * 100),  # Convert to Python float
            review_authenticity=convert_numpy_types(detection_results)  # Convert numpy types
        )
        
    except Exception as e:
        logger.error(f"Error detecting fake reviews: {e}")
        raise HTTPException(status_code=500, detail=f"Fake review detection failed: {str(e)}")

@app.post("/batch-analysis", response_model=BatchAnalysisResponse)
async def batch_analysis(request: BatchAnalysisRequest):
    """Perform comprehensive analysis including counterfeit detection, review analysis, and fake review detection"""
    
    # Initialize response components
    counterfeit_response = None
    review_response = None
    fake_review_response = None
    
    try:
        # Counterfeit detection if image provided
        if request.image_base64 and counterfeit_detector:
            image = decode_base64_image(request.image_base64)
            counterfeit_result = await counterfeit_detector.analyze_counterfeit(
                image, 
                request.product_description
            )
            
            counterfeit_response = CounterfeitDetectionResponse(
                is_counterfeit=counterfeit_result.is_counterfeit,
                confidence=counterfeit_result.confidence,
                brand_detected=counterfeit_result.brand_detected,
                detected_issues=counterfeit_result.detected_issues,
                logo_count=len(counterfeit_result.input_logos),
                analysis_summary={
                    "logo_similarities": counterfeit_result.logo_similarities,
                    "distortion_scores": counterfeit_result.distortion_scores,
                    "reference_images_found": len([img for img in counterfeit_result.reference_images if img.download_success]),
                    "analysis_details": counterfeit_result.analysis_details
                }
            )
        else:
            # Default response if no image or detector not available
            counterfeit_response = CounterfeitDetectionResponse(
                is_counterfeit=False,
                confidence=0.0,
                brand_detected="unknown",
                detected_issues=[],
                logo_count=0,
                analysis_summary={}
            )
        
        # Review analysis if reviews provided
        if request.reviews:
            # Review analysis
            analysis_result = process_reviews_for_aspects(request.reviews, request.product_id)
            review_response = ReviewAnalysisResponse(
                product_id=request.product_id,
                total_reviews=len(request.reviews),
                positive_aspects=analysis_result["positive_aspects"],
                negative_aspects=analysis_result["negative_aspects"],
                suggestions=analysis_result["suggestions"],
                sentiment_distribution=analysis_result["sentiment_distribution"]
            )
            
            # Fake review detection
            detection_results = detect_fake_reviews(request.reviews)
            fake_count = sum(1 for result in detection_results if result.get("is_fake", False))
            fake_review_response = FakeReviewDetectionResponse(
                total_reviews=int(len(request.reviews)),  # Convert to Python int
                fake_reviews_detected=int(fake_count),  # Convert to Python int
                fake_review_percentage=float((fake_count / len(request.reviews)) * 100),  # Convert to Python float
                review_authenticity=convert_numpy_types(detection_results)  # Convert numpy types
            )
        else:
            # Default responses if no reviews
            review_response = ReviewAnalysisResponse(
                product_id=request.product_id,
                total_reviews=0,
                positive_aspects=[],
                negative_aspects=[],
                suggestions={"improvements": [], "strengths": []},
                sentiment_distribution={"Positive": 0, "Negative": 0, "Neutral": 0}
            )
            
            fake_review_response = FakeReviewDetectionResponse(
                total_reviews=0,
                fake_reviews_detected=0,
                fake_review_percentage=0.0,
                review_authenticity=[]
            )
        
        return BatchAnalysisResponse(
            counterfeit_analysis=counterfeit_response,
            review_analysis=review_response,
            fake_review_analysis=fake_review_response
        )
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@app.post("/detect-counterfeit-unified")
async def detect_counterfeit_unified(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    product_description: str = Form("")
):
    """
    Unified endpoint for counterfeit detection supporting both file upload and base64 image data.
    
    Usage:
    1. File upload: Send image as multipart/form-data with 'file' field
    2. Base64 image: Send image as base64 string in 'image_base64' field
    3. Both methods support optional 'product_description' field
    """
    if not counterfeit_detector:
        raise HTTPException(status_code=500, detail="Counterfeit detector not initialized")
    
    image = None
    
    try:
        # Handle file upload
        if file is not None:
            # Validate file type
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="Uploaded file must be an image")
            
            # Read and decode uploaded image
            contents = await file.read()
            image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid uploaded image file")
        
        # Handle base64 image (if no file was uploaded)
        elif image_base64:
            image = decode_base64_image(image_base64)
        
        else:
            raise HTTPException(
                status_code=400, 
                detail="Either 'file' (uploaded image) or 'image_base64' (base64 string) must be provided"
            )
        
        # Perform counterfeit analysis
        result = await counterfeit_detector.analyze_counterfeit(image, product_description)
        
        return CounterfeitDetectionResponse(
            is_counterfeit=result.is_counterfeit,
            confidence=result.confidence,
            brand_detected=result.brand_detected,
            detected_issues=result.detected_issues,
            logo_count=len(result.input_logos),
            analysis_summary={
                "logo_similarities": result.logo_similarities,
                "distortion_scores": result.distortion_scores,
                "reference_images_found": len([img for img in result.reference_images if img.download_success]),
                "analysis_details": result.analysis_details
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in unified counterfeit detection: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Enhanced batch analysis endpoint that supports file uploads
@app.post("/batch-analysis-enhanced")
async def batch_analysis_enhanced(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    product_description: str = Form(""),
    product_id: str = Form(...),
    reviews: Optional[str] = Form(None)  # JSON string of reviews array
):
    """
    Enhanced batch analysis supporting both file uploads and base64 images.
    
    Parameters:
    - file: Optional uploaded image file
    - image_base64: Optional base64 encoded image string
    - product_description: Product description for better analysis
    - product_id: Product ID for the analysis
    - reviews: JSON string containing array of review texts (optional)
    """
    
    # Parse reviews from JSON string
    parsed_reviews = []
    if reviews:
        try:
            parsed_reviews = json.loads(reviews)
            if not isinstance(parsed_reviews, list):
                raise ValueError("Reviews must be an array")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid reviews JSON: {str(e)}")
    
    # Create batch request object
    batch_request = BatchAnalysisRequest(
        image_base64=None,  # Will be set below if needed
        product_description=product_description,
        reviews=parsed_reviews,
        product_id=product_id
    )
    
    # Handle image input
    if file is not None:
        # Validate and process uploaded file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        
        try:
            # Read and convert to base64 for batch processing
            contents = await file.read()
            image_b64 = base64.b64encode(contents).decode('utf-8')
            batch_request.image_base64 = image_b64
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing uploaded image: {str(e)}")
    
    elif image_base64:
        batch_request.image_base64 = image_base64
    
    # Process using existing batch analysis logic
    return await batch_analysis(batch_request)

def analyze_single_review_sentiment(review: str) -> Dict[str, Any]:
    """Analyze sentiment of a single review without aspect extraction"""
    if not aspect_extractor:
        raise HTTPException(status_code=500, detail="Aspect extractor not initialized")
    
    try:
        # Use aspect extractor just for sentiment analysis on single review
        result = aspect_extractor.extract_aspect(
            inference_source=[review],
            pred_sentiment=True,
            print_result=False
        )
        
        # Extract sentiment from result
        sentiment = "Neutral"
        confidence = 0.5
        
        if result and len(result) > 0:
            entry = result[0]
            if isinstance(entry, dict) and 'sentiment' in entry:
                sentiments = entry['sentiment']
                if sentiments and len(sentiments) > 0:
                    # Get the most common sentiment
                    from collections import Counter
                    sentiment_counts = Counter(sentiments)
                    most_common = sentiment_counts.most_common(1)
                    if most_common:
                        sentiment = most_common[0][0]
                        confidence = most_common[0][1] / len(sentiments)
        
        return {
            "sentiment": sentiment,
            "confidence": float(confidence)  # Convert to Python float
        }
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        return {
            "sentiment": "Neutral",
            "confidence": 0.5  # Already Python float
        }

@app.post("/analyze-single-review", response_model=SingleReviewAnalysisResponse)
async def analyze_single_review(request: SingleReviewAnalysisRequest):
    """
    Analyze a single review for fake detection and sentiment analysis only.
    This is faster than batch analysis as it doesn't extract aspects.
    """
    if not request.review.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty")
    
    try:
        # 1. Fake review detection
        fake_results = detect_fake_reviews([request.review])
        fake_result = fake_results[0] if fake_results else {
            "is_fake": False,
            "fake_probability": 0.5,
            "confidence": 0.0
        }
        
        # 2. Sentiment analysis (without aspect extraction)
        sentiment_result = analyze_single_review_sentiment(request.review)
        
        return SingleReviewAnalysisResponse(
            product_id=request.product_id,
            review_text=request.review[:200] + "..." if len(request.review) > 200 else request.review,
            is_fake=fake_result.get("is_fake", False),
            fake_probability=float(fake_result.get("fake_probability", 0.5)),  # Convert to Python float
            sentiment=sentiment_result["sentiment"],
            sentiment_confidence=float(sentiment_result["confidence"])  # Convert to Python float
        )
        
    except Exception as e:
        logger.error(f"Error analyzing single review: {e}")
        raise HTTPException(status_code=500, detail=f"Single review analysis failed: {str(e)}")

def analyze_single_return_sentiment(return_reason: str) -> Dict[str, Any]:
    """Analyze sentiment of a single return reason without aspect extraction"""
    if not aspect_extractor:
        raise HTTPException(status_code=500, detail="Aspect extractor not initialized")
    
    try:
        # Use aspect extractor just for sentiment analysis on single return
        result = aspect_extractor.extract_aspect(
            inference_source=[return_reason],
            pred_sentiment=True,
            print_result=False
        )
        
        # Extract sentiment from result
        sentiment = "Neutral"
        confidence = 0.5
        
        if result and len(result) > 0:
            entry = result[0]
            if isinstance(entry, dict) and 'sentiment' in entry:
                sentiments = entry['sentiment']
                if sentiments and len(sentiments) > 0:
                    # Get the most common sentiment
                    from collections import Counter
                    sentiment_counts = Counter(sentiments)
                    most_common = sentiment_counts.most_common(1)
                    if most_common:
                        sentiment = most_common[0][0]
                        confidence = most_common[0][1] / len(sentiments)
        
        return {
            "sentiment": sentiment,
            "confidence": float(confidence)  # Convert to Python float
        }
        
    except Exception as e:
        logger.error(f"Error analyzing return sentiment: {e}")
        return {
            "sentiment": "Neutral",
            "confidence": 0.5  # Already Python float
        }

def categorize_return_reason(return_reason: str) -> str:
    """Categorize return reason into predefined categories"""
    return_text = return_reason.lower()
    
    # Quality-related keywords
    quality_keywords = ['quality', 'defective', 'broken', 'damaged', 'poor', 'cheap', 'flimsy', 'tear', 'crack']
    # Size-related keywords  
    size_keywords = ['size', 'fit', 'small', 'large', 'tight', 'loose', 'wrong size']
    # Shipping-related keywords
    shipping_keywords = ['shipping', 'delivery', 'late', 'delay', 'damaged in transit', 'package']
    # Description mismatch
    description_keywords = ['not as described', 'different', 'misleading', 'false advertising']
    # Color/appearance
    appearance_keywords = ['color', 'colour', 'appearance', 'look', 'style']
    
    if any(keyword in return_text for keyword in quality_keywords):
        return "Quality"
    elif any(keyword in return_text for keyword in size_keywords):
        return "Size/Fit"
    elif any(keyword in return_text for keyword in shipping_keywords):
        return "Shipping"
    elif any(keyword in return_text for keyword in description_keywords):
        return "Description Mismatch"
    elif any(keyword in return_text for keyword in appearance_keywords):
        return "Appearance"
    else:
        return "Other"

def process_returns_for_aspects(returns: List[str], product_id: str) -> Dict[str, Any]:
    """Process return reasons to extract aspects and generate suggestions"""
    if not aspect_extractor:
        raise HTTPException(status_code=500, detail="Aspect extractor not initialized")
    
    try:
        # Extract aspects and sentiment
        extracted = aspect_extractor.extract_aspect(
            inference_source=returns,
            pred_sentiment=True,
            print_result=False
        )
        
        # Aggregate aspects by sentiment
        from collections import defaultdict, Counter
        aspect_summary = defaultdict(list)
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        return_categories = defaultdict(int)
        
        # Categorize returns
        for return_reason in returns:
            category = categorize_return_reason(return_reason)
            return_categories[category] += 1
        
        # Add null check for extracted results
        if extracted:
            for entry in extracted:
                if isinstance(entry, dict) and 'aspect' in entry and 'sentiment' in entry:
                    for asp, sent in zip(entry['aspect'], entry['sentiment']):
                        aspect_summary[sent].append(asp)
                        if sent in sentiment_counts:
                            sentiment_counts[sent] += 1
        
        # Get top aspects
        def top_aspects(aspect_list, max_count=10):
            if not aspect_list:
                return []
            return [item for item, _ in Counter(aspect_list).most_common(max_count)]
        
        positive_aspects = top_aspects(aspect_summary.get("Positive", []))
        negative_aspects = top_aspects(aspect_summary.get("Negative", []))
        
        # Generate suggestions using LLM if available
        suggestions = {"improvements": [], "strengths": []}
        if llm and (positive_aspects or negative_aspects or return_categories):
            try:
                prompt = PromptTemplate(
                    input_variables=["positives", "negatives", "categories"],
                    template="""Based on these product return analysis:

Positive aspects: {positives}
Negative aspects: {negatives}  
Return categories: {categories}

Give sellers actionable advice in two categories:
1. Improvements they should make (based on negative aspects and return patterns)
2. Strengths they should highlight (based on positive aspects)

Focus on addressing the main return reasons to reduce future returns.

Return only a JSON object with 'improvements' and 'strengths' keys, each containing an array of strings.
"""
                )
                
                chain = prompt | llm
                response = chain.invoke({
                    "positives": ", ".join(positive_aspects),
                    "negatives": ", ".join(negative_aspects),
                    "categories": ", ".join([f"{k}: {v}" for k, v in return_categories.items()])
                })
                
                # Parse LLM response with proper type checking
                try:
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    # Ensure response_text is a string
                    if isinstance(response_text, list):
                        response_text = " ".join(str(item) for item in response_text)
                    else:
                        response_text = str(response_text)
                    
                    # Extract JSON from response
                    import re
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        suggestions = json.loads(json_match.group())
                except:
                    logger.warning("Could not parse LLM response for return suggestions")
                    
            except Exception as e:
                logger.error(f"Error generating return suggestions: {e}")
        
        return {
            "positive_aspects": positive_aspects,
            "negative_aspects": negative_aspects,
            "return_categories": {k: int(v) for k, v in return_categories.items()},  # Convert to Python int
            "suggestions": suggestions,
            "sentiment_distribution": {k: int(v) for k, v in sentiment_counts.items()}  # Convert to Python int
        }
        
    except Exception as e:
        logger.error(f"Error processing returns: {e}")
        return {
            "positive_aspects": [],
            "negative_aspects": [],
            "return_categories": {},
            "suggestions": {"improvements": [], "strengths": []},
            "sentiment_distribution": {"Positive": 0, "Negative": 0, "Neutral": 0}  # Already Python ints
        }

@app.post("/analyze-single-return", response_model=SingleReturnAnalysisResponse)
async def analyze_single_return(request: SingleReturnAnalysisRequest):
    """
    Analyze a single return reason for fake detection and sentiment analysis.
    This is faster than batch analysis as it doesn't extract aspects.
    """
    if not request.return_reason.strip():
        raise HTTPException(status_code=400, detail="Return reason text cannot be empty")
    
    try:
        # 1. Fake return detection (using same model as reviews)
        fake_results = detect_fake_reviews([request.return_reason])
        fake_result = fake_results[0] if fake_results else {
            "is_fake": False,
            "fake_probability": 0.5,
            "confidence": 0.0
        }
        
        # 2. Sentiment analysis (without aspect extraction)
        sentiment_result = analyze_single_return_sentiment(request.return_reason)
        
        # 3. Categorize return reason
        return_category = categorize_return_reason(request.return_reason)
        
        return SingleReturnAnalysisResponse(
            product_id=request.product_id,
            return_id=request.return_id,
            return_text=request.return_reason[:200] + "..." if len(request.return_reason) > 200 else request.return_reason,
            is_fake=fake_result.get("is_fake", False),
            fake_probability=float(fake_result.get("fake_probability", 0.5)),  # Convert to Python float
            sentiment=sentiment_result["sentiment"],
            sentiment_confidence=float(sentiment_result["confidence"]),  # Convert to Python float
            return_category=return_category
        )
        
    except Exception as e:
        logger.error(f"Error analyzing single return: {e}")
        raise HTTPException(status_code=500, detail=f"Single return analysis failed: {str(e)}")

@app.post("/analyze-returns-batch", response_model=BatchReturnAnalysisResponse)
async def analyze_returns_batch(request: BatchReturnAnalysisRequest):
    """
    Analyze multiple return reasons for comprehensive analysis including aspects, sentiment, and patterns.
    This provides detailed insights for business intelligence and return reduction strategies.
    """
    if not request.returns:
        raise HTTPException(status_code=400, detail="Returns list cannot be empty")
    
    try:
        # 1. Process returns for aspects and categorization
        analysis_result = process_returns_for_aspects(request.returns, request.product_id)
        
        # 2. Fake return detection
        detection_results = detect_fake_reviews(request.returns)  # Reuse fake review detection
        fake_count = sum(1 for result in detection_results if result.get("is_fake", False))
        
        return BatchReturnAnalysisResponse(
            product_id=request.product_id,
            total_returns=int(len(request.returns)),  # Convert to Python int
            positive_aspects=analysis_result["positive_aspects"],
            negative_aspects=analysis_result["negative_aspects"],
            return_categories=convert_numpy_types(analysis_result["return_categories"]),  # Convert numpy types
            suggestions=convert_numpy_types(analysis_result["suggestions"]),  # Convert numpy types
            sentiment_distribution=convert_numpy_types(analysis_result["sentiment_distribution"]),  # Convert numpy types
            fake_returns_detected=int(fake_count),  # Convert to Python int
            fake_return_percentage=float((fake_count / len(request.returns)) * 100)  # Convert to Python float
        )
        
    except Exception as e:
        logger.error(f"Error analyzing returns batch: {e}")
        raise HTTPException(status_code=500, detail=f"Batch return analysis failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 