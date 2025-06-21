#!/usr/bin/env python3
"""
FastAPI Backend for Product Authentication and Review Analysis
Integrates counterfeit detection, review processing, and fake review detection
"""

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
from pydantic import BaseModel, Field
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
            llm = ChatGroq(model="llama3-70b-8192", api_key=groq_api_key)
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
                    "fake_probability": fake_prob,
                    "confidence": max(fake_prob, 1 - fake_prob)
                })
                
        except Exception as e:
            logger.error(f"Error processing batch {i}: {e}")
            # Add error results for this batch
            for review in batch_reviews:
                results.append({
                    "review": review[:100] + "..." if len(review) > 100 else review,
                    "is_fake": False,
                    "fake_probability": 0.5,
                    "confidence": 0.0,
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
                
                # Parse LLM response
                try:
                    response_text = response.content if hasattr(response, 'content') else str(response)
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
            "sentiment_distribution": sentiment_counts
        }
        
    except Exception as e:
        logger.error(f"Error processing reviews: {e}")
        return {
            "positive_aspects": [],
            "negative_aspects": [],
            "suggestions": {"improvements": [], "strengths": []},
            "sentiment_distribution": {"Positive": 0, "Negative": 0, "Neutral": 0}
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
    
    # Validate file type
    if not file.content_type.startswith('image/'):
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
            total_reviews=len(request.reviews),
            positive_aspects=analysis_result["positive_aspects"],
            negative_aspects=analysis_result["negative_aspects"],
            suggestions=analysis_result["suggestions"],
            sentiment_distribution=analysis_result["sentiment_distribution"]
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
            total_reviews=len(request.reviews),
            fake_reviews_detected=fake_count,
            fake_review_percentage=(fake_count / len(request.reviews)) * 100,
            review_authenticity=detection_results
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
        if request.image_base64:
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
            # Default response if no image
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
                total_reviews=len(request.reviews),
                fake_reviews_detected=fake_count,
                fake_review_percentage=(fake_count / len(request.reviews)) * 100,
                review_authenticity=detection_results
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

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    ) 