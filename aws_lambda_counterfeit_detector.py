#!/usr/bin/env python3
"""
AWS Lambda Function for Counterfeit Detection with Image Processing
Integrates SimpleImageOps and EnhancedCounterfeitDetector functionality
"""

import json
import base64
import io
import os
import tempfile
import uuid
from typing import Dict, Any, Optional
import boto3
from PIL import Image
import numpy as np
import cv2

# Import our modules
from simple_image_ops import SimpleImageOps
from enhanced_counterfeit_detector_1 import EnhancedCounterfeitDetector, load_image
import asyncio
import logging

# Configure logging for Lambda
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class LambdaCounterfeitDetector:
    """AWS Lambda-optimized counterfeit detection system"""
    
    def __init__(self):
        self.image_ops = SimpleImageOps()
        self.counterfeit_detector = EnhancedCounterfeitDetector("./my_model/my_model.pt")
        
    def process_base64_image(self, base64_image: str, filename: str|None = None) -> Optional[bytes]:
        """Process base64 encoded image and return as bytes"""
        try:
            # Decode base64 image
            if base64_image.startswith('data:image'):
                # Remove data URL prefix
                base64_image = base64_image.split(',')[1]
            
            image_data = base64.b64decode(base64_image)
            
            logger.info(f"✅ Processed base64 image to bytes (size: {len(image_data)} bytes)")
            return image_data
            
        except Exception as e:
            logger.error(f"❌ Failed to process base64 image: {e}")
            return None
    
    def process_s3_image(self, bucket: str, key: str) -> Optional[bytes]:
        """Download image from S3 and return as bytes"""
        try:
            # Use region from environment variable or default
            region = os.getenv('AWS_REGION', 'ap-south-1')
            s3_client = boto3.client('s3', region_name=region)
            
            # Download to memory
            buffer = io.BytesIO()
            s3_client.download_fileobj(bucket, key, buffer)
            image_data = buffer.getvalue()
            
            logger.info(f"✅ Downloaded S3 image to memory (size: {len(image_data)} bytes)")
            return image_data
            
        except Exception as e:
            logger.error(f"❌ Failed to download S3 image: {e}")
            return None
    
    def bytes_to_image_array(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Convert image bytes to numpy array for counterfeit detection"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Convert to numpy array
            image_array = np.array(image)
            return image_array
        except Exception as e:
            logger.error(f"❌ Failed to convert bytes to image array: {e}")
            return None
    
    def create_temp_file_from_bytes(self, image_bytes: bytes, suffix: str = '.jpg') -> Optional[str]:
        """Create temporary file from image bytes for functions that require file paths"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(image_bytes)
                temp_path = temp_file.name
            return temp_path
        except Exception as e:
            logger.error(f"❌ Failed to create temp file from bytes: {e}")
            return None

    async def analyze_image(self, image_data: bytes, description: str = "", filename: str = "image.jpg") -> Dict[str, Any]:
        """Perform complete image analysis including similarity search and counterfeit detection"""
        temp_file_path = None
        try:
            # Create temporary file for functions that need file paths
            temp_file_path = self.create_temp_file_from_bytes(image_data)
            if not temp_file_path:
                raise Exception("Failed to create temporary file from image data")
            
            # Step 1: Upload to S3 and Pinecone for similarity search
            logger.info("🔍 Starting similarity search...")
            s3_url = self.image_ops.upload_to_s3(temp_file_path)
            
            if s3_url:
                # Upload to Pinecone with S3 URL
                image_id = self.image_ops.upload_image(
                    image_path=temp_file_path,
                    text_description=f"Product analysis - {description}",
                    s3_url=s3_url
                )
                
                # Search for similar images
                similar_images = self.image_ops.search_similar_images(temp_file_path, top_k=5)
                
                similarity_results = []
                for match in similar_images:
                    similarity_results.append({
                        'id': match.id,
                        'score': float(match.score),
                        'filename': match.metadata.get('filename', 'Unknown'),
                        's3_url': match.metadata.get('s3_url', ''),
                        'description': match.metadata.get('text_description', '')
                    })
            else:
                similarity_results = []
                image_id = None
            
            # Step 2: Counterfeit detection analysis
            logger.info("🔍 Starting counterfeit detection...")
            image_array = self.bytes_to_image_array(image_data)
            
            if image_array is not None:
                counterfeit_result = await self.counterfeit_detector.analyze_counterfeit(
                    image_array, description
                )
                
                # Prepare counterfeit analysis results
                counterfeit_analysis = {
                    'is_counterfeit': counterfeit_result.is_counterfeit,
                    'confidence': float(counterfeit_result.confidence),
                    'brand_detected': counterfeit_result.brand_detected,
                    'detected_issues': counterfeit_result.detected_issues,
                    'logo_similarities': [float(sim) for sim in counterfeit_result.logo_similarities],
                    'distortion_scores': [float(score) for score in counterfeit_result.distortion_scores],
                    'num_input_logos': len(counterfeit_result.input_logos),
                    'num_reference_images': len([img for img in counterfeit_result.reference_images if img.download_success]),
                    'analysis_details': {
                        'avg_logo_similarity': float(counterfeit_result.analysis_details.get('avg_logo_similarity', 0.0)),
                        'avg_distortion_score': float(counterfeit_result.analysis_details.get('avg_distortion_score', 0.0))
                    }
                }
            else:
                counterfeit_analysis = {
                    'is_counterfeit': True,
                    'confidence': 0.1,
                    'brand_detected': 'unknown',
                    'detected_issues': ['Failed to load image'],
                    'logo_similarities': [],
                    'distortion_scores': [],
                    'num_input_logos': 0,
                    'num_reference_images': 0,
                    'analysis_details': {
                        'avg_logo_similarity': 0.0,
                        'avg_distortion_score': 0.0
                    }
                }
            
            # Step 3: Combine results
            combined_results = {
                'image_processing': {
                    's3_url': s3_url,
                    'pinecone_id': image_id,
                    'upload_success': s3_url is not None
                },
                'similarity_search': {
                    'similar_images_found': len(similarity_results),
                    'results': similarity_results
                },
                'counterfeit_detection': counterfeit_analysis,
                'overall_scores': {
                    'authenticity_score': 1.0 - float(counterfeit_analysis['confidence']) if counterfeit_analysis['is_counterfeit'] else float(counterfeit_analysis['confidence']),
                    'similarity_score': max([r['score'] for r in similarity_results]) if similarity_results else 0.0,
                    'quality_score': 1.0 - float(counterfeit_analysis['analysis_details']['avg_distortion_score']),
                    'brand_confidence': 1.0 if counterfeit_analysis['brand_detected'] != 'unknown' else 0.0
                }
            }
            
            return combined_results
            
        except Exception as e:
            logger.error(f"❌ Error in image analysis: {e}")
            return {
                'error': str(e),
                'image_processing': {'upload_success': False},
                'similarity_search': {'similar_images_found': 0, 'results': []},
                'counterfeit_detection': {
                    'is_counterfeit': True,
                    'confidence': 0.1,
                    'brand_detected': 'unknown',
                    'detected_issues': [f'Analysis error: {str(e)}']
                },
                'overall_scores': {
                    'authenticity_score': 0.0,
                    'similarity_score': 0.0,
                    'quality_score': 0.0,
                    'brand_confidence': 0.0
                }
            }
        finally:
            # Clean up temporary file
            if temp_file_path:
                self.cleanup_temp_file(temp_file_path)
    
    def cleanup_temp_file(self, file_path: str):
        """Clean up temporary files"""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"🗑️ Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.error(f"❌ Failed to cleanup temp file {file_path}: {e}")

# Global instance for Lambda reuse
detector_instance = None

def lambda_handler(event, context):
    """
    AWS Lambda handler for counterfeit detection
    
    Expected event format:
    {
        "image": {
            "type": "base64" | "s3",
            "data": "base64_encoded_image" | {"bucket": "bucket_name", "key": "object_key"},
            "filename": "optional_filename.jpg"
        },
        "description": "Product description (optional)",
        "analysis_type": "full" | "similarity_only" | "counterfeit_only"
    }
    """
    global detector_instance
    
    try:
        # Initialize detector if not already done
        if detector_instance is None:
            logger.info("🚀 Initializing detector...")
            detector_instance = LambdaCounterfeitDetector()
        
        # Parse event
        image_config = event.get('image', {})
        description = event.get('description', '')
        analysis_type = event.get('analysis_type', 'full')
        
        if not image_config:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'No image configuration provided',
                    'success': False
                })
            }
        
        # Process image based on type
        image_data = None
        filename = "image.jpg"
        
        if image_config.get('type') == 'base64':
            base64_data = image_config.get('data')
            filename = image_config.get('filename', f"{uuid.uuid4()}.jpg")
            image_data = detector_instance.process_base64_image(base64_data, filename)
            
        elif image_config.get('type') == 's3':
            s3_config = image_config.get('data', {})
            bucket = s3_config.get('bucket')
            key = s3_config.get('key')
            filename = key.split('/')[-1] if key else "s3_image.jpg"
            image_data = detector_instance.process_s3_image(bucket, key)
            
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Invalid image type. Must be "base64" or "s3"',
                    'success': False
                })
            }
        
        if not image_data:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': 'Failed to process image',
                    'success': False
                })
            }
        
        # Run analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                detector_instance.analyze_image(image_data, description, filename)
            )
        finally:
            loop.close()
        
        # Return results
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': True,
                'analysis_type': analysis_type,
                'description': description,
                'results': results
            }, indent=2)
        }
        
    except Exception as e:
        logger.error(f"❌ Lambda handler error: {e}")
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f'Internal server error: {str(e)}',
                'success': False
            })
        }

# For local testing
if __name__ == "__main__":
    # Test event
    test_event = {
        "image": {
            "type": "s3",
            "data": {
                "bucket": "images-hackonn-fashion",
                "key": "images/10003.jpg"
            }
        },
        "description": "PUMA t-shirt",
        "analysis_type": "full"
    }
    
    # Mock context
    class MockContext:
        def __init__(self):
            self.function_name = "test"
            self.memory_limit_in_mb = 512
            self.invoked_function_arn = "test"
            self.aws_request_id = "test"
    
    context = MockContext()
    
    # Run test
    result = lambda_handler(test_event, context)
    print(json.dumps(result, indent=2)) 