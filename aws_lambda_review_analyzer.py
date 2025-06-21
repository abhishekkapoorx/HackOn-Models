import json
import torch
import boto3
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers.pipelines import pipeline as pipeline_import
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global variables to store models (loaded once during cold start)
fake_review_tokenizer = None
fake_review_model = None
sentiment_pipeline = None
device = None

def initialize_models():
    """Initialize models during cold start"""
    global fake_review_tokenizer, fake_review_model, sentiment_pipeline, device
    
    if fake_review_tokenizer is None:
        logger.info("Initializing models...")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load fake review detection model
        fake_review_tokenizer = AutoTokenizer.from_pretrained("SravaniNirati/bert_fake_review_detection")
        fake_review_model = AutoModelForSequenceClassification.from_pretrained(
            "SravaniNirati/bert_fake_review_detection"
        ).to(device)
        
        # Load sentiment analysis model
        sentiment_model_name = "yangheng/deberta-v3-base-absa-v1.1"
        sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name, trust_remote_code=True)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name, trust_remote_code=True)
        
        # Create sentiment pipeline
        sentiment_pipeline = pipeline(
            "text-classification",
            model=sentiment_model,
            tokenizer=sentiment_tokenizer,
            return_all_scores=True,
            trust_remote_code=True
        )
        
        logger.info("Models initialized successfully")

def predict_fake_review(text):
    """
    Predict if a review is fake
    Returns probability of being fake (0-1)
    """
    try:
        # Tokenize input
        inputs = fake_review_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = fake_review_model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            
            # Get probability of being fake (class 0 in this model means fake)
            fake_probability = probabilities[0][0].cpu().numpy().item()
            
        return float(fake_probability)
        
    except Exception as e:
        logger.error(f"Error in fake review prediction: {str(e)}")
        return 0.5  # Return neutral probability on error

def get_sentiment_probabilities(text):
    """
    Get sentiment analysis probabilities
    Returns dictionary with sentiment scores
    """
    try:
        # Truncate text to avoid token limits
        truncated_text = text[:512]
        
        # Get sentiment prediction
        result = sentiment_pipeline(truncated_text)
        
        # Convert to standardized format
        sentiment_scores = {}
        for item in result:
            label = item['label'].lower()
            score = float(item['score'])
            sentiment_scores[label] = score
            
        return sentiment_scores
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

def lambda_handler(event, context):
    """
    AWS Lambda handler function
    Expected input: {"text": "review text here"}
    Returns: {
        "fake_probability": float,
        "sentiment_probabilities": dict,
        "status": "success"/"error",
        "message": str
    }
    """
    try:
        # Initialize models if not already loaded
        initialize_models()
        
        # Parse input
        if isinstance(event, str):
            event = json.loads(event)
            
        text = event.get('text', '')
        
        if not text:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'status': 'error',
                    'message': 'No text provided'
                })
            }
        
        # Get predictions
        fake_prob = predict_fake_review(text)
        sentiment_probs = get_sentiment_probabilities(text)
        
        # Prepare response
        response = {
            'fake_probability': fake_prob,
            'sentiment_probabilities': sentiment_probs,
            'status': 'success',
            'message': 'Analysis completed successfully'
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response)
        }
        
    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'message': f'Internal server error: {str(e)}'
            })
        }

# For local testing
if __name__ == "__main__":
    # Test the function locally
    test_event = {
        "text": "This product is absolutely amazing! Best purchase ever. 5 stars!!!"
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2)) 