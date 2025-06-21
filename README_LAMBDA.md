# AWS Lambda Counterfeit Detection System

A comprehensive counterfeit product detection system built for AWS Lambda that integrates computer vision, similarity search, and AI-powered analysis.

## 🚀 Features

### Core Capabilities
- **Counterfeit Detection**: Advanced AI analysis using YOLO, CLIP, and BLIP models
- **Similarity Search**: Vector-based image similarity using Pinecone
- **Brand Recognition**: Automatic brand detection from product images
- **Logo Analysis**: Logo distortion and quality assessment
- **Web Search Integration**: Google Custom Search for reference images
- **S3 Integration**: Automatic image storage with public URLs
- **Comprehensive Scoring**: Multiple quality and authenticity metrics

### AWS Lambda Optimizations
- **Serverless Architecture**: Pay-per-use model with auto-scaling
- **Multiple Input Methods**: Base64 and S3 image input support
- **Efficient Resource Management**: Optimized memory and timeout settings
- **Error Handling**: Comprehensive error handling and logging
- **CORS Support**: Ready for web application integration

## 📋 Prerequisites

### AWS Services Required
- AWS Lambda
- AWS S3
- AWS IAM (for roles and permissions)
- AWS API Gateway (optional, for REST API)

### External Services
- **Pinecone**: Vector database for similarity search
- **Google Custom Search API**: For reference image searching

### Local Dependencies (for deployment)
- Python 3.9+
- AWS CLI configured
- Boto3

## 🛠️ Installation & Setup

### 1. Install Dependencies
```bash
pip install -r lambda_requirements.txt
```

### 2. Environment Variables
Create a `.env` file or configure AWS Lambda environment variables:

```env
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key

# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=ap-south-1
S3_BUCKET_NAME=images-hackonn-fashion

# Google Custom Search
GAPIS_API_KEY=your_google_apis_key
GCSE_ID=your_google_custom_search_engine_id
```

### 3. Model Files
Ensure your YOLO model is available:
```
my_model/
  └── my_model.pt
```

## 📦 Deployment

### Option 1: Automated Deployment
```bash
python deploy_lambda.py
```

### Option 2: Manual Deployment
1. **Package Dependencies**:
   ```bash
   pip install -r lambda_requirements.txt -t lambda_package/
   ```

2. **Copy Source Files**:
   ```bash
   cp aws_lambda_counterfeit_detector.py lambda_package/
   cp simple_image_ops.py lambda_package/
   cp enhanced_counterfeit_detector.py lambda_package/
   cp -r my_model/ lambda_package/
   ```

3. **Create ZIP Package**:
   ```bash
   cd lambda_package
   zip -r ../counterfeit_detection_lambda.zip .
   ```

4. **Deploy to AWS Lambda**:
   - Upload ZIP file through AWS Console
   - Configure environment variables
   - Set handler: `aws_lambda_counterfeit_detector.lambda_handler`
   - Set timeout: 300 seconds
   - Set memory: 3008 MB

## 🔧 Configuration

### Lambda Function Settings
- **Runtime**: Python 3.9
- **Handler**: `aws_lambda_counterfeit_detector.lambda_handler`
- **Timeout**: 300 seconds (5 minutes)
- **Memory**: 3008 MB (maximum)
- **Ephemeral Storage**: 2048 MB

### IAM Permissions Required
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:PutBucketPolicy",
                "s3:CreateBucket",
                "s3:HeadBucket"
            ],
            "Resource": [
                "arn:aws:s3:::images-hackonn-fashion",
                "arn:aws:s3:::images-hackonn-fashion/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        }
    ]
}
```

## 📡 API Usage

### Event Format

#### Base64 Image Input
```json
{
    "image": {
        "type": "base64",
        "data": "iVBORw0KGgoAAAANSUhEUgAA...",
        "filename": "product.jpg"
    },
    "description": "PUMA t-shirt green",
    "analysis_type": "full"
}
```

#### S3 Image Input
```json
{
    "image": {
        "type": "s3",
        "data": {
            "bucket": "my-images-bucket",
            "key": "products/image123.jpg"
        }
    },
    "description": "Nike shoes authentic",
    "analysis_type": "full"
}
```

### Response Format
```json
{
    "statusCode": 200,
    "body": {
        "success": true,
        "analysis_type": "full",
        "description": "PUMA t-shirt green",
        "results": {
            "image_processing": {
                "s3_url": "https://bucket.s3.amazonaws.com/images/file.jpg",
                "pinecone_id": "uuid-string",
                "upload_success": true
            },
            "similarity_search": {
                "similar_images_found": 5,
                "results": [
                    {
                        "id": "uuid",
                        "score": 0.95,
                        "filename": "similar_product.jpg",
                        "s3_url": "https://...",
                        "description": "Fashion item - similar_product.jpg"
                    }
                ]
            },
            "counterfeit_detection": {
                "is_counterfeit": false,
                "confidence": 0.85,
                "brand_detected": "puma",
                "detected_issues": [],
                "logo_similarities": [0.9, 0.87],
                "distortion_scores": [0.1, 0.15],
                "num_input_logos": 2,
                "num_reference_images": 3,
                "analysis_details": {
                    "avg_logo_similarity": 0.885,
                    "avg_distortion_score": 0.125
                }
            },
            "overall_scores": {
                "authenticity_score": 0.85,
                "similarity_score": 0.95,
                "quality_score": 0.875,
                "brand_confidence": 1.0
            }
        }
    }
}
```

## 📊 Scoring System

### Overall Scores Explanation

1. **Authenticity Score** (0.0 - 1.0)
   - Based on counterfeit detection confidence
   - Higher score = more likely authentic

2. **Similarity Score** (0.0 - 1.0)
   - Best match from similarity search
   - Higher score = more similar to known products

3. **Quality Score** (0.0 - 1.0)
   - Based on logo quality and distortion analysis
   - Higher score = better quality logos/images

4. **Brand Confidence** (0.0 - 1.0)
   - Confidence in brand detection
   - 1.0 = brand detected, 0.0 = unknown brand

## 🔍 Analysis Types

- **`full`**: Complete analysis including similarity search and counterfeit detection
- **`similarity_only`**: Only perform similarity search
- **`counterfeit_only`**: Only perform counterfeit detection

## 🚨 Error Handling

Common error responses:
```json
{
    "statusCode": 400,
    "body": {
        "error": "No image configuration provided",
        "success": false
    }
}
```

```json
{
    "statusCode": 500,
    "body": {
        "error": "Internal server error: detailed error message",
        "success": false
    }
}
```

## 🧪 Local Testing

Test the Lambda function locally:
```bash
python aws_lambda_counterfeit_detector.py
```

## 📈 Performance Considerations

### Optimization Tips
1. **Cold Start**: First invocation may take 30-60 seconds
2. **Model Loading**: Models are cached after first load
3. **Concurrent Executions**: Limited to 10 by default
4. **Memory Usage**: 3008 MB recommended for optimal performance
5. **Timeout**: 300 seconds to handle complex analysis

### Cost Optimization
- Use reserved concurrency for predictable workloads
- Consider provisioned concurrency for low-latency requirements
- Monitor CloudWatch metrics for optimization opportunities

## 🔐 Security

### Best Practices
- Store sensitive environment variables in AWS Secrets Manager
- Use IAM roles with minimal required permissions
- Enable AWS CloudTrail for audit logging
- Implement API rate limiting if using API Gateway

## 📝 Monitoring

### CloudWatch Metrics
- Function duration
- Error rate
- Memory utilization
- Concurrent executions

### Custom Logging
The function includes comprehensive logging for:
- Image processing steps
- Model inference timing
- Error tracking
- Performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check CloudWatch logs for error details
2. Verify environment variables are correctly set
3. Ensure all required AWS permissions are configured
4. Test with simple images first

---

**Note**: This system requires significant computational resources and may incur costs from AWS Lambda, S3, Pinecone, and Google APIs. Monitor usage and costs accordingly. 