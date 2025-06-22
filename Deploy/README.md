# Product Authentication & Review Analysis API

A comprehensive FastAPI backend that integrates:
- **Counterfeit Product Detection** using YOLO and CLIP models
- **Review Analysis** with aspect-based sentiment analysis
- **Fake Review Detection** using BERT models

## 🚀 Features

### 🔍 Counterfeit Detection
- Logo detection and comparison using YOLO v8
- Web image search for brand references using Google Custom Search
- Feature extraction and similarity analysis using CLIP
- Distortion and quality scoring
- Comprehensive analysis reports

### 📝 Review Analysis
- Aspect-based sentiment analysis using PyABSA
- Positive/negative aspect extraction
- LLM-powered improvement suggestions using Groq
- Sentiment distribution analysis

### 🕵️ Fake Review Detection
- BERT-based fake review classification
- Batch processing for efficiency
- Confidence scoring for each review

### 🔄 Batch Analysis
- Combined analysis of product images and reviews
- Single endpoint for comprehensive product evaluation

## 📋 API Endpoints

### Core Endpoints
- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /detect-counterfeit` - Analyze product images for counterfeiting
- `POST /upload-image-counterfeit` - Upload image files for analysis
- `POST /analyze-reviews` - Extract aspects and generate suggestions from reviews
- `POST /detect-fake-reviews` - Detect fake reviews in batch
- `POST /batch-analysis` - Comprehensive analysis combining all features


## ☁️ AWS Deployment

### Prerequisites
- AWS CLI installed and configured
- Docker installed
- Sufficient AWS permissions for ECS, ECR, IAM, and Systems Manager

### Quick Deployment

1. **Make deployment script executable**
   ```bash
   chmod +x deploy.sh
   ```

2. **Run deployment script**
   ```bash
   ./deploy.sh
   ```

3. **Follow the prompts** to:
   - Store API keys in AWS Parameter Store
   - Configure networking (subnets, security groups)
   - Deploy to ECS

### Manual AWS Setup

#### 1. Store Secrets in Parameter Store
```bash
aws ssm put-parameter \
    --name '/product-auth/groq-api-key' \
    --value 'your_groq_api_key' \
    --type 'SecureString'

aws ssm put-parameter \
    --name '/product-auth/google-api-key' \
    --value 'your_google_api_key' \
    --type 'SecureString'

aws ssm put-parameter \
    --name '/product-auth/google-cse-id' \
    --value 'your_google_cse_id' \
    --type 'SecureString'
```

#### 2. Create Required IAM Roles

**ECS Task Execution Role:**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "ssm:GetParameters",
                "ssm:GetParameter"
            ],
            "Resource": "*"
        }
    ]
}
```

#### 3. Build and Push to ECR
```bash
# Get account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=us-east-1

# Create ECR repository
aws ecr create-repository --repository-name product-auth-api

# Build and tag image
docker build -t product-auth-api .
docker tag product-auth-api:latest $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/product-auth-api:latest

# Login and push
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/product-auth-api:latest
```

#### 4. Deploy to ECS
- Create ECS cluster with Fargate
- Create task definition using the provided template
- Create service with desired configuration

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GROQ_API_KEY` | Groq API key for LLM suggestions | Yes | - |
| `GAPIS_API_KEY` | Google Custom Search API key | Yes | - |
| `GCSE_ID` | Google Custom Search Engine ID | Yes | - |
| `PORT` | Application port | No | 8000 |
| `LOG_LEVEL` | Logging level | No | INFO |

### Model Requirements

The application requires:
1. **YOLO Model**: Custom trained model for logo detection (`my_model.pt`)
2. **BERT Model**: Downloaded automatically for fake review detection
3. **CLIP Model**: Downloaded automatically for feature extraction
4. **BLIP Model**: Downloaded automatically for brand detection
5. **PyABSA Models**: Downloaded automatically for aspect extraction

### Resource Requirements

| Component | CPU | Memory | Storage |
|-----------|-----|---------|---------|
| Development | 2 cores | 4GB | 10GB |
| Production | 4 cores | 8GB | 20GB |
| With GPU | 4 cores | 8GB | 20GB |

## 🔍 Monitoring and Debugging

### Logs
- Application logs: CloudWatch Logs `/ecs/product-auth`
- Access logs: Available through API endpoints
- Error tracking: Built-in error handling and logging

### Health Checks
- **Basic**: `GET /`
- **Detailed**: `GET /health` (shows model initialization status)

### Performance Monitoring
- Response times logged for all endpoints
- Model inference times tracked
- Memory usage monitoring available

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure YOLO model file exists in `/app/models/`
   - Check memory allocation (8GB+ recommended)
   - Verify model file permissions

2. **API Key Issues**
   - Verify all required API keys are set
   - Check Parameter Store permissions in AWS
   - Ensure API keys are valid and have sufficient quotas

3. **Memory Issues**
   - Increase container memory allocation
   - Use GPU-enabled instances for better performance
   - Consider model quantization for reduced memory usage

4. **Network Issues**
   - Check security group rules for port 8000
   - Verify load balancer configuration
   - Ensure internet access for model downloads

### Performance Optimization

1. **Use GPU instances** for faster inference
2. **Enable model caching** to reduce cold start times
3. **Implement connection pooling** for external APIs
4. **Use CDN** for static assets and model files

## 📝 API Response Formats

### Counterfeit Detection Response
```json
{
    "is_counterfeit": true,
    "confidence": 0.85,
    "brand_detected": "nike",
    "detected_issues": ["Low logo similarity", "High distortion"],
    "logo_count": 2,
    "analysis_summary": {
        "logo_similarities": [0.65, 0.72],
        "distortion_scores": [0.45, 0.38],
        "reference_images_found": 3
    }
}
```

### Review Analysis Response
```json
{
    "product_id": "PROD123",
    "total_reviews": 50,
    "positive_aspects": ["quality", "comfort", "style"],
    "negative_aspects": ["price", "durability"],
    "suggestions": {
        "improvements": ["Improve durability", "Consider pricing"],
        "strengths": ["Highlight comfort", "Emphasize style"]
    },
    "sentiment_distribution": {
        "Positive": 35,
        "Negative": 10,
        "Neutral": 5
    }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section above
- Review API documentation at `/docs` endpoint

## 🔄 Version History

- **v1.0.0**: Initial release with all core features
- Counterfeit detection with YOLO and CLIP
- Review analysis with PyABSA
- Fake review detection with BERT
- AWS deployment support 