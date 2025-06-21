# AWS Lambda Counterfeit Detection - Deployment Guide

## 🚀 Quick Success Summary

✅ **Package Installation**: Fixed! Basic requirements now install successfully  
❌ **AWS Permissions**: Need to be configured  
📦 **Package Size**: 89.8 MB (within Lambda limits)  

## 🔧 Step-by-Step Solution

### Step 1: Fix AWS Permissions

You need to attach the proper IAM policy to your AWS user. You have two options:

#### Option A: Use AWS Console (Recommended)
1. Go to [AWS IAM Console](https://console.aws.amazon.com/iam/)
2. Click "Users" → Find your user (`superadmin`)
3. Click "Add permissions" → "Attach policies directly"
4. Create a new policy using the JSON from `aws_iam_policy.json`
5. Attach the policy to your user

#### Option B: Use AWS CLI
```bash
# Create the policy
aws iam create-policy \
    --policy-name CounterfeitDetectionLambdaPolicy \
    --policy-document file://aws_iam_policy.json

# Attach to your user
aws iam attach-user-policy \
    --user-name superadmin \
    --policy-arn arn:aws:iam::207567796032:policy/CounterfeitDetectionLambdaPolicy
```

### Step 2: Create Lambda Execution Role

Create an IAM role for Lambda execution:

```bash
# Create trust policy for Lambda
cat > lambda-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create the role
aws iam create-role \
    --role-name lambda-execution-role \
    --assume-role-policy-document file://lambda-trust-policy.json

# Attach basic Lambda execution policy
aws iam attach-role-policy \
    --role-name lambda-execution-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Attach S3 access policy
aws iam attach-role-policy \
    --role-name lambda-execution-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### Step 3: Deploy with Basic Requirements

The basic requirements file works perfectly! Deploy using:

```bash
python deploy_lambda.py
```

### Step 4: Add ML Dependencies via Lambda Layers

Since the basic deployment works, add ML libraries through Lambda layers:

#### Create PyTorch Layer
```bash
# Download PyTorch for Lambda
mkdir pytorch-layer
cd pytorch-layer
pip install torch torchvision --target python/lib/python3.9/site-packages/
zip -r pytorch-layer.zip python/

# Upload to Lambda
aws lambda publish-layer-version \
    --layer-name pytorch \
    --zip-file fileb://pytorch-layer.zip \
    --compatible-runtimes python3.9
```

#### Create Transformers Layer
```bash
# Download Transformers
mkdir transformers-layer
cd transformers-layer
pip install transformers --target python/lib/python3.9/site-packages/
zip -r transformers-layer.zip python/

# Upload to Lambda
aws lambda publish-layer-version \
    --layer-name transformers \
    --zip-file fileb://transformers-layer.zip \
    --compatible-runtimes python3.9
```

## 🔥 Quick Fix Commands

### 1. Check Current AWS Configuration
```bash
aws sts get-caller-identity
aws iam list-attached-user-policies --user-name superadmin
```

### 2. Test Basic Lambda Function
```bash
# Test the function with basic requirements
python aws_lambda_counterfeit_detector.py
```

### 3. Add Environment Variables
```bash
# Set required environment variables
export AWS_REGION=ap-south-1
export S3_BUCKET_NAME=images-hackonn-fashion
export PINECONE_API_KEY=your_pinecone_key
export GAPIS_API_KEY=your_google_key
export GCSE_ID=your_search_engine_id
```

## 📋 Troubleshooting

### Issue: AccessDeniedException
**Solution**: Attach the IAM policy from `aws_iam_policy.json` to your user

### Issue: Package too large
**Solution**: Use Lambda layers for heavy dependencies (already implemented)

### Issue: Region not specified
**Solution**: Set `AWS_REGION=ap-south-1` in environment variables

### Issue: Dependencies fail to install
**Solution**: Use `lambda_requirements_basic.txt` (already working)

## 🎯 Current Status

✅ **Requirements**: Working with basic packages  
✅ **Code**: All source files ready  
✅ **Package Size**: 89.8 MB (acceptable)  
❌ **Permissions**: Need IAM policy attachment  
⏳ **ML Dependencies**: Will add via layers after basic deployment  

## 🚀 Next Steps

1. **Fix Permissions**: Attach IAM policy to your AWS user
2. **Deploy Basic**: Run `python deploy_lambda.py` 
3. **Add Layers**: Create Lambda layers for ML dependencies
4. **Test Function**: Invoke Lambda with test payload
5. **Create API**: Set up API Gateway (optional)

## 📞 Support Commands

```bash
# Check deployment status
python setup_aws_config.py

# Test package installation
python test_install.py

# Deploy with verbose output
python deploy_lambda.py --verbose

# Test locally
python aws_lambda_counterfeit_detector.py
```

---

**Note**: The package installation errors are now resolved! Focus on fixing AWS permissions to complete the deployment. 