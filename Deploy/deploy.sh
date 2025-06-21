#!/bin/bash

# AWS FastAPI Deployment Script
# This script builds and deploys the FastAPI application to AWS ECS

set -e

# Configuration
AWS_REGION=${AWS_REGION:-us-east-1}
ECR_REPOSITORY_NAME="product-auth-api"
ECS_CLUSTER_NAME="product-auth-cluster"
ECS_SERVICE_NAME="product-auth-service"
ECS_TASK_DEFINITION="product-auth-task"
IMAGE_TAG=${IMAGE_TAG:-latest}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if AWS CLI is installed and configured
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        echo_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        echo_error "AWS CLI is not configured. Please run 'aws configure' first."
        exit 1
    fi
    
    echo_info "AWS CLI is properly configured"
}

# Get AWS account ID
get_account_id() {
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    echo_info "AWS Account ID: $ACCOUNT_ID"
}

# Create ECR repository if it doesn't exist
create_ecr_repository() {
    echo_info "Creating ECR repository if it doesn't exist..."
    
    if aws ecr describe-repositories --repository-names $ECR_REPOSITORY_NAME --region $AWS_REGION &> /dev/null; then
        echo_info "ECR repository $ECR_REPOSITORY_NAME already exists"
    else
        aws ecr create-repository \
            --repository-name $ECR_REPOSITORY_NAME \
            --region $AWS_REGION \
            --image-scanning-configuration scanOnPush=true
        echo_info "ECR repository $ECR_REPOSITORY_NAME created"
    fi
}

# Build and push Docker image
build_and_push_image() {
    echo_info "Building Docker image..."
    
    # Build the image
    docker build -t $ECR_REPOSITORY_NAME:$IMAGE_TAG .
    
    # Tag for ECR
    docker tag $ECR_REPOSITORY_NAME:$IMAGE_TAG $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME:$IMAGE_TAG
    
    # Login to ECR
    echo_info "Logging in to ECR..."
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
    
    # Push image
    echo_info "Pushing image to ECR..."
    docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME:$IMAGE_TAG
    
    echo_info "Image pushed successfully"
}

# Create ECS task definition
create_task_definition() {
    echo_info "Creating ECS task definition..."
    
    cat > task-definition.json << EOF
{
    "family": "$ECS_TASK_DEFINITION",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "2048",
    "memory": "4096",
    "executionRoleArn": "arn:aws:iam::$ACCOUNT_ID:role/ecsTaskExecutionRole",
    "taskRoleArn": "arn:aws:iam::$ACCOUNT_ID:role/ecsTaskRole",
    "containerDefinitions": [
        {
            "name": "product-auth-container",
            "image": "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME:$IMAGE_TAG",
            "essential": true,
            "portMappings": [
                {
                    "containerPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {
                    "name": "PORT",
                    "value": "8000"
                },
                {
                    "name": "LOG_LEVEL",
                    "value": "INFO"
                }
            ],
            "secrets": [
                {
                    "name": "GROQ_API_KEY",
                    "valueFrom": "/product-auth/groq-api-key"
                },
                {
                    "name": "GAPIS_API_KEY",
                    "valueFrom": "/product-auth/google-api-key"
                },
                {
                    "name": "GCSE_ID",
                    "valueFrom": "/product-auth/google-cse-id"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/product-auth",
                    "awslogs-region": "$AWS_REGION",
                    "awslogs-stream-prefix": "ecs"
                }
            },
            "healthCheck": {
                "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                "interval": 30,
                "timeout": 5,
                "retries": 3,
                "startPeriod": 60
            }
        }
    ]
}
EOF

    # Register task definition
    aws ecs register-task-definition \
        --cli-input-json file://task-definition.json \
        --region $AWS_REGION
    
    echo_info "Task definition registered"
}

# Create CloudWatch log group
create_log_group() {
    echo_info "Creating CloudWatch log group..."
    
    if aws logs describe-log-groups --log-group-name-prefix "/ecs/product-auth" --region $AWS_REGION | grep -q "/ecs/product-auth"; then
        echo_info "Log group already exists"
    else
        aws logs create-log-group \
            --log-group-name "/ecs/product-auth" \
            --region $AWS_REGION
        echo_info "Log group created"
    fi
}

# Create ECS cluster
create_ecs_cluster() {
    echo_info "Creating ECS cluster..."
    
    if aws ecs describe-clusters --clusters $ECS_CLUSTER_NAME --region $AWS_REGION | grep -q "ACTIVE"; then
        echo_info "ECS cluster already exists"
    else
        aws ecs create-cluster \
            --cluster-name $ECS_CLUSTER_NAME \
            --capacity-providers FARGATE \
            --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1 \
            --region $AWS_REGION
        echo_info "ECS cluster created"
    fi
}

# Create or update ECS service
deploy_service() {
    echo_info "Deploying ECS service..."
    
    # Check if service exists
    if aws ecs describe-services --cluster $ECS_CLUSTER_NAME --services $ECS_SERVICE_NAME --region $AWS_REGION | grep -q "ACTIVE"; then
        echo_info "Updating existing service..."
        aws ecs update-service \
            --cluster $ECS_CLUSTER_NAME \
            --service $ECS_SERVICE_NAME \
            --task-definition $ECS_TASK_DEFINITION \
            --region $AWS_REGION
    else
        echo_info "Creating new service..."
        
        # You'll need to replace these subnet and security group IDs with your own
        aws ecs create-service \
            --cluster $ECS_CLUSTER_NAME \
            --service-name $ECS_SERVICE_NAME \
            --task-definition $ECS_TASK_DEFINITION \
            --desired-count 1 \
            --launch-type FARGATE \
            --network-configuration "awsvpcConfiguration={subnets=[subnet-12345,subnet-67890],securityGroups=[sg-12345],assignPublicIp=ENABLED}" \
            --region $AWS_REGION
    fi
    
    echo_info "Service deployment initiated"
}

# Store secrets in AWS Systems Manager Parameter Store
store_secrets() {
    echo_warn "Please store your API keys in AWS Systems Manager Parameter Store:"
    echo "aws ssm put-parameter --name '/product-auth/groq-api-key' --value 'YOUR_GROQ_API_KEY' --type 'SecureString'"
    echo "aws ssm put-parameter --name '/product-auth/google-api-key' --value 'YOUR_GOOGLE_API_KEY' --type 'SecureString'"
    echo "aws ssm put-parameter --name '/product-auth/google-cse-id' --value 'YOUR_GOOGLE_CSE_ID' --type 'SecureString'"
}

# Main deployment function
main() {
    echo_info "Starting deployment process..."
    
    check_aws_cli
    get_account_id
    create_ecr_repository
    create_log_group
    build_and_push_image
    create_task_definition
    create_ecs_cluster
    store_secrets
    
    echo_warn "Before deploying the service, please:"
    echo "1. Store your API keys in Parameter Store (commands shown above)"
    echo "2. Update the subnet and security group IDs in this script"
    echo "3. Ensure you have the necessary IAM roles (ecsTaskExecutionRole and ecsTaskRole)"
    
    read -p "Do you want to proceed with service deployment? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        deploy_service
        echo_info "Deployment completed!"
        echo_info "Your API will be available once the service is running."
        echo_info "Check the ECS console for service status and logs."
    else
        echo_info "Service deployment skipped. You can run this script again when ready."
    fi
    
    # Cleanup
    rm -f task-definition.json
}

# Run main function
main "$@" 