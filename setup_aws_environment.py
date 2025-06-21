#!/usr/bin/env python3
"""
Setup script to configure AWS environment for Lambda deployment
"""

import os
import subprocess
import sys

def check_aws_cli():
    """Check if AWS CLI is installed"""
    try:
        subprocess.run(['aws', '--version'], check=True, capture_output=True)
        print("✓ AWS CLI is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ AWS CLI is not installed")
        print("Please install AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html")
        return False

def setup_aws_credentials():
    """Setup AWS credentials"""
    print("\nSetting up AWS credentials...")
    
    # Check if credentials already exist
    try:
        subprocess.run(['aws', 'sts', 'get-caller-identity'], check=True, capture_output=True)
        print("✓ AWS credentials are already configured")
        return True
    except subprocess.CalledProcessError:
        print("AWS credentials not configured or invalid")
    
    print("\nYou can configure AWS credentials in several ways:")
    print("1. Run 'aws configure' command")
    print("2. Set environment variables:")
    print("   - AWS_ACCESS_KEY_ID")
    print("   - AWS_SECRET_ACCESS_KEY")
    print("   - AWS_DEFAULT_REGION")
    print("3. Use AWS IAM roles (if running on EC2)")
    
    choice = input("\nWould you like to run 'aws configure' now? (y/n): ").lower().strip()
    if choice == 'y':
        try:
            subprocess.run(['aws', 'configure'], check=True)
            return True
        except subprocess.CalledProcessError:
            print("Failed to configure AWS credentials")
            return False
    
    return False

def check_aws_region():
    """Check and set AWS region"""
    region = os.environ.get('AWS_DEFAULT_REGION') or os.environ.get('AWS_REGION')
    
    if region:
        print(f"✓ AWS region is set to: {region}")
        return region
    
    # Try to get from AWS config
    try:
        result = subprocess.run(
            ['aws', 'configure', 'get', 'region'], 
            check=True, 
            capture_output=True, 
            text=True
        )
        region = result.stdout.strip()
        if region:
            print(f"✓ AWS region from config: {region}")
            return region
    except subprocess.CalledProcessError:
        pass
    
    print("AWS region not configured")
    region = input("Enter AWS region (e.g., us-east-1, us-west-2, eu-west-1): ").strip()
    if region:
        # Set region in environment for this session
        os.environ['AWS_DEFAULT_REGION'] = region
        print(f"✓ AWS region set to: {region}")
        
        # Ask if user wants to save it permanently
        save = input("Save this region permanently in AWS config? (y/n): ").lower().strip()
        if save == 'y':
            try:
                subprocess.run(['aws', 'configure', 'set', 'region', region], check=True)
                print("✓ Region saved to AWS config")
            except subprocess.CalledProcessError:
                print("Failed to save region to AWS config")
        
        return region
    else:
        region = 'us-east-1'
        print(f"Using default region: {region}")
        os.environ['AWS_DEFAULT_REGION'] = region
        return region

def create_lambda_execution_role():
    """Create IAM role for Lambda execution"""
    print("\nCreating Lambda execution role...")
    
    role_name = 'review-analyzer-lambda-role'
    
    # Trust policy for Lambda
    trust_policy = {
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
    
    try:
        import boto3
        import json
        
        iam = boto3.client('iam')
        
        # Check if role already exists
        try:
            role = iam.get_role(RoleName=role_name)
            role_arn = role['Role']['Arn']
            print(f"✓ Role {role_name} already exists: {role_arn}")
            return role_arn
        except iam.exceptions.NoSuchEntityException:
            pass
        
        # Create the role
        role_response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='Execution role for review analyzer Lambda function'
        )
        
        role_arn = role_response['Role']['Arn']
        print(f"✓ Created role: {role_arn}")
        
        # Attach basic Lambda execution policy
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
        )
        
        print("✓ Attached basic execution policy")
        
        # Wait a moment for the role to be ready
        print("Waiting for role to be ready...")
        import time
        time.sleep(10)
        
        return role_arn
        
    except Exception as e:
        print(f"Failed to create IAM role: {str(e)}")
        print("You can create the role manually in AWS Console or using AWS CLI")
        return None

def main():
    """Main setup function"""
    print("AWS Environment Setup for Lambda Review Analyzer")
    print("=" * 50)
    
    # Check AWS CLI
    if not check_aws_cli():
        return False
    
    # Setup credentials
    if not setup_aws_credentials():
        print("Please configure AWS credentials before proceeding")
        return False
    
    # Check region
    region = check_aws_region()
    
    # Create Lambda execution role
    create_role = input("\nCreate Lambda execution role automatically? (y/n): ").lower().strip()
    role_arn = None
    
    if create_role == 'y':
        role_arn = create_lambda_execution_role()
    
    if not role_arn:
        print("\nYou'll need to provide the IAM role ARN when deploying the Lambda function")
        print("Example: arn:aws:iam::123456789012:role/lambda-execution-role")
    
    print("\n" + "=" * 50)
    print("Setup completed!")
    print(f"Region: {region}")
    if role_arn:
        print(f"Lambda Role ARN: {role_arn}")
        print(f"\nTo deploy the Lambda function, run:")
        print(f"python deploy_review_analyzer_lambda.py {role_arn}")
    else:
        print("\nTo deploy the Lambda function, run:")
        print("python deploy_review_analyzer_lambda.py <YOUR_ROLE_ARN>")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Setup failed: {str(e)}")
        sys.exit(1) 