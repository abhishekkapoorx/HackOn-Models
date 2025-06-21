#!/usr/bin/env python3
"""
Deployment script for AWS Lambda Review Analyzer function
"""

import boto3
import json
import zipfile
import os
import sys
from pathlib import Path

def get_aws_region():
    """Get AWS region from environment or user input"""
    # Try to get region from environment variables
    region = os.environ.get('AWS_DEFAULT_REGION') or os.environ.get('AWS_REGION')
    
    if not region:
        # Try to get from AWS config
        try:
            session = boto3.Session()
            region = session.region_name
        except:
            pass
    
    if not region:
        # Prompt user for region
        region = input("Enter AWS region (e.g., us-east-1, us-west-2, eu-west-1): ").strip()
        if not region:
            region = 'us-east-1'  # Default fallback
            print(f"Using default region: {region}")
    
    return region

def create_lambda_package():
    """Create deployment package for Lambda"""
    
    # Files to include in the package
    files_to_include = [
        'aws_lambda_review_analyzer.py',
        'lambda_requirements.txt'
    ]
    
    package_name = 'review_analyzer_lambda.zip'
    
    print(f"Creating Lambda deployment package: {package_name}")
    
    with zipfile.ZipFile(package_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add main Lambda function
        zipf.write('aws_lambda_review_analyzer.py', 'lambda_function.py')
        
        # Add requirements file
        zipf.write('lambda_requirements.txt', 'requirements.txt')
    
    print(f"Package created: {package_name}")
    return package_name

def deploy_lambda_function(function_name='review-analyzer', role_arn=None, region=None):
    """Deploy the Lambda function to AWS"""
    
    if not role_arn:
        print("Error: Please provide IAM role ARN for Lambda execution")
        print("Example: python deploy_review_analyzer_lambda.py arn:aws:iam::123456789012:role/lambda-execution-role")
        return
    
    if not region:
        region = get_aws_region()
    
    print(f"Using AWS region: {region}")
    
    # Create the package
    package_path = create_lambda_package()
    
    # Initialize Lambda client with region
    lambda_client = boto3.client('lambda', region_name=region)
    
    try:
        # Read the zip file
        with open(package_path, 'rb') as f:
            zip_content = f.read()
        
        # Check if function exists
        try:
            lambda_client.get_function(FunctionName=function_name)
            print(f"Function {function_name} exists, updating...")
            
            # Update function code
            response = lambda_client.update_function_code(
                FunctionName=function_name,
                ZipFile=zip_content
            )
            
            print(f"Function {function_name} updated successfully")
            
        except lambda_client.exceptions.ResourceNotFoundException:
            print(f"Creating new function: {function_name}")
            
            # Create new function
            response = lambda_client.create_function(
                FunctionName=function_name,
                Runtime='python3.9',
                Role=role_arn,
                Handler='lambda_function.lambda_handler',
                Code={'ZipFile': zip_content},
                Description='Review analyzer with fake detection and sentiment analysis',
                Timeout=300,  # 5 minutes
                MemorySize=3008,  # Maximum memory for model loading
                Environment={
                    'Variables': {
                        'PYTHONPATH': '/var/task:/opt/python'
                    }
                }
            )
            
            print(f"Function {function_name} created successfully")
        
        # Update function configuration for better performance
        lambda_client.update_function_configuration(
            FunctionName=function_name,
            Timeout=300,
            MemorySize=3008,
            Environment={
                'Variables': {
                    'PYTHONPATH': '/var/task:/opt/python',
                    'TRANSFORMERS_CACHE': '/tmp/transformers_cache'
                }
            }
        )
        
        print(f"Function ARN: {response['FunctionArn']}")
        
        # Create API Gateway trigger (optional)
        create_api_gateway_trigger = input("Create API Gateway trigger? (y/n): ").lower().strip()
        if create_api_gateway_trigger == 'y':
            setup_api_gateway(function_name, region)
        
    except Exception as e:
        print(f"Error deploying Lambda function: {str(e)}")
    
    finally:
        # Clean up
        if os.path.exists(package_path):
            os.remove(package_path)
            print(f"Cleaned up package file: {package_path}")

def setup_api_gateway(function_name, region):
    """Setup API Gateway trigger for the Lambda function"""
    
    try:
        # Initialize API Gateway client with region
        apigateway = boto3.client('apigateway', region_name=region)
        lambda_client = boto3.client('lambda', region_name=region)
        
        # Create REST API
        api_response = apigateway.create_rest_api(
            name=f'{function_name}-api',
            description=f'API for {function_name} Lambda function'
        )
        
        api_id = api_response['id']
        
        # Get root resource
        resources = apigateway.get_resources(restApiId=api_id)
        root_resource_id = next(r['id'] for r in resources['items'] if r['path'] == '/')
        
        # Create resource
        resource_response = apigateway.create_resource(
            restApiId=api_id,
            parentId=root_resource_id,
            pathPart='analyze'
        )
        
        resource_id = resource_response['id']
        
        # Create POST method
        apigateway.put_method(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod='POST',
            authorizationType='NONE'
        )
        
        # Get Lambda function ARN
        lambda_response = lambda_client.get_function(FunctionName=function_name)
        function_arn = lambda_response['Configuration']['FunctionArn']
        
        # Setup integration
        account_id = boto3.client('sts', region_name=region).get_caller_identity()['Account']
        
        integration_uri = f"arn:aws:apigateway:{region}:lambda:path/2015-03-31/functions/{function_arn}/invocations"
        
        apigateway.put_integration(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod='POST',
            type='AWS_PROXY',
            integrationHttpMethod='POST',
            uri=integration_uri
        )
        
        # Deploy API
        deployment = apigateway.create_deployment(
            restApiId=api_id,
            stageName='prod'
        )
        
        # Add Lambda permission for API Gateway
        lambda_client.add_permission(
            FunctionName=function_name,
            StatementId='api-gateway-invoke',
            Action='lambda:InvokeFunction',
            Principal='apigateway.amazonaws.com',
            SourceArn=f"arn:aws:execute-api:{region}:{account_id}:{api_id}/*/*"
        )
        
        api_url = f"https://{api_id}.execute-api.{region}.amazonaws.com/prod/analyze"
        print(f"API Gateway URL: {api_url}")
        
        # Test the API
        print("\nTo test the API, send a POST request with JSON body:")
        print('{"text": "This product is amazing! Best purchase ever!"}')
        
    except Exception as e:
        print(f"Error setting up API Gateway: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python deploy_review_analyzer_lambda.py <IAM_ROLE_ARN>")
        print("Example: python deploy_review_analyzer_lambda.py arn:aws:iam::123456789012:role/lambda-execution-role")
        sys.exit(1)
    
    role_arn = sys.argv[1]
    deploy_lambda_function(role_arn=role_arn) 