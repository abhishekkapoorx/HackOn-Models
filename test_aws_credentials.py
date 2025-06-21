#!/usr/bin/env python3
"""
AWS Credentials Test Script
Verifies that AWS credentials are properly configured
"""

import boto3
import os
from botocore.exceptions import NoCredentialsError, ClientError

def test_credentials():
    """Test AWS credentials and connection"""
    print("🔍 Testing AWS credentials...")
    print("=" * 50)
    
    # Check environment variables
    print("\n📋 Environment Variables:")
    env_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION', 'AWS_REGION']
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Mask the credentials for security
            if 'KEY' in var:
                masked_value = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else '*' * len(value)
                print(f"  ✅ {var}: {masked_value}")
            else:
                print(f"  ✅ {var}: {value}")
        else:
            print(f"  ❌ {var}: Not set")
    
    # Check AWS credentials file
    print(f"\n📁 AWS Credentials File:")
    credentials_path = os.path.expanduser('~/.aws/credentials')
    config_path = os.path.expanduser('~/.aws/config')
    
    if os.path.exists(credentials_path):
        print(f"  ✅ Credentials file exists: {credentials_path}")
    else:
        print(f"  ❌ Credentials file not found: {credentials_path}")
    
    if os.path.exists(config_path):
        print(f"  ✅ Config file exists: {config_path}")
    else:
        print(f"  ❌ Config file not found: {config_path}")
    
    # Test AWS connection
    print(f"\n🔗 Testing AWS Connection:")
    
    try:
        # Get default region
        region = os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION') or 'ap-south-1'
        print(f"  🌍 Using region: {region}")
        
        # Test STS (Security Token Service) - lightweight test
        sts_client = boto3.client('sts', region_name=region)
        response = sts_client.get_caller_identity()
        
        print(f"  ✅ Connection successful!")
        print(f"  🆔 Account ID: {response['Account']}")
        print(f"  👤 User ARN: {response['Arn']}")
        
        # Test Lambda service access
        lambda_client = boto3.client('lambda', region_name=region)
        try:
            lambda_client.list_functions(MaxItems=1)
            print(f"  ✅ Lambda service accessible")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'AccessDenied':
                print(f"  ⚠️ Lambda access denied - check IAM permissions")
            else:
                print(f"  ❌ Lambda service error: {error_code}")
        
        # Test S3 service access (optional)
        s3_client = boto3.client('s3', region_name=region)
        try:
            s3_client.list_buckets()
            print(f"  ✅ S3 service accessible")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'AccessDenied':
                print(f"  ⚠️ S3 access denied - may affect image storage")
            else:
                print(f"  ❌ S3 service error: {error_code}")
        
        print(f"\n🎉 AWS credentials are properly configured!")
        return True
        
    except NoCredentialsError:
        print(f"  ❌ No credentials found!")
        print(f"\n💡 Solutions:")
        print(f"  1. Run: aws configure")
        print(f"  2. Set environment variables")
        print(f"  3. Create ~/.aws/credentials file")
        print(f"  📖 See AWS_CREDENTIALS_SETUP.md for details")
        return False
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        print(f"  ❌ AWS API Error: {error_code}")
        print(f"  📋 Error details: {e.response['Error']['Message']}")
        
        if error_code == 'InvalidUserID.NotFound':
            print(f"  💡 Solution: Check your access key ID")
        elif error_code == 'SignatureDoesNotMatch':
            print(f"  💡 Solution: Check your secret access key")
        elif error_code == 'TokenRefreshRequired':
            print(f"  💡 Solution: Refresh your credentials")
        
        return False
        
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False

def main():
    """Main test function"""
    print("AWS Credentials Test")
    print("=" * 50)
    
    success = test_credentials()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Ready for Lambda deployment!")
        print("Run: python deploy_lambda.py")
    else:
        print("❌ Fix credentials before deployment")
        print("See: AWS_CREDENTIALS_SETUP.md")

if __name__ == "__main__":
    main() 