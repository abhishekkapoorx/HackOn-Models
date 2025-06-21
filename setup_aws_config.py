#!/usr/bin/env python3
"""
AWS Configuration Setup Helper
Sets up AWS credentials and region configuration for the counterfeit detection system
"""

import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def check_aws_credentials():
    """Check if AWS credentials are properly configured"""
    print("🔐 Checking AWS credentials...")
    
    try:
        # Try to get caller identity
        region = os.getenv('AWS_REGION', 'ap-south-1')
        sts_client = boto3.client('sts', region_name=region)
        identity = sts_client.get_caller_identity()
        
        print(f"✅ AWS credentials are valid")
        print(f"   Account ID: {identity['Account']}")
        print(f"   User ARN: {identity['Arn']}")
        print(f"   Region: {region}")
        
        return True
        
    except NoCredentialsError:
        print("❌ AWS credentials not found!")
        print("   Please configure your AWS credentials using one of these methods:")
        print("   1. AWS CLI: aws configure")
        print("   2. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
        print("   3. IAM roles (if running on EC2)")
        return False
        
    except ClientError as e:
        print(f"❌ AWS credentials error: {e}")
        return False

def check_s3_access():
    """Check S3 access and bucket configuration"""
    print("\n📦 Checking S3 access...")
    
    try:
        region = os.getenv('AWS_REGION', 'ap-south-1')
        bucket_name = os.getenv('S3_BUCKET_NAME', 'images-hackonn-fashion')
        
        s3_client = boto3.client('s3', region_name=region)
        
        # Try to list buckets
        response = s3_client.list_buckets()
        print(f"✅ S3 access confirmed")
        print(f"   Available buckets: {len(response['Buckets'])}")
        
        # Check if our bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"✅ Target bucket '{bucket_name}' exists")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f"⚠️ Target bucket '{bucket_name}' does not exist")
                print(f"   It will be created automatically during deployment")
            else:
                print(f"❌ Error accessing bucket '{bucket_name}': {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ S3 access error: {e}")
        return False

def check_pinecone_config():
    """Check Pinecone configuration"""
    print("\n🌲 Checking Pinecone configuration...")
    
    api_key = os.getenv('PINECONE_API_KEY')
    
    if not api_key:
        print("❌ PINECONE_API_KEY not found in environment variables")
        return False
    
    try:
        import pinecone
        pc = pinecone.Pinecone(api_key=api_key)
        indexes = pc.list_indexes()
        print(f"✅ Pinecone connection successful")
        print(f"   Available indexes: {len(indexes)}")
        return True
        
    except Exception as e:
        print(f"❌ Pinecone connection error: {e}")
        return False

def check_google_apis():
    """Check Google APIs configuration"""
    print("\n🔍 Checking Google APIs configuration...")
    
    api_key = os.getenv('GAPIS_API_KEY')
    search_engine_id = os.getenv('GCSE_ID')
    
    if not api_key:
        print("❌ GAPIS_API_KEY not found in environment variables")
        return False
    
    if not search_engine_id:
        print("❌ GCSE_ID not found in environment variables")
        return False
    
    print(f"✅ Google APIs configuration found")
    print(f"   API Key: {'*' * (len(api_key) - 8) + api_key[-8:]}")
    print(f"   Search Engine ID: {search_engine_id}")
    
    return True

def setup_environment_file():
    """Create a template .env file if it doesn't exist"""
    env_file = '.env'
    
    if os.path.exists(env_file):
        print(f"\n📄 Environment file '{env_file}' already exists")
        return
    
    print(f"\n📄 Creating template environment file: {env_file}")
    
    template = """# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_REGION=ap-south-1
S3_BUCKET_NAME=images-hackonn-fashion

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here

# Google Custom Search Configuration
GAPIS_API_KEY=your_google_apis_key_here
GCSE_ID=your_google_custom_search_engine_id_here
"""
    
    with open(env_file, 'w') as f:
        f.write(template)
    
    print(f"✅ Template .env file created")
    print(f"   Please edit {env_file} and add your actual API keys and credentials")

def main():
    """Main setup function"""
    print("🚀 AWS Lambda Counterfeit Detection - Configuration Setup")
    print("=" * 60)
    
    # Load environment variables from .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded environment variables from .env file")
    except ImportError:
        print("⚠️ python-dotenv not installed, using system environment variables")
    except Exception as e:
        print(f"⚠️ Could not load .env file: {e}")
    
    print()
    
    # Check all configurations
    checks = []
    checks.append(("AWS Credentials", check_aws_credentials()))
    checks.append(("S3 Access", check_s3_access()))
    checks.append(("Pinecone Config", check_pinecone_config()))
    checks.append(("Google APIs Config", check_google_apis()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Configuration Summary:")
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {check_name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print("🎉 All configuration checks passed!")
        print("   You're ready to deploy the Lambda function")
        print("   Run: python deploy_lambda.py")
    else:
        print("❌ Some configuration checks failed")
        print("   Please fix the issues above before deploying")
        setup_environment_file()
    
    print("\n📚 For detailed setup instructions, see README_LAMBDA.md")

if __name__ == "__main__":
    main() 