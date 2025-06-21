#!/usr/bin/env python3
"""
AWS Lambda Deployment Script for Counterfeit Detection System
"""

import os
import zipfile
import boto3
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from botocore.exceptions import NoCredentialsError, ClientError

class LambdaDeployer:
    def __init__(self):
        # Use region from environment variable or default
        region = os.getenv('AWS_REGION', 'ap-south-1')
        try:
            self.lambda_client = boto3.client('lambda', region_name=region)
            # Test credentials by trying to get caller identity
            sts_client = boto3.client('sts', region_name=region)
            sts_client.get_caller_identity()
            print("✅ AWS credentials verified")
        except NoCredentialsError:
            self._handle_credentials_error()
            raise
        except Exception as e:
            print(f"❌ AWS connection error: {e}")
            raise
        
        self.package_dir = 'lambda_package'
        self.zip_file = 'counterfeit_detection_lambda.zip'
    
    def _handle_credentials_error(self):
        """Handle AWS credentials error with helpful guidance"""
        print("\n❌ AWS Credentials Error!")
        print("=" * 50)
        print("No AWS credentials found. You need to configure them first.")
        print("\n📋 Quick Solutions:")
        print("1. Run: aws configure")
        print("2. Set environment variables:")
        print("   $env:AWS_ACCESS_KEY_ID='your-key'")
        print("   $env:AWS_SECRET_ACCESS_KEY='your-secret'") 
        print("   $env:AWS_DEFAULT_REGION='ap-south-1'")
        print("\n📖 For detailed instructions, see: AWS_CREDENTIALS_SETUP.md")
        print("=" * 50)
        
    def install_dependencies(self, use_minimal=True):
        """Install dependencies to package directory"""
        print("📦 Installing dependencies...")
        
        # Create package directory
        if os.path.exists(self.package_dir):
            shutil.rmtree(self.package_dir)
        os.makedirs(self.package_dir)
        
        # Choose requirements file (prioritize basic for stability)
        if use_minimal:
            if os.path.exists('requirements.txt'):
                requirements_file = 'requirements.txt'
            
        else:
            requirements_file = 'requirements.txt'
        
        if not os.path.exists(requirements_file):
            print(f"⚠️ {requirements_file} not found!")
            return
        
        print(f"📋 Using requirements file: {requirements_file}")
        
        try:
            # Install packages without dependencies first
            subprocess.run([
                'pip', 'install', '-r', requirements_file,
                '-t', self.package_dir, '--upgrade'
            ], check=True)
            
            print("✅ Dependencies installed")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("🔄 Trying alternative installation method...")
            
            # Try installing packages individually
            self._install_dependencies_individually(requirements_file)
    
    def _install_dependencies_individually(self, requirements_file):
        """Install dependencies one by one to handle version conflicts"""
        print("📦 Installing dependencies individually...")
        
        with open(requirements_file, 'r') as f:
            lines = f.readlines()
        
        packages = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                packages.append(line)
        
        installed_count = 0
        failed_packages = []
        
        for package in packages:
            try:
                print(f"  Installing: {package}")
                subprocess.run([
                    'pip', 'install', package,
                    '-t', self.package_dir, '--upgrade', '--no-deps'
                ], check=True, capture_output=True)
                installed_count += 1
                
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Failed to install {package}: {e}")
                failed_packages.append(package)
        
        print(f"✅ Installed {installed_count}/{len(packages)} packages")
        
        if failed_packages:
            print(f"⚠️ Failed packages: {failed_packages}")
            print("   Consider using Lambda layers for these heavy dependencies")
    
    def copy_source_files(self):
        """Copy source code to package directory"""
        print("📋 Copying source files...")
        
        source_files = [
            'aws_lambda_counterfeit_detector.py',
            'simple_image_ops.py',
            'enhanced_counterfeit_detector.py'
        ]
        
        for file in source_files:
            if os.path.exists(file):
                shutil.copy2(file, self.package_dir)
                print(f"  ✅ Copied {file}")
            else:
                print(f"  ❌ Warning: {file} not found")
        
        # Copy model directory if exists
        if os.path.exists('my_model'):
            shutil.copytree('my_model', os.path.join(self.package_dir, 'my_model'))
            print("  ✅ Copied my_model directory")
    
    def create_zip_package(self):
        """Create ZIP package for Lambda"""
        print("🗜️ Creating ZIP package...")
        
        if os.path.exists(self.zip_file):
            os.remove(self.zip_file)
        
        with zipfile.ZipFile(self.zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.package_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.package_dir)
                    zipf.write(file_path, arcname)
        
        # Check size
        size_mb = os.path.getsize(self.zip_file) / (1024 * 1024)
        print(f"✅ ZIP package created: {size_mb:.1f} MB")
        
        if size_mb > 250:
            print("⚠️ Warning: Package size exceeds 250MB limit for direct upload")
            print("   Consider using Lambda layers or S3 for deployment")
    
    def deploy_function(self, config_file='lambda_deployment_config.json'):
        """Deploy or update Lambda function"""
        print("🚀 Deploying Lambda function...")
        
        # Load configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        function_name = config['function_name']
        
        try:
            # Check if function exists
            self.lambda_client.get_function(FunctionName=function_name)
            
            # Update existing function
            print(f"📝 Updating existing function: {function_name}")
            
            with open(self.zip_file, 'rb') as f:
                zip_content = f.read()
            
            response = self.lambda_client.update_function_code(
                FunctionName=function_name,
                ZipFile=zip_content
            )
            
            # Update configuration
            self.lambda_client.update_function_configuration(
                FunctionName=function_name,
                Runtime=config['runtime'],
                Handler=config['handler'],
                Timeout=config['timeout'],
                MemorySize=config['memory_size'],
                Environment={'Variables': config['environment_variables']}
            )
            
            print(f"✅ Function updated successfully")
            
        except self.lambda_client.exceptions.ResourceNotFoundException:
            # Create new function
            print(f"🆕 Creating new function: {function_name}")
            
            with open(self.zip_file, 'rb') as f:
                zip_content = f.read()
            
                         # Get AWS account ID
            region = os.getenv('AWS_REGION', 'ap-south-1')
            sts_client = boto3.client('sts', region_name=region)
            account_id = sts_client.get_caller_identity()['Account']
            
            response = self.lambda_client.create_function(
                FunctionName=function_name,
                Runtime=config['runtime'],
                Role=f"arn:aws:iam::{account_id}:role/lambda-execution-role",
                Handler=config['handler'],
                Code={'ZipFile': zip_content},
                Description=config['description'],
                Timeout=config['timeout'],
                MemorySize=config['memory_size'],
                Environment={'Variables': config['environment_variables']},
                Tags=config['tags']
            )
        
            print(f"✅ Function created successfully")
        
        return response
    
    def create_api_gateway(self, function_name):
        """Create API Gateway for the Lambda function"""
        print("🌐 Creating API Gateway...")
        
        # This would require additional API Gateway setup
        # For now, just print instructions
        print("📋 To create API Gateway:")
        print("1. Go to AWS API Gateway console")
        print("2. Create new REST API")
        print("3. Create resource/method")
        print(f"4. Integrate with Lambda function: {function_name}")
        print("5. Deploy to stage")
    
    def cleanup(self):
        """Clean up temporary files"""
        print("🧹 Cleaning up...")
        
        if os.path.exists(self.package_dir):
            shutil.rmtree(self.package_dir)
        
        if os.path.exists(self.zip_file):
            os.remove(self.zip_file)
        
        print("✅ Cleanup complete")
    
    def deploy_complete_system(self):
        """Deploy the complete system"""
        try:
            self.install_dependencies()
            self.copy_source_files()
            self.create_zip_package()
            
            response = self.deploy_function()
            function_name = response['FunctionName']
            
            print(f"\n🎉 Deployment successful!")
            print(f"Function Name: {function_name}")
            print(f"Function ARN: {response['FunctionArn']}")
            
            self.create_api_gateway(function_name)
            
        except NoCredentialsError:
            print("❌ Deployment failed: AWS credentials not configured")
            print("🔧 Run the AWS credentials setup first!")
            return False
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'AccessDenied':
                print("❌ Deployment failed: Access denied")
                print("🔧 Check your IAM permissions - see aws_iam_policy.json")
            else:
                print(f"❌ AWS API error: {e}")
            return False
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            return False
        finally:
            self.cleanup()
        
        return True

def main():
    """Main deployment function"""
    deployer = LambdaDeployer()
    
    print("🚀 Starting Lambda deployment process...")
    deployer.deploy_complete_system()

if __name__ == "__main__":
    main() 