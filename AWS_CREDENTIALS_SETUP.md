# AWS Credentials Setup Guide

## The Problem
You're getting `NoCredentialsError: Unable to locate credentials` because AWS credentials are not configured on your system.

## Solution Options

### Option 1: AWS CLI Configuration (Recommended)

1. **Install AWS CLI** (if not already installed):
   ```bash
   pip install awscli
   ```

2. **Configure AWS credentials**:
   ```bash
   aws configure
   ```
   
   You'll be prompted to enter:
   - AWS Access Key ID
   - AWS Secret Access Key  
   - Default region name (e.g., `us-east-1`, `ap-south-1`)
   - Default output format (just press Enter for default)

3. **Get your AWS credentials**:
   - Go to [AWS Console](https://console.aws.amazon.com/)
   - Navigate to IAM → Users → Your User → Security credentials
   - Click "Create access key"
   - Choose "Command Line Interface (CLI)"
   - Download and save the credentials

### Option 2: Environment Variables

Set these environment variables in your system:

**Windows (PowerShell):**
```powershell
$env:AWS_ACCESS_KEY_ID="your-access-key-id"
$env:AWS_SECRET_ACCESS_KEY="your-secret-access-key"
$env:AWS_DEFAULT_REGION="ap-south-1"
```

**Windows (Command Prompt):**
```cmd
set AWS_ACCESS_KEY_ID=your-access-key-id
set AWS_SECRET_ACCESS_KEY=your-secret-access-key
set AWS_DEFAULT_REGION=ap-south-1
```

### Option 3: Credentials File

Create a credentials file at:
- Windows: `C:\Users\YourUsername\.aws\credentials`
- Linux/Mac: `~/.aws/credentials`

Content:
```ini
[default]
aws_access_key_id = your-access-key-id
aws_secret_access_key = your-secret-access-key
```

And a config file at:
- Windows: `C:\Users\YourUsername\.aws\config`
- Linux/Mac: `~/.aws/config`

Content:
```ini
[default]
region = ap-south-1
output = json
```

## Required IAM Permissions

Your AWS user must have these permissions:
- `lambda:*` (Full Lambda access)
- `iam:PassRole` (To assign execution role)
- `iam:CreateRole` (To create execution role if needed)
- `iam:AttachRolePolicy`
- `s3:*` (For S3 operations if using image storage)

## Verification

Test your credentials:
```bash
aws sts get-caller-identity
```

This should return your AWS account details if credentials are working.

## Quick Setup Script

Run this after configuring credentials:
```bash
python setup_aws_config.py
```

## Troubleshooting

1. **Invalid credentials**: Re-check your access key and secret key
2. **Region issues**: Make sure the region is set correctly
3. **Permissions**: Ensure your AWS user has the required permissions
4. **MFA**: If you have MFA enabled, you might need temporary credentials

## Security Best Practices

1. **Never commit credentials to git**
2. **Use least privilege principle** - only grant necessary permissions
3. **Rotate access keys regularly**
4. **Consider using AWS IAM roles** for production deployments
5. **Enable CloudTrail** for credential usage monitoring

After setting up credentials, run the deployment again:
```bash
python deploy_lambda.py
``` 