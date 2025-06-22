#!/usr/bin/env python3
"""
Configuration settings for Seller Aura Calculator
"""

import os
from typing import Dict, Any

class Config:
    """Configuration class for Seller Aura Calculator"""
    
    # AWS Configuration
    AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    # DynamoDB Table Names
    SELLERS_TABLE = os.getenv('SELLERS_TABLE', 'Sellers')
    PRODUCTS_TABLE = os.getenv('PRODUCTS_TABLE', 'Products')
    PRODUCT_ANALYSIS_TABLE = os.getenv('PRODUCT_ANALYSIS_TABLE', 'ProductAnalysis')
    SELLER_SCORES_TABLE = os.getenv('SELLER_SCORES_TABLE', 'SellerScores')
    
    # Scoring Weights (must sum to 1.0 excluding penalty)
    SCORING_WEIGHTS = {
        'authenticity': float(os.getenv('WEIGHT_AUTHENTICITY', 0.30)),
        'similarity': float(os.getenv('WEIGHT_SIMILARITY', 0.20)),
        'quality': float(os.getenv('WEIGHT_QUALITY', 0.20)),
        'brand_confidence': float(os.getenv('WEIGHT_BRAND_CONFIDENCE', 0.15)),
        'review_sentiment': float(os.getenv('WEIGHT_REVIEW_SENTIMENT', 0.10)),
        'fake_review_penalty': float(os.getenv('WEIGHT_FAKE_REVIEW_PENALTY', 0.05))
    }
    
    # Risk Thresholds
    RISK_THRESHOLDS = {
        'high_risk': float(os.getenv('THRESHOLD_HIGH_RISK', 0.3)),
        'medium_risk': float(os.getenv('THRESHOLD_MEDIUM_RISK', 0.6)),
        'low_risk': float(os.getenv('THRESHOLD_LOW_RISK', 0.8))
    }
    
    # Scoring Parameters
    MAX_FAKE_REVIEW_PENALTY = float(os.getenv('MAX_FAKE_REVIEW_PENALTY', 0.3))
    MAX_CONSISTENCY_BONUS = float(os.getenv('MAX_CONSISTENCY_BONUS', 0.1))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Report Configuration
    REPORT_TOP_PERFORMERS_COUNT = int(os.getenv('REPORT_TOP_PERFORMERS', 10))
    REPORT_BOTTOM_PERFORMERS_COUNT = int(os.getenv('REPORT_BOTTOM_PERFORMERS', 10))
    
    # Processing Configuration
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 25))  # DynamoDB batch size
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
    RETRY_DELAY = float(os.getenv('RETRY_DELAY', 1.0))
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration settings"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate scoring weights
        total_weight = sum([
            cls.SCORING_WEIGHTS['authenticity'],
            cls.SCORING_WEIGHTS['similarity'],
            cls.SCORING_WEIGHTS['quality'],
            cls.SCORING_WEIGHTS['brand_confidence'],
            cls.SCORING_WEIGHTS['review_sentiment']
        ])
        
        if abs(total_weight - 1.0) > 0.01:
            validation_results['errors'].append(
                f"Scoring weights sum to {total_weight:.3f}, should sum to 1.0"
            )
            validation_results['valid'] = False
        
        # Validate risk thresholds
        if not (cls.RISK_THRESHOLDS['high_risk'] < cls.RISK_THRESHOLDS['medium_risk'] < cls.RISK_THRESHOLDS['low_risk']):
            validation_results['errors'].append(
                "Risk thresholds must be: high_risk < medium_risk < low_risk"
            )
            validation_results['valid'] = False
        
        # Check AWS credentials
        if not (cls.AWS_ACCESS_KEY_ID and cls.AWS_SECRET_ACCESS_KEY):
            validation_results['warnings'].append(
                "AWS credentials not found in environment variables. "
                "Make sure AWS CLI is configured or credentials are set."
            )
        
        # Validate ranges
        if not (0 <= cls.MAX_FAKE_REVIEW_PENALTY <= 1.0):
            validation_results['errors'].append(
                "MAX_FAKE_REVIEW_PENALTY must be between 0 and 1"
            )
            validation_results['valid'] = False
        
        if not (0 <= cls.MAX_CONSISTENCY_BONUS <= 1.0):
            validation_results['errors'].append(
                "MAX_CONSISTENCY_BONUS must be between 0 and 1"
            )
            validation_results['valid'] = False
        
        return validation_results
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("🔧 Seller Aura Calculator Configuration")
        print("=" * 50)
        print(f"AWS Region: {cls.AWS_REGION}")
        print(f"Tables: {cls.SELLERS_TABLE}, {cls.PRODUCTS_TABLE}, {cls.PRODUCT_ANALYSIS_TABLE}, {cls.SELLER_SCORES_TABLE}")
        print()
        print("Scoring Weights:")
        for weight, value in cls.SCORING_WEIGHTS.items():
            print(f"  {weight}: {value}")
        print()
        print("Risk Thresholds:")
        for threshold, value in cls.RISK_THRESHOLDS.items():
            print(f"  {threshold}: {value}")
        print()
        print(f"Max Fake Review Penalty: {cls.MAX_FAKE_REVIEW_PENALTY}")
        print(f"Max Consistency Bonus: {cls.MAX_CONSISTENCY_BONUS}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Log Level: {cls.LOG_LEVEL}")

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration"""
    SELLERS_TABLE = 'Dev_Sellers'
    PRODUCTS_TABLE = 'Dev_Products'
    PRODUCT_ANALYSIS_TABLE = 'Dev_ProductAnalysis'
    SELLER_SCORES_TABLE = 'Dev_SellerScores'
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production environment configuration"""
    SELLERS_TABLE = 'Prod_Sellers'
    PRODUCTS_TABLE = 'Prod_Products'
    PRODUCT_ANALYSIS_TABLE = 'Prod_ProductAnalysis'
    SELLER_SCORES_TABLE = 'Prod_SellerScores'
    LOG_LEVEL = 'WARNING'

class TestConfig(Config):
    """Test environment configuration"""
    SELLERS_TABLE = 'Test_Sellers'
    PRODUCTS_TABLE = 'Test_Products'
    PRODUCT_ANALYSIS_TABLE = 'Test_ProductAnalysis'
    SELLER_SCORES_TABLE = 'Test_SellerScores'
    LOG_LEVEL = 'DEBUG'
    BATCH_SIZE = 5  # Smaller batch for testing

# Configuration factory
def get_config(environment: str = None) -> Config:
    """Get configuration based on environment"""
    if environment is None:
        environment = os.getenv('ENVIRONMENT', 'development').lower()
    
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'test': TestConfig
    }
    
    return config_map.get(environment, Config)

# Sample .env file content for reference
SAMPLE_ENV_CONTENT = """
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1

# Environment
ENVIRONMENT=development

# Table Names (optional - will use defaults if not set)
SELLERS_TABLE=Sellers
PRODUCTS_TABLE=Products
PRODUCT_ANALYSIS_TABLE=ProductAnalysis
SELLER_SCORES_TABLE=SellerScores

# Scoring Weights (optional - will use defaults if not set)
WEIGHT_AUTHENTICITY=0.30
WEIGHT_SIMILARITY=0.20
WEIGHT_QUALITY=0.20
WEIGHT_BRAND_CONFIDENCE=0.15
WEIGHT_REVIEW_SENTIMENT=0.10
WEIGHT_FAKE_REVIEW_PENALTY=0.05

# Risk Thresholds (optional)
THRESHOLD_HIGH_RISK=0.3
THRESHOLD_MEDIUM_RISK=0.6
THRESHOLD_LOW_RISK=0.8

# Processing Settings (optional)
BATCH_SIZE=25
MAX_RETRIES=3
LOG_LEVEL=INFO
"""

if __name__ == "__main__":
    # Validate and print configuration
    config = get_config()
    config.print_config()
    
    print("\n🔍 Configuration Validation:")
    validation = config.validate_config()
    
    if validation['valid']:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has errors:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    if validation['warnings']:
        print("⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    print(f"\n📝 Sample .env file content:")
    print("-" * 30)
    print(SAMPLE_ENV_CONTENT) 