# Seller Pipeline - Aura Score Calculator 🌟

This directory contains the Seller Aura Score Calculator, which computes authenticity and trustworthiness scores for sellers based on their product performance metrics.

## Overview

The **Seller Aura Score** represents the overall authenticity and trustworthiness of a seller based on:

- **Product Authenticity Scores** (30%) - From counterfeit detection analysis
- **Product Quality Scores** (20%) - Based on logo distortion and image quality
- **Product Similarity Scores** (20%) - How well products match reference images
- **Brand Confidence Scores** (15%) - Brand detection confidence
- **Review Sentiment Scores** (10%) - Customer review sentiment analysis
- **Fake Review Penalty** (5%) - Penalty for detected fake reviews

## Features

- ✅ **DynamoDB Integration** - Stores and retrieves seller, product, and analysis data
- ✅ **Automated Score Calculation** - Processes all sellers and their products
- ✅ **Risk Categorization** - Classifies sellers as low, medium, or high risk
- ✅ **Consistency Bonus** - Rewards sellers with consistent product quality
- ✅ **Comprehensive Reporting** - Generates detailed analytics reports
- ✅ **Sample Data** - Includes test data for development and testing

## Installation

1. **Install Dependencies**:
```bash
pip install boto3 python-dotenv
```

2. **Configure AWS Credentials**:
```bash
aws configure
# OR set environment variables:
# export AWS_ACCESS_KEY_ID=your_access_key
# export AWS_SECRET_ACCESS_KEY=your_secret_key
# export AWS_DEFAULT_REGION=us-east-1
```

## Usage

### Quick Start

```bash
cd SellerPipeline
python seller_aura_calculator.py
```

The script provides an interactive menu:

1. **Create/Check Tables** - Sets up required DynamoDB tables
2. **Populate Sample Data** - Adds test data for development
3. **Calculate All Seller Aura Scores** - Processes all sellers
4. **Generate Aura Report** - Creates detailed analytics report
5. **Exit** - Quit the application

### Programmatic Usage

```python
from seller_aura_calculator import SellerAuraCalculator

# Initialize calculator
calculator = SellerAuraCalculator(aws_region='us-east-1')

# Create tables if needed
calculator.create_tables_if_not_exist()

# Calculate scores for all sellers
seller_scores = calculator.calculate_all_seller_aura_scores()

# Generate report
report = calculator.generate_seller_aura_report(seller_scores)
```

## Scoring Algorithm

### Product Aura Score
```
Product Aura = (Authenticity × 0.30) + 
               (Similarity × 0.20) + 
               (Quality × 0.20) + 
               (Brand Confidence × 0.15) + 
               (Review Sentiment × 0.10) - 
               (Fake Review Penalty × 0.05)
```

### Seller Aura Score
```
Seller Aura = Average(Product Aura Scores) + Consistency Bonus

Where Consistency Bonus = max(0, 0.1 - (std_dev_scores / 2))
```

### Risk Categories
- **Low Risk**: Aura Score ≥ 0.8
- **Medium Risk**: 0.6 ≤ Aura Score < 0.8  
- **High Risk**: Aura Score < 0.6

## Output Reports

The generated reports include:

### Summary Statistics
- Total sellers analyzed
- Average and median aura scores
- Score distribution
- Risk category breakdown

### Top/Bottom Performers
- Best performing sellers
- Worst performing sellers
- Product count and risk categories

### Detailed Analytics
- Complete seller scores breakdown
- Individual metric performance
- Consistency bonuses
- Trend analysis

### Sample Report Output
```json
{
  "summary": {
    "total_sellers": 5,
    "average_aura_score": 0.742,
    "median_aura_score": 0.735,
    "highest_score": 0.891,
    "lowest_score": 0.623
  },
  "risk_distribution": {
    "low_risk": 2,
    "medium_risk": 2,
    "high_risk": 1
  },
  "top_performers": [...],
  "detailed_scores": [...]
}
```

## Configuration

### Scoring Weights
Modify weights in the `SellerAuraCalculator` class:

```python
self.weights = {
    'authenticity': 0.30,      # Product authenticity importance
    'similarity': 0.20,        # Reference similarity importance
    'quality': 0.20,           # Product quality importance
    'brand_confidence': 0.15,  # Brand detection confidence
    'review_sentiment': 0.10,  # Review sentiment importance
    'fake_review_penalty': 0.05 # Fake review penalty weight
}
```

### Risk Thresholds
Adjust risk categorization thresholds:

```python
self.risk_thresholds = {
    'high_risk': 0.3,    # Below this = high risk
    'medium_risk': 0.6,  # Below this = medium risk
    'low_risk': 0.8      # Above this = low risk
}
```

## Integration with Main System

This script integrates with the existing counterfeit detection system by:

1. **Reading Product Analysis Data** - Uses results from `enhanced_counterfeit_detector.py`
2. **Processing Review Analytics** - Incorporates sentiment and fake review detection
3. **Storing Results** - Saves calculated scores to DynamoDB for API access
4. **Generating Reports** - Creates analytics for business intelligence

## Error Handling

The script includes robust error handling for:

- AWS credential issues
- DynamoDB table access problems
- Missing or malformed data
- Network connectivity issues
- Data type conversion errors

## Monitoring and Logging

- **Comprehensive Logging** - All operations are logged with timestamps
- **Progress Tracking** - Real-time progress updates during processing
- **Error Reporting** - Detailed error messages and stack traces
- **Performance Metrics** - Processing time and throughput statistics

## Contributing

To extend the functionality:

1. **Add New Metrics** - Modify the scoring algorithm in `calculate_product_aura_score()`
2. **Custom Weights** - Adjust scoring weights based on business requirements
3. **Additional Reports** - Extend `generate_seller_aura_report()` with new analytics
4. **Data Sources** - Integrate additional data sources for scoring

## License

This project is part of the Amazon HackOn Fashion Authentication system. 