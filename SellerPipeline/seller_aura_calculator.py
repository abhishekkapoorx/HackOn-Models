#!/usr/bin/env python3
"""
Seller Aura Score Calculator

This script calculates seller aura scores based on their product aura scores.
Aura represents the authenticity and trustworthiness of sellers and their products.

Updated to work with the ProjectAura_Hackathon single-table design.
"""

import boto3
import json
import logging
import statistics
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProductAuraScore:
    """Data class for product aura score components"""
    product_id: str
    seller_id: str
    authenticity_score: float
    similarity_score: float
    quality_score: float
    brand_confidence: float
    review_sentiment_score: float
    fake_review_penalty: float
    overall_product_aura: float
    last_updated: str

@dataclass
class SellerAuraScore:
    """Data class for seller aura score"""
    seller_id: str
    seller_name: str
    product_count: int
    avg_authenticity_score: float
    avg_similarity_score: float
    avg_quality_score: float
    avg_brand_confidence: float
    avg_review_sentiment: float
    fake_review_penalty: float
    consistency_bonus: float
    overall_seller_aura: float
    risk_category: str
    last_calculated: str
    product_scores: List[ProductAuraScore]

class SellerAuraCalculator:
    """Main class for calculating seller aura scores"""
    
    def __init__(self, aws_region: str = 'ap-south-1'):
        """
        Initialize the calculator with DynamoDB connection
        
        Args:
            aws_region: AWS region for DynamoDB
        """
        self.aws_region = aws_region
        self.dynamodb = boto3.resource('dynamodb', region_name=aws_region)
        
        # Initialize table connection (single table design)
        self.table = self.dynamodb.Table('ProjectAura_Hackathon')
        
        # Scoring weights and thresholds
        self.weights = {
            'authenticity': 0.30,
            'similarity': 0.20,
            'quality': 0.20,
            'brand_confidence': 0.15,
            'review_sentiment': 0.10,
            'fake_review_penalty': 0.05
        }
        
        self.risk_thresholds = {
            'high_risk': 0.3,
            'medium_risk': 0.6,
            'low_risk': 0.8
        }
        
    def check_table_exists(self):
        """Check if the ProjectAura_Hackathon table exists"""
        try:
            self.table.load()
            logger.info("✓ ProjectAura_Hackathon table exists")
            return True
        except Exception as e:
            logger.error(f"Error accessing ProjectAura_Hackathon table: {str(e)}")
            logger.error("Please ensure the table exists and you have proper permissions")
            return False
    
    def get_all_sellers(self) -> List[Dict]:
        """Retrieve all sellers from DynamoDB using single table design"""
        try:
            response = self.table.scan(
                FilterExpression='begins_with(PK, :seller_prefix)',
                ExpressionAttributeValues={':seller_prefix': 'SELLER#'}
            )
            sellers = response['Items']
            
            # Handle pagination
            while 'LastEvaluatedKey' in response:
                response = self.table.scan(
                    FilterExpression='begins_with(PK, :seller_prefix)',
                    ExpressionAttributeValues={':seller_prefix': 'SELLER#'},
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                sellers.extend(response['Items'])
            
            logger.info(f"Retrieved {len(sellers)} sellers from DynamoDB")
            return sellers
            
        except Exception as e:
            logger.error(f"Error retrieving sellers: {str(e)}")
            return []
    
    def get_seller_products(self, seller_id: str) -> List[Dict]:
        """Get all products for a specific seller using GSI1"""
        try:
            response = self.table.query(
                IndexName='GSI1',
                KeyConditionExpression='GSI1PK = :seller_key',
                ExpressionAttributeValues={':seller_key': f'SELLER#{seller_id}'}
            )
            
            products = response['Items']
            
            # Handle pagination
            while 'LastEvaluatedKey' in response:
                response = self.table.query(
                    IndexName='GSI1',
                    KeyConditionExpression='GSI1PK = :seller_key',
                    ExpressionAttributeValues={':seller_key': f'SELLER#{seller_id}'},
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                products.extend(response['Items'])
            
            # Filter only products (not seller data)
            products = [p for p in products if p.get('EntityType') == 'PRODUCT']
            
            return products
            
        except Exception as e:
            logger.error(f"Error retrieving products for seller {seller_id}: {str(e)}")
            return []
    
    def calculate_product_aura_score(self, product: Dict) -> ProductAuraScore:
        """Calculate aura score for a single product using existing data structure"""
        # Extract basic product info
        product_id = product.get('ProductId', product.get('PK', ''))
        seller_id = product.get('SellerId', '')
        
        # Extract existing aura trust ledger data
        aura_trust_ledger = product.get('AuraTrustLedger', {})
        
        # Handle nested DynamoDB structure
        if isinstance(aura_trust_ledger, dict) and 'PillarScores' in aura_trust_ledger:
            pillar_scores = aura_trust_ledger['PillarScores']
            diagnostic_factors = aura_trust_ledger.get('DiagnosticFactors', {})
            
            # Handle DynamoDB's nested structure format
            if isinstance(pillar_scores, dict) and 'M' in pillar_scores:
                pillar_data = pillar_scores['M']
                authenticity_score = float(pillar_data.get('AuthenticityScore', {}).get('N', '50')) / 100.0
                quality_score = float(pillar_data.get('QualityScore', {}).get('N', '50')) / 100.0
                compliance_score = float(pillar_data.get('ComplianceScore', {}).get('N', '50')) / 100.0
                customer_satisfaction = float(pillar_data.get('CustomerSatisfactionScore', {}).get('N', '50')) / 100.0
            else:
                # Direct format - handle Decimal conversion
                def safe_float_convert(value, default=50):
                    if isinstance(value, Decimal):
                        return float(value)
                    return float(value if value is not None else default)
                
                authenticity_score = safe_float_convert(pillar_scores.get('AuthenticityScore', 50)) / 100.0
                quality_score = safe_float_convert(pillar_scores.get('QualityScore', 50)) / 100.0
                compliance_score = safe_float_convert(pillar_scores.get('ComplianceScore', 50)) / 100.0
                customer_satisfaction = safe_float_convert(pillar_scores.get('CustomerSatisfactionScore', 50)) / 100.0
            
            # Handle diagnostic factors
            if isinstance(diagnostic_factors, dict) and 'M' in diagnostic_factors:
                diag_data = diagnostic_factors['M']
                image_quality = float(diag_data.get('ImageQuality', {}).get('N', '50')) / 100.0
                seller_reputation = float(diag_data.get('SellerReputation', {}).get('N', '50')) / 100.0
                review_authenticity = float(diag_data.get('ReviewAuthenticity', {}).get('N', '50')) / 100.0
            else:
                # Direct format - handle Decimal conversion
                def safe_float_convert_diag(value, default=50):
                    if isinstance(value, Decimal):
                        return float(value)
                    return float(value if value is not None else default)
                
                image_quality = safe_float_convert_diag(diagnostic_factors.get('ImageQuality', 50)) / 100.0
                seller_reputation = safe_float_convert_diag(diagnostic_factors.get('SellerReputation', 50)) / 100.0
                review_authenticity = safe_float_convert_diag(diagnostic_factors.get('ReviewAuthenticity', 50)) / 100.0
        else:
            # Fallback to default scores
            authenticity_score = 0.5
            quality_score = 0.5
            compliance_score = 0.5
            customer_satisfaction = 0.5
            image_quality = 0.5
            seller_reputation = 0.5
            review_authenticity = 0.5
        
        # Use existing aura score if available
        existing_aura = product.get('CurrentAuraScore', product.get('AuraScore', 50))
        if isinstance(existing_aura, str):
            existing_aura = float(existing_aura)
        elif isinstance(existing_aura, Decimal):
            existing_aura = float(existing_aura)
        
        overall_product_aura = existing_aura / 100.0 if existing_aura > 1 else existing_aura
        
        # Calculate fake review penalty (simplified)
        fake_review_penalty = max(0, 1 - review_authenticity) * 0.1
        
        return ProductAuraScore(
            product_id=product_id,
            seller_id=seller_id,
            authenticity_score=authenticity_score,
            similarity_score=image_quality,  # Using image quality as proxy
            quality_score=quality_score,
            brand_confidence=compliance_score,
            review_sentiment_score=customer_satisfaction,
            fake_review_penalty=fake_review_penalty,
            overall_product_aura=overall_product_aura,
            last_updated=datetime.now().isoformat()
        )
    
    def calculate_seller_aura_score(self, seller: Dict, product_scores: List[ProductAuraScore]) -> SellerAuraScore:
        """Calculate overall aura score for a seller based on their product scores"""
        seller_id = seller.get('SellerId', seller.get('PK', '').replace('SELLER#', ''))
        seller_name = seller.get('SellerDisplayName', seller.get('BusinessName', f'Seller {seller_id}'))
        
        if not product_scores:
            # No products, assign neutral score
            return SellerAuraScore(
                seller_id=seller_id,
                seller_name=seller_name,
                product_count=0,
                avg_authenticity_score=0.5,
                avg_similarity_score=0.5,
                avg_quality_score=0.5,
                avg_brand_confidence=0.5,
                avg_review_sentiment=0.5,
                fake_review_penalty=0.0,
                consistency_bonus=0.0,
                overall_seller_aura=0.5,
                risk_category='medium_risk',
                last_calculated=datetime.now().isoformat(),
                product_scores=product_scores
            )
        
        # Calculate average scores
        avg_authenticity = statistics.mean([p.authenticity_score for p in product_scores])
        avg_similarity = statistics.mean([p.similarity_score for p in product_scores])
        avg_quality = statistics.mean([p.quality_score for p in product_scores])
        avg_brand_confidence = statistics.mean([p.brand_confidence for p in product_scores])
        avg_review_sentiment = statistics.mean([p.review_sentiment_score for p in product_scores])
        avg_fake_penalty = statistics.mean([p.fake_review_penalty for p in product_scores])
        
        # Calculate consistency bonus (reward sellers with consistently good scores)
        authenticity_scores = [p.authenticity_score for p in product_scores]
        quality_scores = [p.quality_score for p in product_scores]
        
        # Calculate standard deviation - lower deviation = more consistent = higher bonus
        auth_std = statistics.stdev(authenticity_scores) if len(authenticity_scores) > 1 else 0
        quality_std = statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0
        
        # Consistency bonus: 0 to 0.1 based on how consistent the scores are
        consistency_bonus = max(0, 0.1 - (auth_std + quality_std) / 2)
        
        # Calculate overall seller aura score
        base_score = (
            avg_authenticity * self.weights['authenticity'] +
            avg_similarity * self.weights['similarity'] +
            avg_quality * self.weights['quality'] +
            avg_brand_confidence * self.weights['brand_confidence'] +
            avg_review_sentiment * self.weights['review_sentiment'] -
            avg_fake_penalty * self.weights['fake_review_penalty']
        )
        
        overall_seller_aura = base_score + consistency_bonus
        overall_seller_aura = max(0.0, min(1.0, overall_seller_aura))
        
        # Determine risk category
        if overall_seller_aura >= self.risk_thresholds['low_risk']:
            risk_category = 'low_risk'
        elif overall_seller_aura >= self.risk_thresholds['medium_risk']:
            risk_category = 'medium_risk'
        else:
            risk_category = 'high_risk'
        
        return SellerAuraScore(
            seller_id=seller_id,
            seller_name=seller_name,
            product_count=len(product_scores),
            avg_authenticity_score=avg_authenticity,
            avg_similarity_score=avg_similarity,
            avg_quality_score=avg_quality,
            avg_brand_confidence=avg_brand_confidence,
            avg_review_sentiment=avg_review_sentiment,
            fake_review_penalty=avg_fake_penalty,
            consistency_bonus=consistency_bonus,
            overall_seller_aura=overall_seller_aura,
            risk_category=risk_category,
            last_calculated=datetime.now().isoformat(),
            product_scores=product_scores
        )
    
    def save_seller_score(self, seller_score: SellerAuraScore) -> bool:
        """Save seller aura score to DynamoDB"""
        try:
            # Convert dataclass to dict and handle Decimal conversion
            score_dict = asdict(seller_score)
            
            # Convert floats to Decimals for DynamoDB
            for key, value in score_dict.items():
                if isinstance(value, float):
                    score_dict[key] = Decimal(str(round(value, 6)))
                elif key == 'product_scores':
                    # Handle nested product scores
                    converted_products = []
                    for product in value:
                        converted_product = {}
                        for pk, pv in product.items():
                            if isinstance(pv, float):
                                converted_product[pk] = Decimal(str(round(pv, 6)))
                            else:
                                converted_product[pk] = pv
                        converted_products.append(converted_product)
                    score_dict[key] = converted_products
            
            # Add table keys for single table design
            seller_key = f"SELLER_AURA#{seller_score.seller_id}"
            score_dict['PK'] = seller_key
            score_dict['SK'] = seller_key
            score_dict['EntityType'] = 'SELLER_AURA'
            score_dict['GSI1PK'] = f"AURA_SCORES#{seller_score.risk_category}"
            score_dict['GSI1SK'] = f"SCORE#{int(seller_score.overall_seller_aura * 1000):04d}"
            
            self.table.put_item(Item=score_dict)
            return True
            
        except Exception as e:
            logger.error(f"Error saving seller score for {seller_score.seller_id}: {str(e)}")
            return False
    
    def update_seller_aura_directly(self, seller_score: SellerAuraScore) -> bool:
        """Update the seller's main record with the calculated aura score"""
        try:
            seller_key = f"SELLER#{seller_score.seller_id}"
            
            # Convert aura score to percentage for storage
            aura_percentage = int(seller_score.overall_seller_aura * 100)
            
            # Prepare update expression
            update_expression = "SET AuraScore = :aura_score, " \
                               "UpdatedAt = :updated_at, " \
                               "TrustMetrics = :trust_metrics"
            
            expression_attribute_values = {
                ':aura_score': Decimal(str(aura_percentage)),
                ':updated_at': datetime.now().isoformat(),
                ':trust_metrics': {
                    'OverallAuraScore': Decimal(str(round(seller_score.overall_seller_aura, 3))),
                    'AuthenticityScore': Decimal(str(round(seller_score.avg_authenticity_score, 3))),
                    'QualityScore': Decimal(str(round(seller_score.avg_quality_score, 3))),
                    'SimilarityScore': Decimal(str(round(seller_score.avg_similarity_score, 3))),
                    'BrandConfidence': Decimal(str(round(seller_score.avg_brand_confidence, 3))),
                    'ReviewSentiment': Decimal(str(round(seller_score.avg_review_sentiment, 3))),
                    'ConsistencyBonus': Decimal(str(round(seller_score.consistency_bonus, 3))),
                    'RiskCategory': seller_score.risk_category,
                    'ProductCount': Decimal(str(seller_score.product_count)),
                    'LastCalculated': seller_score.last_calculated
                }
            }
            
            # Update the seller record
            self.table.update_item(
                Key={'PK': seller_key, 'SK': seller_key},
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_attribute_values
            )
            
            logger.info(f"✓ Updated seller {seller_score.seller_id} with aura score {aura_percentage}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating seller record for {seller_score.seller_id}: {str(e)}")
            return False
    
    def calculate_all_seller_aura_scores(self) -> List[SellerAuraScore]:
        """Calculate aura scores for all sellers"""
        logger.info("Starting seller aura score calculation...")
        
        # Get all sellers
        sellers = self.get_all_sellers()
        if not sellers:
            logger.warning("No sellers found in database")
            return []
        
        all_seller_scores = []
        
        for seller in sellers:
            seller_id = seller.get('SellerId', seller.get('PK', '').replace('SELLER#', ''))
            logger.info(f"Processing seller: {seller_id}")
            
            # Get products for this seller
            products = self.get_seller_products(seller_id)
            logger.info(f"Found {len(products)} products for seller {seller_id}")
            
            # Calculate product aura scores
            product_scores = []
            for product in products:
                product_score = self.calculate_product_aura_score(product)
                product_scores.append(product_score)
            
            # Calculate seller aura score
            seller_score = self.calculate_seller_aura_score(seller, product_scores)
            all_seller_scores.append(seller_score)
            
            # Save to database
            if self.save_seller_score(seller_score):
                logger.info(f"✓ Saved aura score for seller {seller_id}: {seller_score.overall_seller_aura:.3f}")
            else:
                logger.error(f"✗ Failed to save aura score for seller {seller_id}")
        
        return all_seller_scores
    
    def calculate_and_update_all_sellers(self) -> List[SellerAuraScore]:
        """Calculate aura scores and update seller records directly"""
        logger.info("Starting seller aura score calculation with direct updates...")
        
        # Get all sellers
        sellers = self.get_all_sellers()
        if not sellers:
            logger.warning("No sellers found in database")
            return []
        
        all_seller_scores = []
        successful_updates = 0
        
        for seller in sellers:
            seller_id = seller.get('SellerId', seller.get('PK', '').replace('SELLER#', ''))
            logger.info(f"Processing seller: {seller_id}")
            
            # Get products for this seller
            products = self.get_seller_products(seller_id)
            logger.info(f"Found {len(products)} products for seller {seller_id}")
            
            # Calculate product aura scores
            product_scores = []
            for product in products:
                product_score = self.calculate_product_aura_score(product)
                product_scores.append(product_score)
            
            # Calculate seller aura score
            seller_score = self.calculate_seller_aura_score(seller, product_scores)
            all_seller_scores.append(seller_score)
            
            # Save separate aura record
            aura_saved = self.save_seller_score(seller_score)
            
            # Update seller record directly
            seller_updated = self.update_seller_aura_directly(seller_score)
            
            if aura_saved and seller_updated:
                successful_updates += 1
                logger.info(f"✓ Successfully updated seller {seller_id}: {seller_score.overall_seller_aura:.3f}")
            else:
                logger.error(f"✗ Failed to fully update seller {seller_id}")
        
        logger.info(f"✅ Successfully updated {successful_updates}/{len(sellers)} sellers")
        return all_seller_scores
    
    def generate_seller_aura_report(self, seller_scores: List[SellerAuraScore]) -> Dict:
        """Generate a comprehensive report of seller aura scores"""
        if not seller_scores:
            return {"error": "No seller scores available"}
        
        # Sort sellers by aura score (highest first)
        sorted_sellers = sorted(seller_scores, key=lambda x: x.overall_seller_aura, reverse=True)
        
        # Calculate statistics
        all_scores = [s.overall_seller_aura for s in seller_scores]
        avg_score = statistics.mean(all_scores)
        median_score = statistics.median(all_scores)
        
        # Risk category distribution
        risk_distribution = {}
        for seller in seller_scores:
            risk_cat = seller.risk_category
            risk_distribution[risk_cat] = risk_distribution.get(risk_cat, 0) + 1
        
        # Top and bottom performers
        top_performers = sorted_sellers[:10]
        bottom_performers = sorted_sellers[-10:] if len(sorted_sellers) >= 10 else []
        
        report = {
            "summary": {
                "total_sellers": len(seller_scores),
                "average_aura_score": round(avg_score, 3),
                "median_aura_score": round(median_score, 3),
                "highest_score": round(sorted_sellers[0].overall_seller_aura, 3),
                "lowest_score": round(sorted_sellers[-1].overall_seller_aura, 3),
                "report_generated": datetime.now().isoformat()
            },
            "risk_distribution": risk_distribution,
            "top_performers": [
                {
                    "seller_id": s.seller_id,
                    "seller_name": s.seller_name,
                    "aura_score": round(s.overall_seller_aura, 3),
                    "product_count": s.product_count,
                    "risk_category": s.risk_category
                }
                for s in top_performers
            ],
            "bottom_performers": [
                {
                    "seller_id": s.seller_id,
                    "seller_name": s.seller_name,
                    "aura_score": round(s.overall_seller_aura, 3),
                    "product_count": s.product_count,
                    "risk_category": s.risk_category
                }
                for s in bottom_performers
            ],
            "detailed_scores": [
                {
                    "seller_id": s.seller_id,
                    "seller_name": s.seller_name,
                    "overall_aura_score": round(s.overall_seller_aura, 3),
                    "product_count": s.product_count,
                    "avg_authenticity": round(s.avg_authenticity_score, 3),
                    "avg_quality": round(s.avg_quality_score, 3),
                    "avg_similarity": round(s.avg_similarity_score, 3),
                    "consistency_bonus": round(s.consistency_bonus, 3),
                    "risk_category": s.risk_category
                }
                for s in sorted_sellers
            ]
        }
        
        return report
    
    def get_existing_seller_scores(self) -> List[SellerAuraScore]:
        """Retrieve existing seller aura scores from the database"""
        try:
            response = self.table.scan(
                FilterExpression='begins_with(PK, :aura_prefix)',
                ExpressionAttributeValues={':aura_prefix': 'SELLER_AURA#'}
            )
            
            existing_scores = response['Items']
            
            # Handle pagination
            while 'LastEvaluatedKey' in response:
                response = self.table.scan(
                    FilterExpression='begins_with(PK, :aura_prefix)',
                    ExpressionAttributeValues={':aura_prefix': 'SELLER_AURA#'},
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                existing_scores.extend(response['Items'])
            
            # Convert back to SellerAuraScore objects (simplified)
            seller_scores = []
            for item in existing_scores:
                seller_score = SellerAuraScore(
                    seller_id=item['seller_id'],
                    seller_name=item['seller_name'],
                    product_count=int(item['product_count']),
                    avg_authenticity_score=float(item['avg_authenticity_score']),
                    avg_similarity_score=float(item['avg_similarity_score']),
                    avg_quality_score=float(item['avg_quality_score']),
                    avg_brand_confidence=float(item['avg_brand_confidence']),
                    avg_review_sentiment=float(item['avg_review_sentiment']),
                    fake_review_penalty=float(item['fake_review_penalty']),
                    consistency_bonus=float(item['consistency_bonus']),
                    overall_seller_aura=float(item['overall_seller_aura']),
                    risk_category=item['risk_category'],
                    last_calculated=item['last_calculated'],
                    product_scores=[]  # Simplified for report
                )
                seller_scores.append(seller_score)
            
            return seller_scores
            
        except Exception as e:
            logger.error(f"Error retrieving existing seller scores: {str(e)}")
            return []

def main():
    """Main function to run the seller aura calculator"""
    print("🌟 Seller Aura Score Calculator")
    print("=" * 50)
    
    # Initialize calculator
    try:
        calculator = SellerAuraCalculator()
        logger.info("✓ Calculator initialized")
    except Exception as e:
        logger.error(f"Failed to initialize calculator: {str(e)}")
        return
    
    # Menu options
    while True:
        print("\nOptions:")
        print("1. Check Table Connection")
        print("2. Calculate All Seller Aura Scores (Save Separately)")
        print("3. Calculate & Update Seller Records Directly")
        print("4. Generate Aura Report")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            if calculator.check_table_exists():
                print("✅ Table connection successful!")
            else:
                print("❌ Table connection failed!")
                
        elif choice == '2':
            print("\n🔄 Calculating seller aura scores (saving separately)...")
            start_time = time.time()
            seller_scores = calculator.calculate_all_seller_aura_scores()
            end_time = time.time()
            
            print(f"\n✅ Calculated aura scores for {len(seller_scores)} sellers")
            print(f"⏱️  Processing time: {end_time - start_time:.2f} seconds")
            print("📝 Scores saved as separate SELLER_AURA records")
            
        elif choice == '3':
            print("\n🔄 Calculating seller aura scores and updating seller records directly...")
            start_time = time.time()
            seller_scores = calculator.calculate_and_update_all_sellers()
            end_time = time.time()
            
            print(f"\n✅ Processed {len(seller_scores)} sellers")
            print(f"⏱️  Processing time: {end_time - start_time:.2f} seconds")
            print("🔄 Seller records updated with new aura scores and trust metrics")
            
        elif choice == '4':
            print("\n📊 Generating aura report...")
            # Get existing scores from database
            existing_scores = calculator.get_existing_seller_scores()
            
            if not existing_scores:
                print("❌ No seller scores found. Please calculate scores first.")
                continue
            
            report = calculator.generate_seller_aura_report(existing_scores)
            
            # Save report to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"seller_aura_report_{timestamp}.json"
            
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"📄 Report saved to: {report_filename}")
            print(f"📈 Total sellers analyzed: {report['summary']['total_sellers']}")
            print(f"📊 Average aura score: {report['summary']['average_aura_score']}")
            
        elif choice == '5':
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()