# Product Authentication & Review Analysis System

A comprehensive AI-powered system for product authenticity verification, review analysis, and seller trustworthiness assessment. This project combines advanced computer vision, natural language processing, and machine learning techniques to combat counterfeit products and fake reviews in e-commerce platforms.

## 🌟 Key Components

### 1. Enhanced Counterfeit Detector (`Deploy/enhanced_counterfeit_detector.py`)
### 2. FastAPI Backend (`Deploy/main.py`)
### 3. Seller Aura Calculator (`SellerPipeline/seller_aura_calculator.py`)

---

## 🔍 Enhanced Counterfeit Detector

The Enhanced Counterfeit Detector is a sophisticated AI system that analyzes product images to detect counterfeit items through multiple verification layers.

### 🎯 Core Features

#### **Brand Detection**
- **Image Captioning**: Uses Salesforce's BLIP model for automatic brand recognition from product images
- **Description Parsing**: Extracts brand information from product descriptions
- **Multi-brand Support**: Recognizes popular brands across categories (Nike, Adidas, Gucci, Rolex, Apple, etc.)

#### **Web-based Reference Search**
- **Google Custom Search Integration**: Automatically searches for authentic reference images
- **Smart Query Building**: Constructs optimized search queries using product descriptions
- **Image Validation**: Filters and validates reference images for relevance
- **Keyword Extraction**: Intelligently extracts colors, materials, and style keywords

#### **Advanced Logo Detection**
- **YOLO Integration**: Custom-trained YOLO model for precise logo detection
- **Feature Extraction**: Uses CLIP model for deep logo feature analysis
- **Quality Assessment**: Calculates logo quality and distortion scores
- **Multi-logo Support**: Handles products with multiple logos or brand elements

#### **Comprehensive Analysis Engine**
- **Similarity Comparison**: Compares detected logos with authentic reference images
- **Distortion Analysis**: Identifies logo deformations and quality issues
- **Confidence Scoring**: Provides detailed confidence metrics for each analysis
- **Issue Detection**: Automatically identifies specific authenticity concerns

#### **Visualization & Reporting**
- **Visual Analysis**: Generates detailed analysis visualizations with bounding boxes
- **Comprehensive Reports**: Creates JSON reports with all analysis details
- **Multiple Output Formats**: Supports both programmatic and visual outputs

### 🛠️ Technical Specifications

```python
# Key Models Used
- YOLO: Custom logo detection model
- CLIP: OpenAI's vision-language model for feature extraction
- BLIP: Salesforce's image captioning model for brand detection
- Google Custom Search API: For reference image retrieval
```

### 📊 Analysis Metrics

- **Logo Similarity Scores**: Cosine similarity between detected and reference logos
- **Distortion Scores**: Symmetry and edge consistency analysis
- **Quality Scores**: Sharpness, contrast, and brightness assessment
- **Confidence Levels**: Overall authenticity confidence (0-1 scale)

---

## 🚀 FastAPI Backend Service

The FastAPI backend provides a comprehensive REST API for product authentication, review analysis, and fake content detection.

### 🎯 Core Endpoints

#### **Counterfeit Detection APIs**
```http
POST /detect-counterfeit
POST /upload-image-counterfeit  
POST /detect-counterfeit-unified
```

**Features:**
- **Base64 Image Support**: Accept images as base64 encoded strings
- **File Upload Support**: Direct image file upload via multipart/form-data
- **Product Description Integration**: Enhanced analysis using product descriptions
- **Real-time Processing**: Fast analysis with detailed results

#### **Review Analysis APIs**
```http
POST /analyze-reviews
POST /analyze-single-review
POST /detect-fake-reviews
```

**Features:**
- **Aspect-based Sentiment Analysis**: Extract specific product aspects and sentiments
- **Fake Review Detection**: AI-powered fake review identification using BERT
- **Batch Processing**: Analyze multiple reviews simultaneously
- **Single Review Analysis**: Fast individual review processing

#### **Return Analysis APIs**
```http
POST /analyze-single-return
POST /analyze-returns-batch
```

**Features:**
- **Return Categorization**: Automatically categorize return reasons (Quality, Size, Shipping, etc.)
- **Sentiment Analysis**: Analyze sentiment of return reasons
- **Fake Return Detection**: Identify potentially fraudulent returns
- **Business Intelligence**: Generate actionable insights for return reduction

#### **Comprehensive Analysis**
```http
POST /batch-analysis
POST /batch-analysis-enhanced
```

**Features:**
- **Multi-modal Analysis**: Combines image and text analysis
- **Unified Processing**: Single endpoint for complete product assessment
- **Flexible Input**: Supports various input formats and combinations

### 🤖 AI Models Integration

#### **Counterfeit Detection**
- **Enhanced Counterfeit Detector**: Custom-built multi-stage analysis system
- **YOLO Logo Detection**: Pre-trained model for logo identification
- **CLIP Feature Extraction**: Advanced visual feature analysis

#### **Text Analysis**
- **BERT Fake Review Detection**: Fine-tuned model for fake content identification
- **PyABSA Aspect Extraction**: Advanced aspect-based sentiment analysis
- **Groq LLM Integration**: Large language model for generating business insights

### 📈 Response Features

#### **Detailed Analytics**
- **Confidence Scoring**: Probability scores for all predictions
- **Issue Identification**: Specific problems detected with products/reviews
- **Similarity Metrics**: Detailed comparison results
- **Risk Assessment**: Categorized risk levels

#### **Business Intelligence**
- **Actionable Suggestions**: AI-generated improvement recommendations
- **Trend Analysis**: Sentiment and authenticity distribution
- **Performance Metrics**: Comprehensive scoring across multiple dimensions

### 🔧 Technical Features

- **Async Processing**: Non-blocking request handling
- **Error Handling**: Comprehensive error management and logging
- **Data Validation**: Pydantic models for request/response validation
- **CORS Support**: Cross-origin resource sharing enabled
- **Health Monitoring**: Built-in health check endpoints

---

## 📊 Seller Aura Calculator

The Seller Aura Calculator is a sophisticated scoring system that evaluates seller trustworthiness based on their product portfolios and performance metrics.

### 🎯 Core Features

#### **Multi-dimensional Scoring**
- **Authenticity Score**: Based on counterfeit detection results
- **Quality Score**: Product quality assessment metrics
- **Similarity Score**: Logo and image quality consistency
- **Brand Confidence**: Brand detection accuracy
- **Review Sentiment**: Customer satisfaction analysis
- **Fake Review Penalty**: Reduction for detected fake reviews

#### **Advanced Analytics**
- **Consistency Bonus**: Rewards sellers with consistent quality across products
- **Risk Categorization**: Automatic classification (Low/Medium/High Risk)
- **Statistical Analysis**: Mean, median, and distribution calculations
- **Trend Monitoring**: Historical performance tracking

#### **DynamoDB Integration**
- **Single Table Design**: Efficient data storage and retrieval
- **GSI Optimization**: Global Secondary Index for fast queries
- **Batch Processing**: Handles large seller datasets
- **Real-time Updates**: Live score updates and calculations

### 📊 Scoring Algorithm

#### **Weighted Scoring System**
```python
Weights:
- Authenticity: 30%
- Similarity: 20%
- Quality: 20%
- Brand Confidence: 15%
- Review Sentiment: 10%
- Fake Review Penalty: 5%
```

#### **Risk Thresholds**
- **High Risk**: < 0.3 (30%)
- **Medium Risk**: 0.3 - 0.6 (30-60%)
- **Low Risk**: > 0.6 (60%+)

### 🔍 Analysis Capabilities

#### **Product-level Analysis**
- **Individual Scoring**: Each product gets comprehensive metrics
- **Quality Assessment**: Multi-factor quality evaluation
- **Authenticity Verification**: Counterfeit probability assessment
- **Historical Tracking**: Performance over time

#### **Seller-level Aggregation**
- **Portfolio Analysis**: Overall seller performance across all products
- **Consistency Measurement**: Standard deviation-based consistency scoring
- **Comparative Analysis**: Ranking against other sellers
- **Improvement Recommendations**: Actionable insights for score enhancement

### 📈 Reporting Features

#### **Comprehensive Reports**
- **Executive Summary**: High-level metrics and trends
- **Detailed Analytics**: Product-by-product breakdown
- **Risk Distribution**: Seller categorization statistics
- **Performance Rankings**: Top and bottom performers identification

#### **Data Export**
- **JSON Reports**: Machine-readable comprehensive data
- **Statistical Summaries**: Key performance indicators
- **Historical Tracking**: Trend analysis and change detection
- **Custom Filtering**: Risk category and score-based filtering

### 🛠️ Operational Features

#### **Database Operations**
- **Bulk Processing**: Efficient handling of large seller datasets
- **Incremental Updates**: Smart updating of changed records only
- **Data Integrity**: Comprehensive validation and error handling
- **Performance Optimization**: Efficient queries and batch operations

#### **Management Interface**
- **Interactive CLI**: Command-line interface for operations
- **Real-time Monitoring**: Live processing status and progress
- **Error Recovery**: Robust error handling and retry mechanisms
- **Flexible Configuration**: Customizable weights and thresholds

---

## 🏗️ System Architecture

### **Data Flow**
1. **Input Processing**: Images and text data ingestion
2. **AI Analysis**: Multi-model processing pipeline
3. **Score Calculation**: Weighted scoring algorithms
4. **Database Storage**: Efficient data persistence
5. **Report Generation**: Comprehensive analytics output

### **Integration Points**
- **AWS Services**: DynamoDB, S3, Lambda integration
- **External APIs**: Google Custom Search, Groq LLM
- **ML Models**: YOLO, CLIP, BERT, BLIP integration
- **Web Services**: RESTful API endpoints

### **Scalability Features**
- **Async Processing**: Non-blocking operations
- **Batch Processing**: Efficient bulk operations
- **Cloud-native**: AWS-optimized architecture
- **Microservices**: Modular component design

---

## 🚦 Getting Started

### **Prerequisites**
- Python 3.8+
- AWS Account with DynamoDB access
- Google Custom Search API credentials
- Groq API key (optional, for enhanced suggestions)

### **Environment Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env with your API keys and configuration
```

### **Quick Start**
```bash
# Start the FastAPI server
cd Deploy
python main.py

# Calculate seller aura scores
cd SellerPipeline
python seller_aura_calculator.py

# Run counterfeit detection
cd Deploy
python enhanced_counterfeit_detector.py
```

### **API Usage Examples**
```python
# Counterfeit Detection
curl -X POST "http://localhost:8000/detect-counterfeit-unified" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@product_image.jpg" \
  -F "product_description=Nike Air Jordan sneakers"

# Review Analysis
curl -X POST "http://localhost:8000/analyze-reviews" \
  -H "Content-Type: application/json" \
  -d '{"reviews": ["Great product!", "Poor quality"], "product_id": "123"}'
```

---

## 📄 License

This project is part of the Amazon HackOn hackathon submission focusing on combating counterfeit products and fake reviews in e-commerce platforms.

## 🤝 Contributing

This is a hackathon project. For questions or collaboration opportunities, please reach out to the development team.

---

## 🔮 Future Enhancements

- **Real-time Processing**: Live image analysis and scoring
- **Advanced ML Models**: Integration of latest computer vision models
- **Blockchain Integration**: Immutable authenticity records
- **Mobile App**: Consumer-facing authenticity verification app
- **Vendor Dashboard**: Seller portal for aura score monitoring 