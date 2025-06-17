# Image Vector Search with Pinecone and CLIP

This project implements an image vector search system using CLIP (Contrastive Language-Image Pre-training) embeddings and Pinecone vector database. It allows you to:

1. **Upload images** from a folder to Pinecone with CLIP embeddings
2. **Search images by text** using natural language queries
3. **Search images by image** to find similar images
4. **Interactive search** with a command-line interface

## Features

- 🖼️ **CLIP-based embeddings**: Uses OpenAI's CLIP model for high-quality image and text embeddings
- 🔍 **Dual search modes**: Text-to-image and image-to-image search
- 📊 **Visual results**: Displays search results with images and similarity scores
- 🚀 **Pinecone integration**: Scalable vector database for fast similarity search
- 🎯 **Interactive demo**: Command-line interface for easy testing

## Prerequisites

- Python 3.8+
- Pinecone API key
- CUDA-compatible GPU (optional, but recommended for faster processing)

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root with your Pinecone API key:
   ```
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

4. **Get your Pinecone API key**:
   - Sign up at [pinecone.io](https://pinecone.io)
   - Create a new index or use an existing one
   - Copy your API key from the dashboard

## Usage

### Quick Start

Run the complete demo:
```bash
python demo_image_search.py
```

This will:
1. Upload all images from the `public/` folder to Pinecone
2. Perform example text and image searches
3. Start an interactive search mode

### Individual Scripts

#### 1. Upload Images to Pinecone

```bash
python upload_images_pinecone.py
```

This script:
- Scans the `public/` folder for images (jpg, jpeg, png, bmp, tiff, webp)
- Generates CLIP embeddings for each image
- Uploads embeddings to Pinecone with metadata

#### 2. Search Images

```bash
python image_vector_search.py
```

This script demonstrates:
- Text-based image search
- Image-based similarity search
- Visual display of results

### Programmatic Usage

```python
from upload_images_pinecone import ImageVectorUploader
from image_vector_search import ImageVectorSearch

# Upload images
uploader = ImageVectorUploader()
uploader.upload_images_from_folder("public")

# Search images
searcher = ImageVectorSearch()

# Text search
results = searcher.search_by_text("a person", top_k=5)

# Image search
results = searcher.search_by_image("path/to/query/image.jpg", top_k=5)

# Display results
searcher.display_search_results(results)
```

## File Structure

```
Models/
├── public/                    # Images to be indexed
│   ├── image1.jpg
│   └── image2.jpg
├── lib/
│   └── VectorStore.py        # Legacy vector store class
├── upload_images_pinecone.py # Image upload script
├── image_vector_search.py    # Search functionality
├── demo_image_search.py      # Complete demo
├── create_image_vectordb.ipynb # Jupyter notebook
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## How It Works

### 1. Image Processing
- Images are loaded and converted to RGB format
- CLIP model processes images to generate 512-dimensional embeddings
- Embeddings capture semantic meaning of image content

### 2. Vector Storage
- Embeddings are stored in Pinecone vector database
- Each image gets a unique ID and metadata (filename, filepath)
- Pinecone enables fast similarity search using cosine distance

### 3. Search Process
- **Text search**: Query text is converted to CLIP embedding, then compared to image embeddings
- **Image search**: Query image is converted to embedding, then compared to stored image embeddings
- Results are ranked by similarity score (higher = more similar)

## Configuration

### Pinecone Index Settings
- **Dimension**: 512 (CLIP embedding size)
- **Metric**: Cosine similarity
- **Cloud**: AWS (us-east-1)
- **Index Name**: "image-vectors" (configurable)

### Supported Image Formats
- JPG/JPEG
- PNG
- BMP
- TIFF
- WebP

## Performance Tips

1. **GPU Usage**: The system automatically uses CUDA if available for faster processing
2. **Batch Processing**: Images are uploaded in batches of 100 for efficiency
3. **Memory Management**: CLIP model is loaded once and reused for all operations

## Troubleshooting

### Common Issues

1. **"PINECONE_API_KEY not found"**
   - Ensure you have a `.env` file with your API key
   - Check that the key is valid and active

2. **"No images found"**
   - Verify images exist in the `public/` folder
   - Check that images are in supported formats

3. **CUDA out of memory**
   - Reduce batch size in upload script
   - Use CPU instead of GPU (slower but uses less memory)

4. **Pinecone connection errors**
   - Verify your API key and internet connection
   - Check Pinecone service status

### Debug Mode

Add debug prints by modifying the scripts:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Usage

### Custom Embedding Models

You can modify the model by changing the `model_ID` in the classes:
```python
self.model_ID = "openai/clip-vit-large-patch14"  # Larger model
```

### Custom Search Filters

Add metadata filters to your searches:
```python
results = self.index.query(
    vector=embedding.tolist(),
    top_k=5,
    include_metadata=True,
    filter={"type": "image"}  # Custom filter
)
```

### Batch Processing

For large image collections, consider processing in smaller batches:
```python
uploader.upload_images_from_folder("public", batch_size=50)
```

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project!

## License

This project is open source and available under the MIT License. 