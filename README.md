# Simple Image Operations with Pinecone and CLIP

A lightweight Python library for uploading images to Pinecone with CLIP embeddings and performing similarity search.

## Features

- 🖼️ **Single Image Upload**: Upload images with CLIP embeddings, UUID, and optional text descriptions
- 🔍 **Similarity Search**: Find similar images using vector search with top-k results
- 🆔 **UUID Generation**: Automatic unique ID generation for each uploaded image
- 📝 **Text Metadata**: Store optional text descriptions with images
- ⚡ **Fast Search**: Pinecone-powered vector similarity search

## Quick Start

### Installation

1. **Install dependencies**:
   ```bash
   pip install torch transformers pinecone-client pillow python-dotenv
   ```

2. **Set up environment variables**:
   Create a `.env` file with your Pinecone API key:
   ```
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

### Usage

```python
from simple_image_ops import SimpleImageOps

# Initialize
ops = SimpleImageOps()

# 1. Upload an image
image_id = ops.upload_image(
    image_path="path/to/image.jpg",
    text_description="A person in a professional setting"
)

# 2. Search for similar images
results = ops.search_similar_images(
    image_path="path/to/query/image.jpg",
    top_k=5
)
```

## API Reference

### SimpleImageOps Class

#### `upload_image(image_path: str, text_description: Optional[str] = None) -> Optional[str]`

Uploads a single image to Pinecone with CLIP embeddings.

**Parameters:**
- `image_path` (str): Path to the image file
- `text_description` (str, optional): Text description for the image

**Returns:**
- `str`: UUID of the uploaded image, or `None` if failed

**Example:**
```python
image_id = ops.upload_image(
    "public/image1.jpg",
    "A person in a professional setting"
)
print(f"Uploaded with UUID: {image_id}")
```

#### `search_similar_images(image_path: str, top_k: int = 5) -> List[Any]`

Searches for similar images and returns top-k results.

**Parameters:**
- `image_path` (str): Path to the query image
- `top_k` (int): Number of results to return (default: 5)

**Returns:**
- `List[Any]`: List of search results with scores and metadata

**Example:**
```python
results = ops.search_similar_images("public/image2.jpg", top_k=5)
for result in results:
    print(f"Score: {result.score:.4f}")
    print(f"ID: {result.id}")
    print(f"File: {result.metadata.get('filename')}")
```

## How It Works

### 1. Image Processing
- Images are loaded and converted to RGB format
- CLIP model generates 512-dimensional embeddings
- Embeddings capture semantic meaning of image content

### 2. Vector Storage
- Each image gets a unique UUID
- Embeddings stored in Pinecone vector database
- Metadata includes filepath, filename, and optional text description

### 3. Similarity Search
- Query image is converted to CLIP embedding
- Pinecone performs cosine similarity search
- Results ranked by similarity score (higher = more similar)

## Configuration

### Pinecone Index Settings
- **Dimension**: 512 (CLIP embedding size)
- **Metric**: Cosine similarity
- **Cloud**: AWS (us-east-1)
- **Index Name**: "image-vectors"

### Supported Image Formats
- JPG/JPEG
- PNG
- BMP
- TIFF
- WebP

## Example Output

### Upload
```
✅ Uploaded image: public/image1.jpg
   UUID: 1e01f4d5-ac4e-42e3-9b32-3ddd096c9b7e
   Text: A person in a professional setting
```

### Search
```
🔍 Search results for: public/image2.jpg
Found 2 similar images:
  1. Score: 0.7287
     ID: 1e01f4d5-ac4e-42e3-9b32-3ddd096c9b7e
     File: image1.jpg
     Text: A person in a professional setting

  2. Score: 0.7281
     ID: c29290dc-4f7d-44fc-8614-dba95e6b6b39
     File: image1.jpg
     Text: A person in a professional setting
```

## Error Handling

The library includes robust error handling:
- Invalid image paths
- Network connection issues
- Pinecone API errors
- CLIP model loading failures

All functions return `None` or empty lists on failure with descriptive error messages.

## Performance

- **GPU Support**: Automatically uses CUDA if available
- **Batch Processing**: Single image operations for simplicity
- **Memory Efficient**: CLIP model loaded once and reused

## Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Pinecone Client
- Pillow (PIL)
- python-dotenv

## License

This project is open source and available under the MIT License. 