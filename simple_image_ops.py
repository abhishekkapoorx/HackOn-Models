import os
import uuid
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv
import pinecone
from typing import Optional, List, Any

# Load environment variables
load_dotenv()

class SimpleImageOps:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_ID = "openai/clip-vit-base-patch32"
        self.model, self.processor, _ = self._load_clip_model()
        self.pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = "image-vectors"
        self._setup_index()
    
    def _load_clip_model(self):
        """Load CLIP model and processor"""
        model = CLIPModel.from_pretrained(self.model_ID).to(self.device) # type: ignore
        processor = CLIPProcessor.from_pretrained(self.model_ID)
        return model, processor, None
    
    def _setup_index(self):
        """Create or connect to Pinecone index"""
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=512,
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pc.describe_index(self.index_name).status["ready"]:
                import time
                time.sleep(1)
        
        self.index = self.pc.Index(self.index_name)
    
    def upload_image(self, image_path: str, text_description: Optional[str] = None) -> Optional[str]:
        """
        1. Upload single image to Pinecone with CLIP embeddings, UUID, and optional text
        
        Args:
            image_path: Path to the image file
            text_description: Text description for the image (optional)
            
        Returns:
            UUID of the uploaded image, or None if failed
        """
        try:
            # Generate UUID
            image_id = str(uuid.uuid4())
            
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(
                text=None,
                images=image,
                return_tensors="pt"
            )["pixel_values"].to(self.device)
            
            # Generate CLIP embedding
            with torch.no_grad():
                embedding = self.model.get_image_features(inputs)
            
            embedding_np = embedding.cpu().numpy().flatten()
            
            # Prepare metadata
            metadata = {
                "image_path": image_path,
                "filename": os.path.basename(image_path),
                "type": "image"
            }
            
            if text_description:
                metadata["text_description"] = text_description
            
            # Upload to Pinecone
            self.index.upsert(
                vectors=[{
                    "id": image_id,
                    "values": embedding_np.tolist(),
                    "metadata": metadata
                }]
            )
            
            print(f"✅ Uploaded image: {image_path}")
            print(f"   UUID: {image_id}")
            if text_description:
                print(f"   Text: {text_description}")
            
            return image_id
            
        except Exception as e:
            print(f"❌ Failed to upload image {image_path}: {e}")
            return None
    
    def search_similar_images(self, image_path: str, top_k: int = 5) -> List[Any]:
        """
        2. Search for similar images and return top 5 results
        
        Args:
            image_path: Path to the query image
            top_k: Number of results to return (default 5)
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            # Load and process query image
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(
                text=None,
                images=image,
                return_tensors="pt"
            )["pixel_values"].to(self.device)
            
            # Generate CLIP embedding for query
            with torch.no_grad():
                query_embedding = self.model.get_image_features(inputs)
            
            query_embedding_np = query_embedding.cpu().numpy().flatten()
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding_np.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            print(f"🔍 Search results for: {image_path}")
            print(f"Found {len(results.matches)} similar images:")
            
            for i, result in enumerate(results.matches, 1):
                print(f"  {i}. Score: {result.score:.4f}")
                print(f"     ID: {result.id}")
                print(f"     File: {result.metadata.get('filename', 'Unknown')}")
                if 'text_description' in result.metadata:
                    print(f"     Text: {result.metadata['text_description']}")
                print()
            
            return results.matches
            
        except Exception as e:
            print(f"❌ Failed to search for similar images: {e}")
            return []

# Example usage
if __name__ == "__main__":
    ops = SimpleImageOps()
    
    # Example 1: Upload an image with text description
    image_id = ops.upload_image(
        "public/image2.jpg", 
        text_description="A person in a professional setting"
    )
    
    # Example 2: Search for similar images
    results = ops.search_similar_images("public/image1.jpg", top_k=5) 

    print(results)