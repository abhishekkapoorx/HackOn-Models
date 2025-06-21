import os
import uuid
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv
import pinecone
import boto3
from botocore.exceptions import ClientError
import glob
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
        
        # AWS S3 setup
        region = os.getenv('AWS_REGION', 'ap-south-1')
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=region
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'images-hackonn-fashion')
        self._setup_s3_bucket()
    
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
    
    def _setup_s3_bucket(self):
        """Create S3 bucket if it doesn't exist"""
        try:
            # Check if bucket exists
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"✅ S3 bucket '{self.bucket_name}' already exists")
        except ClientError as e:
            # If bucket doesn't exist, create it
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                try:
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                    print(f"✅ Created S3 bucket: {self.bucket_name}")
                    
                    # Make bucket public readable
                    self.s3_client.put_bucket_policy(
                        Bucket=self.bucket_name,
                        Policy=f'''{{
                            "Version": "2012-10-17",
                            "Statement": [
                                {{
                                    "Sid": "PublicReadGetObject",
                                    "Effect": "Allow",
                                    "Principal": "*",
                                    "Action": "s3:GetObject",
                                    "Resource": "arn:aws:s3:::{self.bucket_name}/*"
                                }}
                            ]
                        }}'''
                    )
                    print(f"✅ Made bucket {self.bucket_name} publicly readable")
                except ClientError as create_error:
                    print(f"❌ Failed to create S3 bucket: {create_error}")
            else:
                print(f"❌ Error checking S3 bucket: {e}")
    
    def upload_to_s3(self, image_path: str) -> Optional[str]:
        """Upload image to S3 and return public URL"""
        try:
            filename = os.path.basename(image_path)
            s3_key = f"images/{filename}"
            
            # Upload to S3
            self.s3_client.upload_file(
                image_path,
                self.bucket_name,
                s3_key,
                ExtraArgs={'ContentType': 'image/jpeg'}
            )
            
            # Generate public URL
            s3_url = f"https://{self.bucket_name}.s3.amazonaws.com/{s3_key}"
            print(f"✅ Uploaded to S3: {s3_url}")
            return s3_url
            
        except ClientError as e:
            print(f"❌ Failed to upload {image_path} to S3: {e}")
            return None
    
    def upload_image(self, image_path: str, text_description: Optional[str] = None, s3_url: Optional[str] = None) -> Optional[str]:
        """
        1. Upload single image to Pinecone with CLIP embeddings, UUID, and optional text
        
        Args:
            image_path: Path to the image file
            text_description: Text description for the image (optional)
            s3_url: S3 URL if image was uploaded to S3 (optional)
            
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
            
            if s3_url:
                metadata["s3_url"] = s3_url
                metadata["source"] = "s3"
            
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
            
            print(f"✅ Uploaded to Pinecone: {image_path}")
            print(f"   UUID: {image_id}")
            if s3_url:
                print(f"   S3 URL: {s3_url}")
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
    
    def process_fashion_images(self, images_dir: str = "./data/fashionFullData/fashion-dataset/images", max_images: int = 50):
        """
        Process fashion images: upload to S3 and then to Pinecone
        
        Args:
            images_dir: Directory containing fashion images
            max_images: Maximum number of images to process (default 50)
        """
        print(f"🚀 Starting batch processing of {max_images} images from {images_dir}")
        
        # Get image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(images_dir, ext)))
            image_files.extend(glob.glob(os.path.join(images_dir, ext.upper())))
        
        # Limit to max_images
        image_files = image_files[:max_images]
        
        print(f"📁 Found {len(image_files)} images to process")
        
        successful_uploads = 0
        failed_uploads = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n📸 Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
            
            try:
                # Step 1: Upload to S3
                s3_url = self.upload_to_s3(image_path)
                
                if s3_url:
                    # Step 2: Upload to Pinecone with S3 URL
                    filename = os.path.basename(image_path)
                    description = f"Fashion item - {filename}"
                    
                    image_id = self.upload_image(
                        image_path=image_path,
                        text_description=description,
                        s3_url=s3_url
                    )
                    
                    if image_id:
                        successful_uploads += 1
                        print(f"✅ Successfully processed {filename}")
                    else:
                        failed_uploads += 1
                        print(f"❌ Failed to upload to Pinecone: {filename}")
                else:
                    failed_uploads += 1
                    print(f"❌ Failed to upload to S3: {os.path.basename(image_path)}")
                    
            except Exception as e:
                failed_uploads += 1
                print(f"❌ Error processing {os.path.basename(image_path)}: {e}")
        
        print(f"\n🎉 Batch processing complete!")
        print(f"✅ Successful uploads: {successful_uploads}")
        print(f"❌ Failed uploads: {failed_uploads}")
        print(f"📊 Success rate: {successful_uploads / len(image_files) * 100:.1f}%")

# Example usage
if __name__ == "__main__":
    ops = SimpleImageOps()
    
    # Process 50 fashion images: upload to S3 and Pinecone
    ops.process_fashion_images(
        images_dir="./data/fashionFullData/fashion-dataset/images",
        max_images=50
    )
    
    # Example: Search for similar images
    # results = ops.search_similar_images("public/image1.jpg", top_k=5) 
    # print(results)