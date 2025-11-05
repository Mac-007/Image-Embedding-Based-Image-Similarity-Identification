# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 12:41:36 2025

@author: amitc
"""

# ====== Imports ======
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
import os

# Current Dir Path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Image Dir Path
Image_Path = "./Images/"

# ====== Step 1. Load model and processor ======
# SigLIP: free and high-quality vision-text model from Google
model_name = "google/siglip-base-patch16-224"
processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
model = AutoModel.from_pretrained(model_name)

# ====== Step 2. Load your two images ======
# Replace with your own file paths
img1_path = os.path.join(current_dir, Image_Path, "Dog_1.jpg")
img2_path =  os.path.join(current_dir, Image_Path, "Dog_3.jpg")

image1 = Image.open(img1_path).convert("RGB")
image2 = Image.open(img2_path).convert("RGB")

# ====== Step 3. Preprocess and get embeddings ======
inputs = processor(images=[image1, image2], return_tensors="pt")

with torch.no_grad():
    image_embeds = model.get_image_features(**inputs)

# Normalize embeddings (important for similarity)
image_embeds = F.normalize(image_embeds, p=2, dim=-1)

# ====== Step 4. Compute cosine similarity ======
#similarity = torch.matmul(image_embeds[0], image_embeds[1].T)
similarity = F.cosine_similarity(image_embeds[0], image_embeds[1], dim=0)
print(f"Cosine similarity: {similarity.item():.4f}")

# ====== Step 5. Print embeddings (optional) ======
print("\nEmbedding for Image 1 (first 10 dims):", image_embeds[0][:10])
print("Embedding for Image 2 (first 10 dims):", image_embeds[1][:10])
