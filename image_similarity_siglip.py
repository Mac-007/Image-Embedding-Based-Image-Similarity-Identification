# -*- coding: utf-8 -*-
"""
===========================================================
Image Similarity using Google's SigLIP Vision-Text Model
===========================================================

Author: Dr.Amit Chougule, PhD
Created on: Wed Nov  5 12:41:36 2025

-----------------------------------------------------------
Overview:
-----------------------------------------------------------
This script compares the visual similarity between two images using
Google’s **SigLIP** model — a powerful open-source vision-text model.

-----------------------------------------------------------
What this script does:
-----------------------------------------------------------
1. Loads the **SigLIP** model and its associated processor.
2. Reads two image files from a local directory.
3. Preprocesses both images into model-compatible tensors.
4. Extracts **image embeddings** (feature representations) for both images.
5. Computes the **cosine similarity** between the two embeddings to measure
   how visually similar the images are.
6. Prints the similarity score and the first 10 embedding values for each image.

-----------------------------------------------------------
Usage:
-----------------------------------------------------------
- Place the script in a directory containing an `Images/` subfolder.
- Add two image files (e.g., Dog_1.jpg and Dog_3.jpg) in the `Images/` folder.
- Run the script with Python 3:
  
      python image_similarity_siglip.py

-----------------------------------------------------------
Dependencies:
-----------------------------------------------------------
- torch
- transformers
- Pillow (PIL)

-----------------------------------------------------------
Output:
-----------------------------------------------------------
- Cosine similarity value between the two images.
- First 10 dimensions of each image embedding (for reference).
"""

# ====== Imports ======
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
import os

# ====== Step 0. Define directories ======
# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define relative path to the image folder
IMAGE_PATH = "./Images/"

# ====== Step 1. Load model and processor ======
# Using Google's SigLIP model for high-quality image embeddings
model_name = "google/siglip-base-patch16-224"

# Load the processor (handles image preprocessing)
processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

# Load the pre-trained model
model = AutoModel.from_pretrained(model_name)

# ====== Step 2. Load the two images ======
# Replace these filenames with your own images if needed
img1_path = os.path.join(current_dir, IMAGE_PATH, "Dog_1.jpg")
img2_path = os.path.join(current_dir, IMAGE_PATH, "Dog_3.jpg")

# Open and convert both images to RGB format
image1 = Image.open(img1_path).convert("RGB")
image2 = Image.open(img2_path).convert("RGB")

# ====== Step 3. Preprocess and extract embeddings ======
# Convert images to tensors suitable for the model
inputs = processor(images=[image1, image2], return_tensors="pt")

# Disable gradient computation (we’re only performing inference)
with torch.no_grad():
    image_embeds = model.get_image_features(**inputs)

# Normalize embeddings (important for cosine similarity)
image_embeds = F.normalize(image_embeds, p=2, dim=-1)

# ====== Step 4. Compute cosine similarity ======
# Cosine similarity measures how close the two embedding vectors are
similarity = F.cosine_similarity(image_embeds[0], image_embeds[1], dim=0)
print(f"Cosine similarity: {similarity.item():.4f}")

# ====== Step 5. Print embedding samples (optional) ======
# Display the first 10 dimensions of each image embedding for inspection
print("\nEmbedding for Image 1 (first 10 dims):", image_embeds[0][:10])
print("Embedding for Image 2 (first 10 dims):", image_embeds[1][:10])
