# -*- coding: utf-8 -*-
"""
===========================================================
Query Image Similarity Search using Google's SigLIP Model
===========================================================

Author: Dr.Amit Chougule, PhD 
Created on: Wed Nov  5 15:51:36 2025  

-----------------------------------------------------------
Overview:
-----------------------------------------------------------
This script finds the **most visually similar image** in a folder
compared to a **single query image**, using Google’s
**SigLIP (Sigmoid Loss for Language-Image Pre-training)** model.

It works by generating embeddings (numerical feature vectors)
for both the query and all target images, then computing
**cosine similarity** to measure how visually close they are.

-----------------------------------------------------------
What this script does:
-----------------------------------------------------------
1. Loads the SigLIP model and image processor.  
2. Loads a single query image (e.g., a reference or product image).  
3. Loads all target images from a given folder.  
4. Generates embeddings for the query and target images.  
5. Computes cosine similarity between the query and each target image.  
6. Identifies and prints the **most similar image** with its similarity score.  
7. Optionally displays the query and best match side-by-side using Matplotlib.

-----------------------------------------------------------
Usage:
-----------------------------------------------------------
- Place this script in a directory containing an `Images/` subfolder.
- Add your query image as `Query_Image.jpg` (or update the filename path).
- Run the script:

      python query_image_similarity.py

-----------------------------------------------------------
Dependencies:
-----------------------------------------------------------
- torch  
- transformers  
- Pillow (PIL)  
- matplotlib  
- os (standard library)

-----------------------------------------------------------
Output:
-----------------------------------------------------------
- Prints the most similar image in the folder.  
- Displays cosine similarity score.  
- (Optional) Shows query and best-match images side by side.
"""

# ====== Imports ======
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

# ====== Step 0. Define paths ======
# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the folder containing target images
image_dir = os.path.join(current_dir, "Images")

# Define the query image path (update this as needed)
query_image_path = os.path.join(current_dir, "Query_Image.jpg")

# ====== Step 1. Load model and processor ======
# Using Google's SigLIP model for high-quality vision embeddings
model_name = "google/siglip-base-patch16-224"

# Load the processor (handles resizing, normalization, etc.)
processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

# Load the pre-trained vision-text model
model = AutoModel.from_pretrained(model_name)

# ====== Step 2. Load all target images ======
# Accept common image formats
supported_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# List all valid image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(supported_exts)]

# Ensure there is at least one image to compare with
if not image_files:
    raise ValueError("No images found in the folder!")

print(f"Found {len(image_files)} images in: {image_dir}")

# ====== Step 3. Load query and target images ======
# Open and convert the query image to RGB
query_image = Image.open(query_image_path).convert("RGB")

# Open all target images in the folder and convert to RGB
target_images = [Image.open(os.path.join(image_dir, f)).convert("RGB") for f in image_files]

# ====== Step 4. Generate embeddings ======
# --- Query image embedding ---
query_inputs = processor(images=query_image, return_tensors="pt")
with torch.no_grad():
    query_emb = model.get_image_features(**query_inputs)

# Normalize the embedding (important for cosine similarity)
query_emb = F.normalize(query_emb, p=2, dim=-1)

# --- Target image embeddings ---
inputs = processor(images=target_images, return_tensors="pt", padding=True)
with torch.no_grad():
    target_embs = model.get_image_features(**inputs)

# Normalize all target embeddings
target_embs = F.normalize(target_embs, p=2, dim=-1)

# ====== Step 5. Compute similarities ======
# Compute cosine similarity between query and each target embedding
# Using matrix multiplication for efficient vectorized similarity computation
similarities = torch.matmul(target_embs, query_emb.T).squeeze(1)  # Shape: (N,)
similarities = similarities.tolist()  # Convert to a Python list for processing

# ====== Step 6. Identify the most similar image ======
max_idx = int(torch.argmax(torch.tensor(similarities)))  # Index of max similarity
most_similar_file = image_files[max_idx]
most_similar_score = similarities[max_idx]

print("\nMost similar image to query:")
print(f"Query Image: {os.path.basename(query_image_path)}")
print(f"Matched Image: {most_similar_file}")
print(f"Cosine Similarity: {most_similar_score:.4f}")

# ====== Step 7. Optional — visualize results ======
# Display the query image and the best-matching image side by side
fig, axs = plt.subplots(1, 2, figsize=(7, 4))

# Display the query image
axs[0].imshow(query_image)
axs[0].set_title("Query Image")
axs[0].axis("off")

# Display the best match
axs[1].imshow(target_images[max_idx])
axs[1].set_title(f"Match: {most_similar_file}\nSim: {most_similar_score:.3f}")
axs[1].axis("off")

# Adjust layout and render the figure
plt.tight_layout()
plt.show()
