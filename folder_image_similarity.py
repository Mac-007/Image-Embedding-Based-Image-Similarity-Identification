# -*- coding: utf-8 -*-
"""
===========================================================
Folder-Wide Image Similarity Comparison using Google's SigLIP
===========================================================

Author: Dr.Amit Chougule, PhD 
Created on: Wed Nov  5 12:41:36 2025  

-----------------------------------------------------------
Overview:
-----------------------------------------------------------
This script compares all images within a specified folder using
Google’s **SigLIP** model — a free and high-quality vision-text model.

It calculates cosine similarities between every pair of image embeddings
and identifies which two images in the folder are the most visually similar.

-----------------------------------------------------------
What this script does:
-----------------------------------------------------------
1. Loads the **SigLIP** model and processor.
2. Reads all image files in a given folder.
3. Preprocesses all images into model-ready tensors.
4. Generates **image embeddings** (numerical feature vectors).
5. Computes **pairwise cosine similarities** between all unique image pairs.
6. Prints:
   - The most visually similar image pair.
   - Their cosine similarity score.
   - (Optionally) All pairwise similarity scores for inspection.
   - Also, provides top 3 (or top-k) images in the folder are the most visually similar.

-----------------------------------------------------------
Usage:
-----------------------------------------------------------
- Place this script in a directory that contains an `Images/` folder.
- Add multiple image files (e.g., JPG, PNG, etc.) inside the folder.
- Run the script with Python 3:

      python folder_image_similarity.py

-----------------------------------------------------------
Dependencies:
-----------------------------------------------------------
- torch  
- transformers  
- Pillow (PIL)  
- itertools (standard library)

-----------------------------------------------------------
Output:
-----------------------------------------------------------
- The names of the two most similar images in the folder.
- Their cosine similarity value.
- Optional: A list of similarity scores for every image pair.
"""

# ====== Imports ======
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
import os
import itertools

# ====== Step 0. Define paths ======
current_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(current_dir, "Images")

# ====== Step 1. Load model and processor ======
model_name = "google/siglip-base-patch16-224"
processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
model = AutoModel.from_pretrained(model_name)

# ====== Step 2. Collect all image file paths ======
supported_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(supported_exts)]

if len(image_files) < 2:
    raise ValueError("Need at least two images in the folder to compare.")

print(f"Found {len(image_files)} images in: {image_dir}")

# ====== Step 3. Load and preprocess images ======
images = [Image.open(os.path.join(image_dir, f)).convert("RGB") for f in image_files]
inputs = processor(images=images, return_tensors="pt", padding=True)

# ====== Step 4. Generate image embeddings ======
with torch.no_grad():
    embeddings = model.get_image_features(**inputs)

embeddings = F.normalize(embeddings, p=2, dim=-1)

# ====== Step 5. Compute pairwise cosine similarities ======
n = len(image_files)
similarities = []
pairs = []

for i, j in itertools.combinations(range(n), 2):
    sim = F.cosine_similarity(embeddings[i], embeddings[j], dim=0).item()
    similarities.append(sim)
    pairs.append((image_files[i], image_files[j]))

# ====== Step 6. Display the most similar pair ======
max_sim_idx = torch.argmax(torch.tensor(similarities)).item()
most_similar_pair = pairs[max_sim_idx]
max_sim = similarities[max_sim_idx]

print("\nMost similar image pair:")
print(f"{most_similar_pair[0]}  <-->  {most_similar_pair[1]}")
print(f"Cosine similarity: {max_sim:.4f}")

# ====== Optional: Display all pairwise similarities ======
print("\nAll pairwise similarities:")
for (img1, img2), sim in zip(pairs, similarities):
    print(f"{img1} <-> {img2}: {sim:.4f}")

# ====== Step 7. Get Top-K Most Similar Pairs ======
top_k = 3
top_indices = torch.topk(torch.tensor(similarities), k=top_k).indices

print(f"\nTop {top_k} most similar image pairs:")
for rank, idx in enumerate(top_indices):
    img1, img2 = pairs[idx]
    sim = similarities[idx]
    print(f"{rank+1}. {img1} <-> {img2} : {sim:.4f}")