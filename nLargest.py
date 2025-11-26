import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# -------------------
# CONFIG
# -------------------
input_folder = "documents/input_images"  # folder containing all images
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
N = 5
output_folder = "documents/segments"

# -------------------
# LOAD SAM MODEL
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", device)

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

mask_generator = SamAutomaticMaskGenerator(sam)

# -------------------
# PROCESS ALL IMAGES IN FOLDER
# -------------------
os.makedirs(output_folder, exist_ok=True)

# only process common image formats
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(valid_extensions):
        continue

    image_path = os.path.join(input_folder, filename)
    print(f"Processing: {filename}")

    # LOAD IMAGE
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: could not read {filename}, skipping")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # RUN INFERENCE
    masks = mask_generator.generate(image_rgb)

    # SORT MASKS BY AREA
    masks_sorted = sorted(masks, key=lambda m: m["area"], reverse=True)
    topN_masks = masks_sorted[:N]

    # SAVE TOP N SEGMENTS
    for i, m in enumerate(topN_masks):
        mask = m["segmentation"]
        result = np.zeros_like(image)
        result[mask] = image[mask]

        # output filename starts with input image name
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, f"{base_name}_segment_{i + 1}.png")
        cv2.imwrite(output_path, result)

print(f"All done! Top {N} segments saved in folder: {output_folder}/")
