import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# -------------------
# CONFIG
# -------------------
input_folder = "documents/input_images"
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

mask_generator = SamAutomaticMaskGenerator(
    sam,
    min_mask_region_area=5000      # <<< MASSIVE MEMORY SAVER
)

# -------------------
# PROCESS
# -------------------
os.makedirs(output_folder, exist_ok=True)
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

for filename in os.listdir(input_folder):

    if not filename.lower().endswith(valid_extensions):
        continue

    print(f"Processing: {filename}")
    image_path = os.path.join(input_folder, filename)

    # -------- LOAD IMAGE --------
    image = cv2.imread(image_path)
    if image is None:
        print("Warning: unreadable image, skipping.")
        continue

    # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    # CRUCIAL PART – reduce size first
    MAX_SIDE = 1100
    h, w = image.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        image = cv2.resize(image, (int(w*s), int(h*s)))
    # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # -------- INFERENCE ----------
    masks = mask_generator.generate(image_rgb)


    # -------- SORT + PICK N ------
    masks_sorted = sorted(masks, key=lambda m: m["area"], reverse=True)
    topN_masks = masks_sorted[:N]


    # -------- SAVE RESULT --------
    base_name = os.path.splitext(filename)[0]

    for i, m in enumerate(topN_masks):
        mask = m["segmentation"]
        result = np.zeros_like(image)
        result[mask] = image[mask]

        out = os.path.join(output_folder, f"{base_name}_segment_{i+1}.png")
        cv2.imwrite(out, result)

        del result
        del mask

    # MEMORY CLEANUP
    del masks
    del masks_sorted
    del topN_masks
    del image_rgb
    del image

    if device == "cuda":
        torch.cuda.empty_cache()

print("DONE")
