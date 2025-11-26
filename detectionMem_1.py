import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# -------------------
# CONFIG
# -------------------
input_folder = "documents/input_images"
sam_checkpoint = "documents/sam_vit_h_4b8939.pth"
model_type = "vit_h"
N = 5  # number of top masks to save
output_folder = "documents/segments"

os.makedirs(output_folder, exist_ok=True)

# -------------------
# LOAD MODEL
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    pred_iou_thresh=0.8,
    stability_score_thresh=0.9,
    box_nms_thresh=0.7,
    min_mask_region_area=5000
)

saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# -------------------
# MAIN LOOP
# -------------------
for filename in os.listdir(input_folder):

    if not filename.lower().endswith(valid_extensions):
        continue

    image_path = os.path.join(input_folder, filename)
    print("Processing:", filename)

    image = cv2.imread(image_path)
    if image is None:
        print("Warning: skip", filename)
        continue

    # Resize if needed
    MAX_SIDE = 1100
    h, w = image.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w*scale), int(h*scale)))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate SAM masks
    masks = mask_generator.generate(image_rgb)

    # Compute saliency
    success, saliency = saliency_detector.computeSaliency(image)
    if not success:
        saliency = np.ones(image.shape[:2], dtype=np.float32)
    saliency = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)

    # Score masks
    scored = []
    for m in masks:
        mask = m["segmentation"]
        score = saliency[mask].mean() * m["area"]
        scored.append((score, mask))
    del masks

    # Sort and keep top N
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:N]

    base = os.path.splitext(filename)[0]

    for i, (_, mask) in enumerate(top):
        # Find bounding box of mask
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue

        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()

        # Crop original image to mask region
        cropped = image[y1:y2+1, x1:x2+1]

        # Save the cropped region
        out_path = os.path.join(output_folder, f"{base}_seg_{i+1}.png")
        cv2.imwrite(out_path, cropped)

        print(f"Saved: {out_path}")

        del cropped
        del mask

    # Free memory
    del scored
    del top
    del image
    del image_rgb
    del saliency
    if device == "cuda":
        torch.cuda.empty_cache()

print("DONE")
