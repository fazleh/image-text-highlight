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
    min_mask_region_area=5000       # << reduces number of tiny masks massively
)

saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

for filename in os.listdir(input_folder):

    if not filename.lower().endswith(valid_extensions):
        continue

    image_path = os.path.join(input_folder, filename)
    print("Processing:", filename)

    # ---- load and resize image (RAM SAVER!)
    image = cv2.imread(image_path)
    if image is None:
        print("Warning: skip", filename)
        continue

    # ↓↓↓ VERY IMPORTANT ↓↓↓
    MAX_SIDE = 1100
    h, w = image.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w*scale), int(h*scale)))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ---- inference
    masks = mask_generator.generate(image_rgb)

    # ---- saliency
    success, saliency = saliency_detector.computeSaliency(image)
    if not success:
        saliency = np.ones(image.shape[:2], dtype=np.float32)

    saliency = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)

    # ---- score + free memory ASAP
    scored = []
    for m in masks:
        mask = m["segmentation"]
        score = saliency[mask].mean() * m["area"]
        scored.append((score, mask))

    # kill all SAM mask dicts immediately
    del masks

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:N]

    base = os.path.splitext(filename)[0]

    for i, (_, mask) in enumerate(top):
        out = np.zeros_like(image)
        out[mask] = image[mask]

        p = os.path.join(output_folder, f"{base}_seg_{i+1}.png")
        cv2.imwrite(p, out)

        del out
        del mask

    # FREE MEMORY
    del scored
    del top
    del image
    del image_rgb
    del saliency

    if device == "cuda":
        torch.cuda.empty_cache()

print("DONE")
