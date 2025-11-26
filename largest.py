import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# -----------------------------------------------------------
# 1. Input/Output folders
# -----------------------------------------------------------
input_dir = r"documents/input_images"
output_dir = r"documents/segments"
os.makedirs(output_dir, exist_ok=True)


# -----------------------------------------------------------
# 2. Load SAM ViT-B (low memory)
# -----------------------------------------------------------
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


# -----------------------------------------------------------
# 3. Stable low-memory SAM settings
# -----------------------------------------------------------
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=0,
    points_per_batch=8
)


# -----------------------------------------------------------
# Extract ROI from mask
# -----------------------------------------------------------
def crop_object_from_mask(image, mask):
    """
    image: BGR image
    mask: boolean mask (H, W)
    returns: cropped image with transparency
    """
    # Bounding box of the mask
    ys, xs = np.where(mask)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    # Crop image & mask
    cropped_img = image[y1:y2+1, x1:x2+1]
    cropped_mask = mask[y1:y2+1, x1:x2+1]

    # Create transparent PNG
    rgba = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = (cropped_mask * 255).astype(np.uint8)

    return rgba


# -----------------------------------------------------------
# 4. Process images
# -----------------------------------------------------------
for filename in os.listdir(input_dir):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        continue

    img_path = os.path.join(input_dir, filename)
    print(f"\nProcessing {filename} ...")

    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Cannot read: {filename}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate SAM masks
    try:
        masks = mask_generator.generate(image_rgb)
    except Exception as e:
        print(f"❌ SAM error on {filename}: {e}")
        continue

    if len(masks) == 0:
        print(f"⚠ No segments in {filename}")
        continue

    # Find the largest mask = largest object
    largest_mask = max(masks, key=lambda m: m["area"])
    mask = largest_mask["segmentation"]

    # Crop that object
    cropped = crop_object_from_mask(image, mask)

    # Save output as transparent PNG
    out_path = os.path.join(
        output_dir,
        f"{os.path.splitext(filename)[0]}_largest_object.png"
    )
    cv2.imwrite(out_path, cropped)

    print(f"✅ Saved largest object: {out_path}")


print("\n🎉 Finished!")
