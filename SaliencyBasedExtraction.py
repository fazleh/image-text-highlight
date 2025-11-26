import os
import cv2
import numpy as np

INPUT_DIR = "documents/input_images"
OUTPUT_DIR = "documents/segments"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_salient_region(img):
    """
    Computes saliency map using OpenCV's StaticSaliencySpectralResidual
    and returns a cropped salient region.
    """

    # Initialize saliency detector
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

    # Compute saliency map (float32 between 0 and 1)
    success, saliency_map = saliency.computeSaliency(img)
    if not success:
        raise RuntimeError("Saliency computation failed.")

    # Convert saliency to 0–255 for thresholding
    saliency_uint8 = (saliency_map * 255).astype("uint8")

    # Threshold (Otsu automatically finds best threshold)
    _, mask = cv2.threshold(saliency_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find bounding box of salient region
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None  # no region found

    x, y, w, h = cv2.boundingRect(coords)

    # Crop the salient part
    cropped = img[y:y + h, x:x + w]

    return saliency_uint8, mask, cropped


def process_all_images():
    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue

        path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(path)

        if img is None:
            print(f"Skipping unreadable file: {filename}")
            continue

        try:
            saliency_map, mask, cropped = extract_salient_region(img)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

        # Save outputs
        base = os.path.splitext(filename)[0]
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_saliency.png"), saliency_map)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_mask.png"), mask)

        if cropped is not None:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_segment.png"), cropped)
        else:
            print(f"No salient region for {filename}")

        print(f"Processed {filename}")


if __name__ == "__main__":
    process_all_images()
