import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide

# -------------------
# CONFIG
# -------------------
input_folder = "documents/input_images"
sam_checkpoint = "documents/sam_vit_h_4b8939.pth"
model_type = "vit_h"
N = 5
output_folder = "documents/segments"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", device)


# -------------------
# LOAD SAM
# -------------------
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
sam.eval()

mask_generator = SamAutomaticMaskGenerator(sam)

transform = ResizeLongestSide(sam.image_encoder.img_size)


# -------------------
# ATTENTION MAP FUNC
# -------------------
def get_attention_map(sam_model, image_np, layer=-1, head=0):

    img_resized = transform.apply_image(image_np)
    img_tensor = torch.as_tensor(img_resized).permute(2,0,1).contiguous().float()[None] / 255
    img_tensor = sam_model.preprocess(img_tensor).to(device)

    features = {}

    def hook_fn(module, inp, out):
        # out is (B, heads, tokens, tokens)
        features["attn"] = out[1]

    if layer == -1:
        layer_idx = len(sam_model.image_encoder.blocks) - 1
    else:
        layer_idx = layer

    h = sam_model.image_encoder.blocks[layer_idx].attn.attn_drop.register_forward_hook(hook_fn)

    with torch.no_grad():
        sam_model.image_encoder(img_tensor)

    h.remove()

    attn = features["attn"][0]               # (heads, tokens, tokens)
    attn = attn[head]                        # (tokens, tokens)
    attn = attn[0, 1:]                       # CLS → patches

    side = int(np.sqrt(attn.shape[0]))
    attn_map = attn.reshape(side, side).cpu().numpy()

    return attn_map



# -------------------
# PROCESS IMAGES
# -------------------
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):

    if not filename.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff")):
        continue

    print("Processing:", filename)

    path = os.path.join(input_folder,filename)
    image = cv2.imread(path)

    if image is None:
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H,W = image.shape[:2]


    # --- SAM masks ---
    masks = mask_generator.generate(image_rgb)


    # --- attention ---
    attn_map = get_attention_map(sam, image_rgb)

    attn_map = cv2.resize(attn_map, (W,H))
    attn_map = (attn_map - attn_map.min())/(attn_map.max()-attn_map.min())


    # --- score masks ---
    scored=[]
    for m in masks:
        msk = m["segmentation"]
        score = attn_map[msk].mean() * m["area"]
        scored.append((score,m))

    scored.sort(key=lambda x:x[0], reverse=True)
    top = [m for _,m in scored[:N]]

    base = os.path.splitext(filename)[0]

    for i,m in enumerate(top):
        mask = m["segmentation"]
        out = np.zeros_like(image)
        out[mask]=image[mask]
        cv2.imwrite(f"{output_folder}/{base}_seg_{i+1}.png", out)

    heat = cv2.applyColorMap((attn_map*255).astype(np.uint8), cv2.COLORMAP_JET)
    ov = cv2.addWeighted(image,0.6,heat,0.4,0)
    cv2.imwrite(f"{output_folder}/{base}_attn.png", ov)

print("DONE!")
