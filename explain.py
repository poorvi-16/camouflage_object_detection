import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO

# 1. Load the model
model = YOLO('models/best.pt')

# 2. Setup the "Hook" 
# This catches the data as it flows through the model
activations = {}
def get_activations(name):
    def hook(model, input, output):
        # YOLOv8 outputs can be complex; we ensure we get the tensor
        if isinstance(output, (list, tuple)):
            activations[name] = output[0].detach()
        else:
            activations[name] = output.detach()
    return hook

# We attach the hook to the last layer of the Backbone (usually layer 9 or 22)
# This is where the most 'meaningful' features are gathered
target_layer = model.model.model[-2]
target_layer.register_forward_hook(get_activations('target_layer'))

# 3. Process the Image
img_path = 'data/test_image.png'
if not os.path.exists(img_path):
    print("Error: test_image.png not found.")
else:
    img = cv2.imread(img_path)
    # YOLO expects 640x640
    input_img = cv2.resize(img, (640, 640))
    img_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    # 4. Run Inference (This triggers the hook)
    with torch.no_grad():
        model.model(img_tensor)

    # 5. Process the Heatmap
    # We take the mean across the channels to see the most active areas
    act = activations['target_layer'].squeeze(0)
    heatmap = torch.mean(act, dim=0).cpu().numpy()

    # Normalize the heatmap between 0 and 1
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 6. Overlay Heatmap on Image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Merge heatmap with original image (60% image, 40% heatmap)
    result = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # 7. Save
    os.makedirs('exports', exist_ok=True)
    cv2.imwrite('exports/heatmap.png', result)
    print("SUCCESS! Heatmap generated manually in exports/heatmap.png")