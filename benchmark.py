import time
from ultralytics import YOLO

# Load both models
pt_model = YOLO('models/best.pt')
onnx_model = YOLO('models/best.onnx', task='detect')

# Test image path
img_path = 'data/test_image.png' # Ensure you have an image here

# Benchmark PyTorch
start = time.time()
pt_model.predict(img_path, device='cpu')
print(f"PyTorch Inference Time: {time.time() - start:.4f}s")

# Benchmark ONNX
start = time.time()
onnx_model.predict(img_path, device='cpu')
print(f"ONNX Inference Time: {time.time() - start:.4f}s")