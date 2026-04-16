from ultralytics import YOLO

# 1. Load your PyTorch model
model = YOLO('models/best.pt')

# 2. Export to ONNX format
# 'dynamic=True' allows the model to handle different image sizes
# 'int8=True' or 'half=True' can further compress it, but let's start with standard
path = model.export(format='onnx', dynamic=True)

print(f"Success! Optimized model saved at: {path}")