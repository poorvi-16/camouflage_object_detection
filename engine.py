import os
from ultralytics import YOLO
import cv2

class CamoDetector:
    def __init__(self, model_path='models/best.onnx'):
        # Load the optimized ONNX model
        self.model = YOLO(model_path, task='detect')
        
    def process_folder(self, input_dir='data', output_dir='exports/results'):
        os.makedirs(output_dir, exist_ok=True)
        
        images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Found {len(images)} images. Processing...")
        
        for img_name in images:
            img_path = os.path.join(input_dir, img_name)
            
            # Run inference
            results = self.model(img_path, conf=0.4)
            
            # Save the visualized result
            for r in results:
                res_img = r.plot()
                save_path = os.path.join(output_dir, f"detected_{img_name}")
                cv2.imwrite(save_path, res_img)
                
        print(f"Finished! Check your results in: {output_dir}")

if __name__ == "__main__":
    detector = CamoDetector()
    detector.process_folder()