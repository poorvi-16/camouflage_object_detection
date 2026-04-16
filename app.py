import gradio as gr
from ultralytics import YOLO
import cv2

# Load the optimized model
model = YOLO('models/best.onnx', task='detect')

def predict(image):
    results = model(image, conf=0.3)
    for r in results:
        return r.plot()

# Custom CSS for a professional dark theme
custom_css = """
body { background-color: #0b0f19; }
.gradio-container { font-family: 'Inter', sans-serif; }
#title { text-align: center; color: #ffffff; margin-bottom: 20px; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🛡️ Camouflage AI Engine", elem_id="title")
    gr.Markdown("### Professional-grade hidden object detection using YOLOv8 & ONNX.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="Input Source")
            btn = gr.Button("Analyze Image", variant="primary")
        with gr.Column():
            output_img = gr.Image(type="numpy", label="Detection Result")
            
    btn.click(fn=predict, inputs=input_img, outputs=output_img)
    
    gr.Markdown("---")
    gr.Markdown("Developed by Poorvika Srinivas | CS Engineering | Optimized for CPU Inference")

if __name__ == "__main__":
    demo.launch()