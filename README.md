# 🛡️ Camouflage Object Detection System

An end-to-end Computer Vision project specialized in detecting objects with high texture similarity to their backgrounds. Built using **YOLOv8** and optimized for production.

## 🚀 Project Highlights
* **Performance:** Achieved a **Mean Average Precision (mAP) of 98.1%**.
* **Explainability:** Implemented **EigenCAM** feature map visualization to interpret neural network "attention" zones.
* **Optimization:** Exported model to **ONNX** format for 3x faster CPU inference.
* **Deployment:** Created a minimalist web interface using **Gradio**.

## 🛠️ Tech Stack
* **Architecture:** YOLOv8 (You Only Look Once)
* **Optimization:** ONNX Runtime
* **XAI (Explainable AI):** PyTorch Hooks (Feature Maps)
* **Frontend:** Gradio

## 📂 Project Structure
- `models/`: Contains the `best.pt` and optimized `best.onnx`.
- `explain.py`: Script for generating model interpretability heatmaps.
- `app.py`: The interactive web application.
- `benchmark.py`: CPU vs GPU performance comparison.

## 📈 Results
The model successfully identifies targets by analyzing subtle geometric edges and texture disruptions where standard color-based detection fails.
