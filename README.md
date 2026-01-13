---
title: Neural Network Quantizer
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Neural Network Weight Quantizer & Visualizer

A comprehensive Full-Stack application for analyzing, simulating, and visualizing the effects of 8-bit, 4-bit, and NF4 (Normal Float 4) quantization on neural network weights.

![Status](https://img.shields.io/badge/status-active-success.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Tech](https://img.shields.io/badge/tech-Pytorch%20|%20FastAPI%20|%20React-blueviolet)

## 📌 Overview

This tool allows researchers and developers to understand how different quantization methods affect weight distributions and model precision. By visualizing the quantization process, it helps in selecting the right strategy for model compression.

## ⭐ Key Features

*   **Multi-Method Quantization**: Support for INT8, INT4 (Asymmetric/Symmetric), and NF4 (QLoRA style).
*   **Real-time Visualization**: Interactive heatmaps and histograms showing weight usage before and after quantization.
*   **Error Analysis**: Automatic calculation of MSE (Mean Squared Error), L1 Norm, and Memory Savings.
*   **Hugging Face Integration**: Load layers directly from HF Transformers models.
*   **Dynamic Configuration**: Adjust bit-width, group size, and quantization modes on the fly.
*   **GPU Acceleration**: Automatically utilizes CUDA if available for faster processing.

## 🎯 Use Cases

1.  **Educational**: Teaching tool to demonstrate how quantization works at the bit level.
2.  **Research**: Experimenting with new codebooks (like NF4) or custom quantization kernels.
3.  **Model Optimization**: Profiling layers to determine which can be aggressively quantized without accuracy loss.

## 🏗️ Architecture

For a deep dive into the system design, please see the [ARCHITECTURE.md](ARCHITECTURE.md) file.

## 🚀 Getting Started

### Prerequisites

*   Python 3.10+
*   Node.js 18+
*   Git

### 1. Clone the Repository

```bash
git clone https://github.com/HAQ-NAWAZ-MALIK/Neural-Network-Quantizer-.git
cd Neural-Network-Quantizer-
```

### 2. Backend Setup (FastAPI)

```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Frontend Setup (React)

```bash
cd ../frontend
npm install
```

### 4. Running Locally

**Terminal 1 (Backend):**
```bash
cd backend
python ../app.py
```
*Server runs on http://localhost:7860*

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```
*UI runs on http://localhost:5173*

## 🐳 Docker Deployment

The project is fully containerized for easy deployment.

```bash
# Build the image
docker build -t quantizer .

# Run the container
docker run -p 7860:7860 quantizer
```
Access the application at `http://localhost:7860`.

## 📦 API Reference

The backend exposes a Swagger UI for exploring endpoints.

*   **Docs**: `http://localhost:7860/api/docs`
*   **Redoc**: `http://localhost:7860/api/redoc`

**Key Endpoints:**
*   `POST /api/quantize/weights`: Core quantization engine.
*   `POST /api/models/load`: Load specific layers from HF models.
*   `POST /api/analysis`: Statistical analysis of weight tensors.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
