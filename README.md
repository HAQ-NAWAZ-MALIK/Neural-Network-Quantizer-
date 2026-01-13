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

# Neural Network Weight Quantizer

Quantize neural network weights to lower precision formats (INT8, INT4, NF4) with interactive visualizations.

## Features

- 🔢 Multi-bit quantization (4-bit, 8-bit)
- 📊 Interactive weight visualizations
- 🤗 HuggingFace model support (optional)
- ⚡ GPU acceleration (when available)
- 📈 Quantization error analysis
- 🔄 Method comparison (INT8 vs INT4 vs NF4)

## Quick Start

1. Use the **Quantizer** tab to test on random weights
2. Compare different methods in the **Analysis** tab
3. Optionally load a HuggingFace model in the **Models** tab

## API

The backend exposes a REST API at `/api`:

- `GET /api/system/info` - System capabilities
- `POST /api/quantize/weights` - Quantize custom weights
- `POST /api/models/load` - Load HuggingFace model
- `POST /api/analysis/compare` - Compare methods

## 🚀 Deployment

### Hugging Face Spaces
This project is configured for **Hugging Face Spaces** using the Docker SDK.

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Select **Docker** as the SDK.
3. Push this repository to your Space:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push space main
   ```

### Docker
Run locally with Docker:
```bash
docker build -t quantizer .
docker run -p 7860:7860 quantizer
```
Open `http://localhost:7860`.
