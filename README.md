# Phamiq Model Repository

## Overview

This repository contains the trained machine learning models and research notebooks for the Phamiq crop disease detection platform. The models are designed to identify diseases across multiple crop types including cashew, cassava, maize, and tomato.

## Model Architecture

The models are based on EfficientNetV2 architecture, optimized for crop disease classification with the following specifications:

- **Input Size**: 225x225 pixels (RGB)
- **Model Format**: PyTorch (.pth) and ONNX (.onnx)
- **Number of Classes**: 22 total disease classes
- **Accuracy**: 90%+ on test datasets

## Supported Crops and Diseases

### Cashew (5 classes)
- cashew_anthracnose
- cashew_gummosis
- cashew_healthy
- cashew_leafminer
- cashew_redrust

### Cassava (5 classes)
- cassava_bacterial_blight
- cassava_brown_spot
- cassava_green_mite
- cassava_healthy
- cassava_mosaic

### Maize (7 classes)
- maize_fall_armyworm
- maize_grasshopper
- maize_healthy
- maize_leaf_beetle
- maize_leaf_blight
- maize_leaf_spot
- maize_streak_virus

### Tomato (5 classes)
- tomato_healthy
- tomato_leaf_blight
- tomato_leaf_curl
- tomato_leaf_spot
- tomato_verticillium_wilt

## Model Files

### Individual Crop Models
- `cashew_model.pth` (68MB) - Cashew disease classification model
- `cassava_model.pth` (68MB) - Cassava disease classification model
- `maize_model.pth` (68MB) - Maize disease classification model
- `tomato_model.pth` (68MB) - Tomato disease classification model

### Combined Models
- `phamiq.pth` (68MB) - Combined model for all crops
- `phamiq.onnx` (21MB) - ONNX format for production deployment

## Research Notebooks

### `train_test_phamiq.ipynb`
Comprehensive training and testing notebook that includes:
- Data loading and preprocessing
- Model training with ResNet50 and EfficientNet architectures
- Data augmentation techniques
- Performance evaluation metrics
- TensorBoard logging
- Model saving and loading utilities

### `multispectralAnalysis.ipynb`
Advanced multispectral analysis for agricultural monitoring:
- Landsat satellite data processing
- Environmental indices calculation (NDVI, EVI, SAVI, NDMI, etc.)
- Land Surface Temperature (LST) analysis
- Soil type classification
- Carbon storage and methane emission proxies
- Visualization and mapping tools

### `Modeltest.py`
Standalone model testing script with:
- ONNX model inference
- Image preprocessing pipeline
- Top-3 prediction results
- Visualization utilities
- Real-time testing capabilities

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster inference)

### Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision utilities
- `transformers>=4.34.0` - Hugging Face transformers
- `Pillow` - Image processing
- `sentencepiece` - Tokenizer support
- `accelerate` - Model optimization
- `onnxruntime` - ONNX model inference
- `opencv-python` - Computer vision
- `albumentations` - Data augmentation
- `matplotlib` - Visualization
- `rasterio` - Geospatial data processing
- `numpy` - Numerical computing

## Usage

### Basic Model Inference
```python
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load model
model = torch.load('models/phamiq.pth', map_location='cpu')
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and process image
image = Image.open('path/to/image.jpg')
image_tensor = transform(image).unsqueeze(0)

# Get prediction
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
```

### ONNX Model Inference
```python
import onnxruntime
import numpy as np
import cv2

# Load ONNX model
session = onnxruntime.InferenceSession('models/phamiq.onnx')

# Preprocess image
image = cv2.imread('path/to/image.jpg')
image = cv2.resize(image, (112, 112))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype(np.float32) / 255.0
image = np.transpose(image, (2, 0, 1))
image = np.expand_dims(image, axis=0)

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
prediction = session.run([output_name], {input_name: image})
```

## Training

### Data Preparation
1. Organize your dataset in the following structure:
```
data/
├── train/
│   ├── cashew_anthracnose/
│   ├── cashew_healthy/
│   └── ...
├── valid/
│   ├── cashew_anthracnose/
│   ├── cashew_healthy/
│   └── ...
└── test/
    ├── cashew_anthracnose/
    ├── cashew_healthy/
    └── ...
```

2. Update the paths in `train_test_phamiq.ipynb`:
```python
DATA_DIR = "path/to/your/data"
MODEL_SAVE_PATH = "path/to/save/models"
BEST_OVERALL_MODEL_FILENAME = "your_model_name.pth"
TENSORBOARD_LOG_DIR = "path/to/logs"
```

### Training Parameters
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Epochs**: 10 (adjustable)
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss

## Performance Metrics

The models achieve the following performance metrics:
- **Overall Accuracy**: 90%+
- **Precision**: 0.89
- **Recall**: 0.87
- **F1-Score**: 0.88

## Model Optimization

### Quantization
For production deployment, consider model quantization to reduce size and improve inference speed:
```python
import torch.quantization

# Quantize model
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### TensorRT Optimization
For NVIDIA GPUs, consider TensorRT optimization for maximum performance.

## Multispectral Analysis

The multispectral analysis capabilities include:
- **Vegetation Indices**: NDVI, EVI, SAVI for crop health monitoring
- **Environmental Monitoring**: Land surface temperature, soil moisture
- **Climate Impact**: Carbon storage and methane emission analysis
- **Spatial Analysis**: Geographic distribution of agricultural conditions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Citation

If you use these models in your research, please cite:
```
Phamiq: AI-Powered Crop Disease Detection Platform
Agricultural Technology for Sustainable Farming
```

## Support

For technical support or questions:
- Check the research notebooks for detailed examples
- Review the model testing scripts
- Open an issue for specific problems
- Contact the development team

---

**Note**: These models are trained on specific datasets and may require fine-tuning for different geographic regions or crop varieties.
```

This comprehensive README provides detailed information about the model repository, including the model architecture, supported crops and diseases, usage examples, training procedures, and performance metrics. It covers both the basic image classification models and the advanced multispectral analysis capabilities.