# peanuts

A PyTorch implementation of neural network models for seismic phase detection and arrival time picking.

## Getting Started

```shell
uv sync
uv run train
```

---

## 🔬 Models

This repository implements four neural network architectures, all **corrected to follow their respective paper implementations**:

### 📊 PhaseNet
- **Paper**: "PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method" (Zhu & Beroza, 2018)
- **Architecture**: U-Net based with (7,1) kernels for time-series data
- **Output**: 3-class probability distribution (P-wave, S-wave, Noise) with Softmax2d
- **Features**: Optimized for seismic waveform analysis

### 👁️ Attention U-Net
- **Paper**: "Attention U-Net: Learning Where to Look for the Pancreas" (Oktay et al., 2018)
- **Architecture**: U-Net with attention gates in decoder
- **Output**: 3-class probability distribution with attention-guided feature selection
- **Features**: Spatial attention mechanism for better feature focus

### 🔄 R2U-Net
- **Paper**: "Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)" (Alom et al., 2018)
- **Architecture**: U-Net with Recurrent Convolutional Layers (RCL) and residual connections
- **Output**: 3-class probability distribution with enhanced feature representation
- **Features**: Recurrent blocks with configurable time steps (default t=2)

### 🎯 R2AU-Net
- **Paper**: Combination of R2U-Net and Attention U-Net
- **Architecture**: Integrates RCL blocks with attention gates
- **Output**: 3-class probability distribution with both recurrent and attention mechanisms
- **Features**: Best of both worlds - recurrent processing and spatial attention

## ✅ Paper Compliance

All models have been **corrected to follow their respective paper implementations**:

- ✅ **Softmax2d output layers** for proper probability distributions
- ✅ **Correct skip connection ordering** (residual first, then upsampled features)
- ✅ **Proper attention gate implementation** with interpolation and element-wise operations
- ✅ **Paper-compliant RCL blocks** with configurable time steps
- ✅ **Accurate down/up convolution sequences** matching original architectures

## 🧪 Testing

Run the compliance verification test:

```bash
python test_paper_compliance.py
```

This will verify:
- Model output shapes and probability distributions
- Skip connection implementations
- Attention mechanism functionality
- RCL recurrent processing

## 🏗️ Architecture Details

### Common Features
- **Input**: 3-channel seismic data (E-N-Z components)
- **Kernel Size**: (7,1) optimized for time-series analysis
- **Stride**: (4,1) for temporal downsampling
- **Output**: 3-class probability maps via Softmax2d

### Model-Specific Features

| Model | RCL Blocks | Attention Gates | Residual Connections | Parameters |
|-------|------------|-----------------|---------------------|------------|
| PhaseNet | ❌ | ❌ | ❌ | ~0.5M |
| Attention U-Net | ❌ | ✅ | ❌ | ~0.7M |
| R2U-Net | ✅ | ❌ | ✅ | ~1.2M |
| R2AU-Net | ✅ | ✅ | ✅ | ~1.5M |

## 📈 Usage

```python
from peanuts.models import PhaseNet, AttentionUNet, R2UNet, R2AUNet
import torch

# Initialize any model
model = PhaseNet()  # or AttentionUNet(), R2UNet(), R2AUNet()

# Input: (batch_size, 3, height, width)
input_data = torch.randn(2, 3, 128, 256)

# Forward pass
output = model(input_data)
# Output: (2, 3, 128, 256) probability distribution

# Extract predictions
predictions = torch.argmax(output, dim=1)  # Class predictions
probabilities = output  # Raw probabilities for each class
```

## 🔍 Key Improvements Made

### 1. **Output Layer Standardization**
- Added `nn.Softmax2d()` to all models for proper probability outputs
- Ensures sum of probabilities = 1.0 across classes for each pixel

### 2. **Skip Connection Correction**
- Fixed concatenation order: `[residual, upsampled]` (paper-compliant)
- Consistent across all U-Net variants

### 3. **Attention Gate Enhancement**
- Proper interpolation handling with `bilinear` mode
- Clear implementation of W_g, W_x, and ψ transformations
- Safe size handling for different scale features

### 4. **RCL Block Optimization**
- Configurable time steps (t parameter)
- Proper recurrent formulation: `h[t] = f(W_h * h[t-1] + W_x * x)`
- Default t=2 following paper recommendations

### 5. **Architecture Consistency**
- All models follow the same down/up convolution patterns
- Proper residual connections in R2U-Net variants
- Consistent kernel sizes and stride patterns

## 📚 References

1. Zhu, W., & Beroza, G. C. (2018). PhaseNet: A deep-neural-network-based seismic arrival time picking method. arXiv preprint arXiv:1803.03211.

2. Oktay, O., et al. (2018). Attention u-net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999.

3. Alom, M. Z., et al. (2018). Recurrent residual convolutional neural network based on u-net (r2u-net) for medical image segmentation. arXiv preprint arXiv:1802.06955.

## 🎯 Project Goals

This implementation aims to provide:
- **Paper-accurate** implementations of established architectures
- **Seismic-optimized** configurations for geophysical applications  
- **Consistent interfaces** across all model variants
- **Thorough testing** for implementation verification
