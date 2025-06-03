# Sensor Network Project Setup with VS Code

## Technologies Used
- **Programming Language**: Python 3.8+
- **Key Libraries**:
  - `numpy` (Numerical computing)
  - `matplotlib` (Data visualization)
  - `torch` (PyTorch for deep learning)
  - `tkinter` (GUI for visualization)
- **IDE**: Visual Studio Code (VS Code)
- **OS**: Ubuntu

## Step-by-Step Setup Guide

### 1. Install Python
1. Download from [python.org/downloads](https://www.python.org/downloads/)
2. **Check "Add Python to PATH"** during installation
3. Verify installation:
   ```bash
   python --version
   pip --version

## Install VS Code
1. Download [VS Code](https://code.visualstudio.com/download)

### Install these extensions:

1. Python (Microsoft)

2. Pylance


## Install Required Libraries
```bash
    pip install numpy matplotlib torch
```
### For GPU acceleration (NVIDIA CUDA):
```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Recommended Hyperparamiter : 
```python
    NETWORK_AREA = [100, 100]
    NUM_OF_SENSORS_IN_NETWORK = 10
    NUM_OF_MAIN_UNITS = 1
    BATTERY_CAPACITY = 100
    TRANSMISSION_COST = 1
    MINING_COST = 0.5
    NUM_NODES = NUM_OF_SENSORS_IN_NETWORK + NUM_OF_MAIN_UNITS
    LOAD_BEST_MODEL = False
    SAVE_BEST_MODEL = True
```

## Hardware Requirements
Minimum: 4GB RAM, Dual-core CPU
Recommended: 8GB+ RAM, NVIDIA GPU