# Runpod Setup Guide for Nasdaq Futures Trading System

## Quick Start on Runpod A5000

### 1. Launch Runpod Instance

1. **Choose GPU**: Select **RTX A5000** (24GB VRAM) for optimal performance
2. **Container**: Use **RunPod PyTorch 2.4.0** template with CUDA 12.4
3. **Storage**: Add **50-100GB** network volume mounted at `/workspace`
4. **Ports**: Expose port **8888** for Jupyter Lab

### 2. Initial Setup

Once your pod is running, connect via Jupyter Lab or SSH and run:

```bash
# Navigate to workspace
cd /workspace

# Clone the project (if not already done)
git clone <your-repo-url> kuant-trading
cd kuant-trading

# Install Runpod-optimized requirements
pip install -r requirements_runpod.txt

# If pytorch-forecasting fails, try the alternative installation:
pip install git+https://github.com/sktime/pytorch-forecasting.git
```

### 3. Fix Common Issues

If you encounter `ModuleNotFoundError: No module named 'pytorch_forecasting'`:

```bash
# Method 1: Install specific version
pip install pytorch-forecasting==1.0.0 --no-deps
pip install -r requirements_runpod.txt

# Method 2: Install from source
pip uninstall pytorch-forecasting -y
pip install git+https://github.com/sktime/pytorch-forecasting.git

# Method 3: Install with conda (if available)
conda install pytorch-forecasting pytorch -c pytorch -c conda-forge
```

### 4. Verify Installation

```python
import torch
import pytorch_forecasting as ptf
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"PyTorch Forecasting: {ptf.__version__}")
```

Expected output for A5000:
```
PyTorch: 2.4.0+cu124
CUDA Available: True
GPU: NVIDIA RTX A5000
PyTorch Forecasting: 1.0.0 (or similar)
```

### 5. Run Training

```bash
# Run cloud training (auto-detects Runpod)
python train_cloud.py

# Or specify dates
python train_cloud.py --start-date 2023-01-01 --end-date 2024-01-01

# Force Runpod environment (if detection fails)
python train_cloud.py --cloud-env runpod
```

## Runpod-Specific Optimizations

The training script automatically applies these optimizations for Runpod A5000:

- **Batch Size**: 48 (optimized for 24GB VRAM)
- **Mixed Precision**: Enabled (A5000 has Tensor Cores)
- **Memory Management**: 90% GPU memory allocation
- **TF32**: Enabled for faster training
- **Persistent Storage**: Uses `/workspace` directory
- **Checkpointing**: Aggressive saving for spot instances

## Cost Optimization Tips

1. **Use Spot Instances**: ~50% cheaper, script handles interruptions
2. **Stop When Idle**: Always stop your pod when not training
3. **Use Network Volumes**: Keep data persistent across sessions
4. **Monitor Usage**: Check GPU utilization with `nvidia-smi`

## Troubleshooting

### Memory Issues
```bash
# Check GPU memory
nvidia-smi

# Reduce batch size in train_cloud.py if needed
# Edit line ~130: 'batch_size': 32  # Reduced from 48
```

### CUDA Issues
```bash
# Check CUDA version
nvidia-smi
nvcc --version

# Reinstall PyTorch if needed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Storage Issues
```bash
# Check disk space
df -h

# Clean up if needed
docker system prune -f
pip cache purge
```

## Model Serving on Runpod

After training, you can deploy your model as a Runpod Serverless Endpoint:

1. Save your trained model to `/workspace/models/`
2. Create a simple inference script
3. Deploy using Runpod's Serverless platform
4. Scale automatically based on demand

## Performance Expectations

On RTX A5000 (24GB):
- **TFT Training**: ~2-3 hours for full dataset
- **RL Training**: ~4-6 hours for 1M timesteps
- **Memory Usage**: ~20GB for optimal batch sizes
- **Training Speed**: ~30% faster than RTX 3090

## Support

If you encounter issues:
1. Check the training logs in `/workspace/logs/`
2. Verify GPU utilization with `nvidia-smi`
3. Review the error messages in Jupyter Lab
4. Use Runpod's community Discord for platform-specific help 