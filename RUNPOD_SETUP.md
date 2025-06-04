# Runpod Setup Guide for Nasdaq Futures Trading System

## Quick Start on Runpod A4500/A5000

### 1. Launch Runpod Instance

1. **Choose GPU**: Select **RTX A4500** (20GB VRAM) or **RTX A5000** (24GB VRAM) for optimal performance
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

# 🔧 CRITICAL FIX: Fix pytorch-forecasting compatibility
python fix_pytorch_forecasting.py

# OR use the built-in verification in training script:
python train_cloud.py --verify-pytorch-forecasting

# If the above fails, manually install compatible version:
# pip uninstall pytorch-forecasting -y
# pip install pytorch-forecasting==1.0.0

# Verify the fix worked
python -c "
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer
print(f'TFT is LightningModule: {issubclass(TemporalFusionTransformer, pl.LightningModule)}')
"
```

Expected output should show: `TFT is LightningModule: True`

### 3. Test Installation

```bash
# Test basic imports
python -c "
import torch
import pytorch_forecasting
import stable_baselines3
print('✅ All packages imported successfully')
print(f'GPU Available: {torch.cuda.is_available()}')
print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"
```

### 4. Check Cache and Start Training

```bash
# Check if any cached data exists
python train_cloud.py --cache-info

# Start training (will cache data automatically)
python train_cloud.py

# Or start with specific date range for faster initial testing
python train_cloud.py --start-date 2024-01-01 --end-date 2024-06-01
```

## 🔧 Troubleshooting

### PyTorch Forecasting Import Issues

If you get: `TypeError: 'model' must be a 'LightningModule'`

**Solution 1 (Automatic Fix):**
```bash
python fix_pytorch_forecasting.py

# OR use built-in verification:
python train_cloud.py --verify-pytorch-forecasting
```

**Solution 2 (Manual Fix):**
```bash
# Find pytorch-forecasting installation
python -c "import pytorch_forecasting; print(pytorch_forecasting.__file__)"

# Replace the imports (adjust path as needed)
find /usr/local/lib/python*/site-packages/pytorch_forecasting -name "*.py" -exec sed -i.bak 's/lightning\.pytorch/pytorch_lightning/g' {} \;
```

**Solution 3 (Version Downgrade):**
```bash
pip uninstall pytorch-forecasting -y
pip install pytorch-forecasting==1.0.0 pytorch-lightning==2.0.8
```

### GPU Memory Issues

```bash
# Check GPU memory
nvidia-smi

# If out of memory, reduce batch size
export TFT_BATCH_SIZE=32  # Default is 48 for A5000
python train_cloud.py
```

### Data Loading Slow

```bash
# Use cached data (after first run)
python train_cloud.py  # Uses cache automatically

# Force fresh processing
python train_cloud.py --force-refresh

# Check cache status
python train_cloud.py --cache-info
```

## 🚀 Training Options

### Quick Test (Recommended First)
```bash
# Small dataset for testing
python train_cloud.py --start-date 2024-01-01 --end-date 2024-03-01
```

### Full Training
```bash
# Full dataset (will take longer)
python train_cloud.py
```

### Cache Management
```bash
# View cache
python train_cloud.py --cache-info

# Clear cache (if needed)
python train_cloud.py --clear-cache
```

## 💡 Performance Tips

### A4500 (20GB VRAM)
- Batch size: 32-40
- Use mixed precision: ✅ Enabled automatically
- Expected training time: 2-4 hours full dataset

### A5000 (24GB VRAM)  
- Batch size: 48-64
- Use mixed precision: ✅ Enabled automatically
- Expected training time: 1.5-3 hours full dataset

### Spot Instances
- Aggressive checkpointing: ✅ Enabled automatically
- Use `/workspace` for persistence
- Consider shorter training runs with caching

## 📁 File Structure

```
/workspace/kuant-trading/
├── data/cache/          # Cached processed data
├── models/              # Trained models
├── logs/               # Training logs
├── train_cloud.py      # Main training script
└── fix_pytorch_forecasting.py  # Compatibility fix
```

After successful training, models will be saved in:
- `models/tft_cloud_runpod.pt` 
- `models/rl_final_runpod.zip`

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