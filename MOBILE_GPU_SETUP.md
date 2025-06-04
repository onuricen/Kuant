# Mobile GPU Training Guide - GTX 1060 Ti & Similar

This guide helps you train the Nasdaq Futures Hybrid ML Trading System on mobile GPUs with limited VRAM (4-8GB).

## 🧪 **NEW: Post-Training Accuracy Testing**

After training, validate your model performance with comprehensive testing:

```bash
# Test trained models automatically
python test_model_accuracy.py

# Mobile GPU specific testing with smaller dataset
python test_model_accuracy.py --test-start 2024-10-01 --test-end 2024-12-31
```

**Validates:**
- 🧠 **Model Accuracy**: TFT predictions, RL trading performance
- 🛡️ **Risk Compliance**: Position sizing, drawdown protection  
- 📈 **Market Robustness**: Performance across different conditions
- 🔗 **System Integration**: TFT-RL signal alignment
- 📊 **Performance Metrics**: Win rate, Sharpe ratio, profit factor

## Hardware Requirements

### Minimum Specs
- **GPU**: GTX 1060 Ti Mobile (6GB VRAM) or equivalent
- **RAM**: 16GB system RAM (32GB recommended)
- **Storage**: 10GB free space for data and models
- **CUDA**: Version 11.0+ (check with `nvidia-smi`)

### Supported GPUs
- GTX 1060 Ti (6GB) ✅
- GTX 1070 (8GB) ✅ 
- GTX 1080 (8GB) ✅
- RTX 2060 (6GB) ✅
- RTX 3060 (6-8GB) ✅

## Quick Start

### 1. Install Dependencies
```bash
# Use mobile-optimized requirements
pip install -r requirements_mobile.txt

# Verify CUDA installation
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 2. Check GPU Memory
```bash
# Monitor GPU memory
nvidia-smi

# Check available VRAM in Python
python -c "import torch; print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')"
```

### 3. Run Mobile Training
```bash
# Quick training with optimized settings
python train_mobile_gpu.py

# With specific date range (recommended for mobile)
python train_mobile_gpu.py --start-date "2010-06-07" --end-date "2010-08-07"

# Force CPU if GPU issues
python train_mobile_gpu.py --cpu-only
```

## Mobile GPU Optimizations

### Automatic Optimizations Applied
- **Batch Size**: 16 (vs 64 standard)
- **Hidden Size**: 64 (vs 128 standard)
- **Lookback Periods**: 100 (vs 200 standard)
- **Attention Heads**: 2 (vs 4 standard)
- **Training Epochs**: 50 (vs 100 standard)
- **RL Timesteps**: 500K (vs 1M standard)

### Memory Management Features
- **Automatic VRAM monitoring**
- **Gradient accumulation** for effective larger batches
- **Memory cleanup** between training phases
- **Float32 precision** instead of float64
- **Single environment** instead of vectorized

## Training Performance Expectations

### GTX 1060 Ti (6GB) Estimates
- **Data Loading**: 2-3 minutes
- **TFT Training**: 45-90 minutes
- **RL Training**: 60-120 minutes
- **Total Time**: 2-4 hours
- **Memory Usage**: 4-5GB VRAM peak

### Performance vs Standard Training
| Metric | Standard | Mobile GPU | Impact |
|--------|----------|------------|---------|
| Training Speed | 100% | 60-80% | Acceptable |
| Model Quality | 100% | 85-95% | Minimal loss |
| Memory Usage | 8-12GB | 4-6GB | 50% reduction |

## Troubleshooting

### Common Issues & Solutions

#### 1. CUDA Out of Memory
```bash
# Error: RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce batch size further: `config.tft.BATCH_SIZE = 8`
- Use CPU training: `--cpu-only`
- Close other GPU applications
- Restart and try again

#### 2. Driver Issues
```bash
# Check CUDA version compatibility
nvcc --version
nvidia-smi
```
**Solutions:**
- Update GPU drivers
- Install compatible PyTorch version
- Use CPU fallback

#### 3. Slow Training
**Optimizations:**
- Reduce dataset size
- Lower model complexity
- Use mixed precision (if supported)

### Memory Monitoring Commands
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Python memory tracking
python -c "
import torch
print(f'Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB')
print(f'Reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f}GB')
"
```

## Advanced Configuration

### Custom Mobile Settings
```python
# In src/config.py, modify MobileGPUConfig:
class CustomMobileConfig(MobileGPUConfig):
    def __init__(self):
        super().__init__()
        # Ultra-conservative settings for 4GB GPUs
        self.tft.BATCH_SIZE = 8
        self.tft.HIDDEN_SIZE = 32
        self.tft.LOOKBACK_PERIODS = 50
```

### Environment Variables
```bash
# Force specific GPU
export CUDA_VISIBLE_DEVICES=0

# Reduce PyTorch memory caching
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Model Quality Considerations

### Quality vs Performance Trade-offs
- **Hidden Size 64**: ~10% accuracy reduction
- **Lookback 100**: ~5% accuracy reduction  
- **Batch Size 16**: ~3% accuracy reduction
- **Combined**: ~15-20% overall reduction

### Mitigation Strategies
1. **Longer Training**: More epochs compensate for smaller batches
2. **Learning Rate Adjustment**: Lower LR for stability
3. **Ensemble Methods**: Train multiple models
4. **Transfer Learning**: Start from larger model

## Monitoring & Logging

### Training Progress
```bash
# View training logs
tail -f logs/mobile_training.log

# MLflow tracking (if enabled)
mlflow ui --port 5000
```

### Performance Metrics
- **GPU Utilization**: Should be 80-95%
- **VRAM Usage**: Should be <90% of available
- **Temperature**: Keep <80°C for mobile GPUs

## Alternative Approaches

### If Mobile GPU Training Fails

#### 1. CPU-Only Training
```bash
python train_mobile_gpu.py --cpu-only
# Expect 5-10x slower but guaranteed to work
```

#### 2. Cloud Training
- **Google Colab**: Free T4 GPU
- **Kaggle Notebooks**: Free GPU time
- **AWS EC2**: p3.2xlarge instances

#### 3. Hybrid Approach
```bash
# Train TFT on cloud, RL locally
python train_mobile_gpu.py --rl-only --tft-path cloud_model.pt
```

## Expected Results

### Successful Mobile Training Output
```
=== Mobile Training Pipeline Completed ===
Average Episode Reward: 45.67
Models saved to:
  TFT Model: models/tft_mobile_model.pt
  RL Model: models/rl_mobile_final.zip
```

### Model Files Sizes
- **TFT Model**: ~50-100MB
- **RL Model**: ~10-20MB
- **Processed Data**: ~100-200MB

## Next Steps After Training

1. **Evaluate Performance**: Run backtests
2. **Model Comparison**: vs CPU/cloud trained models
3. **Production Deployment**: Use mobile models for inference
4. **Continuous Learning**: Retrain with new data

## Support

### Getting Help
- Check logs in `logs/mobile_training.log`
- Monitor GPU with `nvidia-smi`
- Review memory usage patterns
- Try CPU fallback if persistent issues

### Community Resources
- PyTorch Mobile GPU guides
- Stable-Baselines3 documentation
- pytorch-forecasting memory optimization tips 