# Cloud Training Guide - Remote Training Providers

This guide covers all the best options for training your Nasdaq Futures Hybrid ML Trading System on cloud platforms, from free to enterprise-grade solutions.

## 🧪 **NEW: Post-Training Accuracy Validation**

After training completion, your system includes **comprehensive accuracy testing**:

```bash
# Automatically test trained models
python test_model_accuracy.py

# Test with specific models and date range  
python test_model_accuracy.py \
  --tft-model models/tft_model.pt \
  --rl-model models/rl_final_model.zip \
  --test-start 2024-07-01 \
  --test-end 2024-12-31
```

**5-Layer Testing Suite:**
- 🧠 **TFT Model Accuracy**: Prediction accuracy, directional accuracy, confidence calibration
- 🎮 **RL Agent Performance**: Trading returns, win rate, Sharpe ratio, risk metrics  
- 🔗 **Hybrid Integration**: TFT-RL signal alignment and decision consistency
- 🛡️ **Risk Management**: Position sizing compliance, drawdown protection
- 📈 **Market Regime Testing**: Performance across trending/ranging, high/low volatility

**Results:**
- Real-time console dashboard with pass/fail results
- Detailed JSON report (`model_accuracy_results.json`)
- Performance assessment (EXCELLENT/GOOD/NEEDS IMPROVEMENT)
- Specific recommendations for model improvement

## 🏆 Recommended Cloud Providers

### 1. **Free Options** (Great for Testing & Learning)

#### Google Colab (⭐ **Best Free Option**)
- **GPU**: Tesla T4 (16GB VRAM) - FREE for 12 hours
- **Pro ($10/month)**: A100/V100 access, 24-hour sessions
- **Setup Time**: 2 minutes
- **Perfect for**: Prototyping, learning, short training runs

**Quick Start:**
```python
# In Colab cell
!git clone https://github.com/yourusername/nasdaq-futures-trading.git
%cd nasdaq-futures-trading
!pip install -r requirements.txt
!python train_cloud.py
```

#### Kaggle Notebooks
- **GPU**: Tesla P100/T4 (16GB VRAM) - FREE 30 hours/week
- **Dataset Integration**: Built-in data hosting
- **Time Limit**: 9 hours per session
- **Perfect for**: Competition-style training, public sharing

#### Paperspace Gradient (Free Tier)
- **GPU**: M4000 (8GB VRAM) - FREE 6 hours/month
- **Persistent Storage**: Available
- **Perfect for**: Testing cloud deployment

### 2. **Paid Cloud Options** (Production Quality)

#### AWS EC2 (⭐ **Best Performance**)
| Instance Type | GPU | VRAM | Cost/Hour | Best For |
|---------------|-----|------|-----------|----------|
| g4dn.xlarge | T4 | 16GB | $0.55 | Development |
| p3.2xlarge | V100 | 16GB | $3.06 | Training |
| p4d.24xlarge | A100 | 40GB | $32.77 | Large Scale |

**Advantages:**
- Best price/performance ratio
- Spot instances up to 70% cheaper
- Complete infrastructure control
- Easy scaling

#### Google Cloud Platform (GCP)
| Instance Type | GPU | VRAM | Cost/Hour | Best For |
|---------------|-----|------|-----------|----------|
| n1-standard-4 + K80 | K80 | 12GB | $0.45 | Budget |
| n1-standard-8 + T4 | T4 | 16GB | $0.95 | Balanced |
| a2-highgpu-1g + A100 | A100 | 40GB | $3.67 | High-end |

**Advantages:**
- Excellent TPU options
- Google AI Platform integration
- Strong MLOps tools

#### Azure Machine Learning
| Instance Type | GPU | VRAM | Cost/Hour | Best For |
|---------------|-----|------|-----------|----------|
| Standard_NC6 | K80 | 12GB | $0.90 | Budget |
| Standard_NC6s_v3 | V100 | 16GB | $3.06 | Training |
| Standard_ND40rs_v2 | V100 | 32GB | $22.03 | Large Scale |

**Advantages:**
- Enterprise integration
- Strong security features
- Hybrid cloud options

### 3. **Specialized ML Platforms**

#### Lambda Labs (⭐ **Best Value for ML**)
| Instance Type | GPU | VRAM | Cost/Hour | Best For |
|---------------|-----|------|-----------|----------|
| 1x RTX 6000 Ada | RTX 6000 | 48GB | $1.10 | Excellent Value |
| 1x A100 PCIe | A100 | 80GB | $1.29 | High Memory |
| 8x A100 SXM4 | A100 | 640GB | $12.00 | Massive Scale |

**Advantages:**
- Best price for ML workloads
- Pre-configured ML environments
- Excellent customer support

#### Paperspace Core
| Instance Type | GPU | VRAM | Cost/Hour | Best For |
|---------------|-----|------|-----------|----------|
| P4000 | P4000 | 8GB | $0.45 | Development |
| P5000 | P5000 | 16GB | $0.78 | Training |
| V100 | V100 | 16GB | $2.30 | Production |

**Advantages:**
- Simple setup and management
- Good documentation
- Jupyter notebook integration

#### RunPod
| Instance Type | GPU | VRAM | Cost/Hour | Best For |
|---------------|-----|------|-----------|----------|
| RTX 3070 | RTX 3070 | 8GB | $0.34 | Budget |
| RTX 3090 | RTX 3090 | 24GB | $0.69 | High Memory |
| A40 | A40 | 48GB | $1.25 | Production |

**Advantages:**
- Very competitive pricing
- Gaming GPU options
- Flexible billing

## 🚀 Quick Setup Guides

### Google Colab Setup (Recommended for Beginners)

1. **Open Colab**
   ```
   https://colab.research.google.com/
   ```

2. **Upload Your Code**
   ```python
   # Mount Google Drive
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Clone repository
   !git clone https://github.com/yourusername/nasdaq-futures-trading.git
   %cd nasdaq-futures-trading
   
   # Install dependencies
   !pip install -r requirements.txt
   
   # Check GPU
   import torch
   print(f"GPU Available: {torch.cuda.is_available()}")
   print(f"GPU: {torch.cuda.get_device_name(0)}")
   ```

3. **Upload Data**
   ```python
   # Option 1: Upload from local
   from google.colab import files
   uploaded = files.upload()
   
   # Option 2: Use Google Drive
   !cp /content/drive/MyDrive/trading_data/* ./data/
   ```

4. **Run Training**
   ```python
   !python train_cloud.py
   ```

### AWS EC2 Setup (Best Performance)

1. **Launch Instance**
   ```bash
   # AMI: Deep Learning AMI (Ubuntu 18.04)
   # Instance: p3.2xlarge (V100, 16GB VRAM)
   # Storage: 100GB EBS
   # Security Group: SSH (22), Jupyter (8888)
   ```

2. **Connect and Setup**
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   
   # Clone repository
   git clone https://github.com/yourusername/nasdaq-futures-trading.git
   cd nasdaq-futures-trading
   
   # Install dependencies (conda already available)
   conda create -n trading python=3.9
   conda activate trading
   pip install -r requirements.txt
   ```

3. **Start Training**
   ```bash
   # Run training
   python train_cloud.py
   
   # Or use screen for long sessions
   screen -S trading
   python train_cloud.py
   # Ctrl+A, D to detach
   ```

4. **Monitor Progress**
   ```bash
   # Reattach to screen
   screen -r trading
   
   # Check GPU usage
   nvidia-smi
   
   # View logs
   tail -f logs/cloud_training_aws.log
   ```

### Lambda Labs Setup (Best Value)

1. **Create Instance**
   ```
   https://lambdalabs.com/service/gpu-cloud
   # Choose: 1x RTX 6000 Ada (48GB VRAM)
   # OS: Ubuntu 20.04 + PyTorch
   ```

2. **SSH Setup**
   ```bash
   ssh ubuntu@your-lambda-ip
   
   # Environment is pre-configured
   git clone https://github.com/yourusername/nasdaq-futures-trading.git
   cd nasdaq-futures-trading
   pip install -r requirements.txt
   ```

3. **Run Training**
   ```bash
   python train_cloud.py
   ```

## 📊 Cost Comparison & Recommendations

### Training Time Estimates (Full Dataset)
| Provider | Instance | Cost/Hour | Training Time | Total Cost |
|----------|----------|-----------|---------------|------------|
| **Colab Pro** | A100 | $0.00-10 | 1-2 hours | **$0-10** |
| **Lambda** | RTX 6000 | $1.10 | 2-3 hours | **$2-4** |
| **AWS Spot** | p3.2xlarge | $0.92 | 1.5-2 hours | **$1-2** |
| **AWS On-Demand** | p3.2xlarge | $3.06 | 1.5-2 hours | **$5-6** |
| **GCP** | T4 | $0.95 | 3-4 hours | **$3-4** |

### 💡 **Money-Saving Tips**

1. **Use Spot/Preemptible Instances** (AWS/GCP)
   - 60-90% cheaper than on-demand
   - Risk: Can be interrupted
   - Solution: Enable checkpointing

2. **Start with Free Tiers**
   - Colab: 12 hours free daily
   - Kaggle: 30 hours/week
   - Test your code before paying

3. **Optimize Training Time**
   - Use smaller datasets for initial testing
   - Enable checkpointing for resumability
   - Monitor GPU utilization

## 🔧 Cloud-Specific Optimizations

### Colab Optimizations
```python
# Enable GPU in Colab
# Runtime > Change runtime type > GPU

# Use Colab-specific configs
!python train_cloud.py --cloud-env colab

# Save models to Drive
from google.colab import drive
drive.mount('/content/drive')
# Models auto-save to Drive with cloud script
```

### AWS Optimizations
```bash
# Use AWS CLI for data transfer
aws s3 cp s3://your-bucket/data/ ./data/ --recursive

# Spot instance automation
aws ec2 request-spot-instances \
  --spot-price "1.00" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification file://spot-config.json
```

### Multi-GPU Training (Enterprise)
```python
# For multiple GPUs (p3.8xlarge, p4d.24xlarge)
!python train_cloud.py --multi-gpu

# Distributed training across instances
# Requires custom setup
```

## 📁 Data Management

### Upload Methods

1. **Small Data (<1GB)**
   ```python
   # Direct upload in Colab/Kaggle
   from google.colab import files
   uploaded = files.upload()
   ```

2. **Large Data (>1GB)**
   ```bash
   # AWS S3
   aws s3 cp your-data.zip s3://your-bucket/
   aws s3 cp s3://your-bucket/your-data.zip ./
   
   # Google Cloud Storage
   gsutil cp your-data.zip gs://your-bucket/
   gsutil cp gs://your-bucket/your-data.zip ./
   ```

3. **GitHub LFS** (for code + small data)
   ```bash
   git lfs track "*.csv"
   git add .gitattributes
   git add data/*.csv
   git commit -m "Add data"
   git push
   ```

### Cloud Storage Integration
```python
# AWS S3 integration
import boto3
s3 = boto3.client('s3')
s3.download_file('bucket-name', 'data.zip', 'local-data.zip')

# Google Cloud Storage
from google.cloud import storage
client = storage.Client()
bucket = client.bucket('your-bucket')
blob = bucket.blob('data.zip')
blob.download_to_filename('local-data.zip')
```

## 🛡️ Best Practices

### Security
```bash
# Use environment variables for secrets
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret

# Don't commit credentials
echo "*.env" >> .gitignore
echo "credentials.json" >> .gitignore
```

### Monitoring
```python
# Add to training script
import wandb
wandb.init(project="nasdaq-futures-trading")
wandb.log({"loss": loss, "reward": reward})

# Email notifications
import smtplib
def send_completion_email():
    # Send email when training completes
    pass
```

### Checkpointing
```python
# Automatic checkpointing (already included in cloud script)
# Saves every 5K-25K steps depending on platform
# Resume with: python train_cloud.py --resume-from checkpoint.zip
```

## 🎯 Recommended Workflows

### **Beginner Workflow**
1. Start with **Google Colab** (free)
2. Test with small dataset (1-2 months)
3. Verify everything works
4. Scale to **Colab Pro** or **Lambda Labs**

### **Production Workflow**
1. Develop on **Colab**
2. Full training on **AWS Spot** instances
3. Model serving on **AWS/Lambda** for inference
4. Monitoring with **MLflow/WandB**

### **Budget Workflow** (under $10/month)
1. **Colab** for development (free)
2. **Lambda Labs** for training (~$5-10)
3. Local inference or **AWS Lambda**

### **Enterprise Workflow**
1. **AWS/GCP** with enterprise accounts
2. **Multi-GPU** training for speed
3. **MLOps** pipelines for automation
4. **Production deployment** on cloud

## 🚨 Common Issues & Solutions

### GPU Out of Memory
```python
# Reduce batch size
python train_cloud.py --batch-size 16

# Use gradient accumulation
# Already handled in cloud script
```

### Connection Timeouts
```bash
# Use screen/tmux for long sessions
screen -S trading
python train_cloud.py
# Ctrl+A, D to detach
```

### Data Transfer Slow
```bash
# Use compression
tar -czf data.tar.gz data/
# Upload compressed, extract on cloud

# Use multi-part upload for large files
aws configure set default.s3.multipart_threshold 64MB
```

### CUDA/GPU Issues
```bash
# Check GPU availability
nvidia-smi

# Force CPU mode if GPU issues
python train_cloud.py --cpu-only

# Clear GPU memory if needed
python -c "import torch; torch.cuda.empty_cache()"
```

### Colab-Specific Issues

#### "AttributeError: 'CloudTrainer' object has no attribute 'logger'"
**Fixed in latest version** - Update your code:
```bash
# Re-download latest version
!git pull origin main
# Or re-clone if needed
```

#### Session Disconnects
```python
# Enable Colab Pro for longer sessions
# Use checkpointing (automatic in cloud script)
# Save models to Google Drive frequently
```

#### Runtime Crashes
```python
# Restart runtime and try again
# Runtime > Restart and run all
# Reduce batch size if memory issues persist
```

## 📞 Getting Help

### Support Channels
- **Colab**: Google Colab Community
- **AWS**: AWS Support (paid)
- **Lambda**: Excellent Discord community
- **Paperspace**: Good documentation + support

### Cost Monitoring
```bash
# AWS Cost Explorer
aws ce get-cost-and-usage

# GCP Billing API
gcloud billing accounts list

# Set up billing alerts for all platforms
```

## 🎉 Quick Start Commands

Choose your platform and run:

```bash
# Google Colab (paste in notebook)
!git clone [your-repo] && cd [repo-name] && pip install -r requirements.txt && python train_cloud.py

# AWS EC2
git clone [your-repo] && cd [repo-name] && pip install -r requirements.txt && python train_cloud.py

# Lambda Labs
git clone [your-repo] && cd [repo-name] && pip install -r requirements.txt && python train_cloud.py

# Any cloud with custom environment
python train_cloud.py --cloud-env [colab|aws|gcp|lambda]
```

## 💰 **Bottom Line Recommendations**

### **For Learning** → **Google Colab** (Free)
### **For Prototyping** → **Lambda Labs** ($3-5)  
### **For Production** → **AWS Spot** ($2-10)
### **For Enterprise** → **AWS/GCP Reserved** (Contact for pricing)

The cloud training script automatically detects your environment and optimizes accordingly!

## 📊 **Training Progress & Feedback**

### **What You'll See During Training**

#### **Phase 1: Data Preparation (Current)**
```
Loading data for colab environment
Limiting data to 3 months: 2010-06-06 to 2010-09-19
Resampled from 89655 1-min bars to 20201 5-min bars
Starting complete feature engineering pipeline
Calculated VWAP features for all timeframes
```

#### **Phase 2: TFT Training**
```
=== Cloud TFT Training ===
Epoch 1/75: [████████████████████] 100% 
  - train_loss: 0.523
  - val_loss: 0.445
  - quantile_loss: 0.234
  - learning_rate: 0.001

Epoch 2/75: [████████████████████] 100%
  - train_loss: 0.467
  - val_loss: 0.421
  - quantile_loss: 0.198
  
Best model saved with validation loss: 0.398
TFT model saved to: models/tft_cloud_colab.pt
```

#### **Phase 3: RL Training (Most Verbose)**
```
=== Cloud RL Training ===
Creating vectorized environment...
Using device: cuda

| rollout/            |          |
|    ep_len_mean      | 1.2e+03  |
|    ep_rew_mean      | 245      |
| time/               |          |
|    fps              | 1250     |
|    iterations       | 100      |
|    time_elapsed     | 79       |
|    total_timesteps  | 100000   |
| train/              |          |
|    approx_kl        | 0.012    |
|    clip_fraction    | 0.12     |
|    clip_range       | 0.2      |
|    entropy_loss     | -2.86    |
|    explained_variance| 0.76    |
|    learning_rate    | 0.0003   |
|    loss             | 0.089    |
|    policy_gradient_loss| -0.034|
|    value_loss       | 0.156    |

Saving model checkpoint: rl_checkpoints_colab/rl_model_colab_100000_steps.zip
Model evaluation: mean_reward = 387.45 (+/- 123.67)
```

#### **Phase 4: Final Evaluation**
```
=== Cloud Evaluation ===
Episode 1: Reward=421.23, Trades=15, Win Rate=66.7%
Episode 2: Reward=398.67, Trades=18, Win Rate=61.1%
...
Average Episode Reward: 405.89
```

### **Real-Time Monitoring Commands**

#### **In Google Colab (Additional Cell)**
```python
# Monitor GPU usage while training
!watch -n 5 nvidia-smi

# Check training logs in real-time
!tail -f logs/cloud_training_colab.log

# View MLflow dashboard
import mlflow
print(f"MLflow UI: http://localhost:5000")
```

#### **Monitor Training Files**
```python
# Check model saves
!ls -la models/

# View latest metrics
!cat logs/cloud_training_colab.log | tail -20

# Check memory usage
!free -h && df -h
```

### **Progress Indicators**

#### **Data Phase**: File loading progress, feature engineering steps
#### **TFT Phase**: Epoch progress bars, loss curves, validation metrics  
#### **RL Phase**: Timestep counters, reward plots, policy metrics
#### **Evaluation**: Episode results, final performance summary

### **Expected Timing (Colab)**
- **Data Preparation**: 2-3 minutes
- **TFT Training**: 15-25 minutes (75 epochs)
- **RL Training**: 45-90 minutes (750k timesteps)
- **Final Evaluation**: 2-3 minutes
- **Total**: ~60-120 minutes

### **Getting More Verbose Output**

Add to your training cell:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Run with extra verbosity
!python train_cloud.py --cloud-env colab --verbose
```