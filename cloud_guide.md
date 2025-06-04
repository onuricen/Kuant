# Cloud Training Guide - Remote Training Providers

This guide covers all the best options for training your Nasdaq Futures Hybrid ML Trading System on cloud platforms, from free to enterprise-grade solutions.

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
screen -S training
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