#!/usr/bin/env python3
"""
Setup script for local CPU testing on MacBook
Installs minimal dependencies needed to validate code before cloud training
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    
    print("✅ Python version OK")
    return True

def install_packages():
    """Install minimal packages for local testing"""
    
    # Essential packages for basic functionality
    essential_packages = [
        "torch>=1.13.0",
        "pandas>=1.5.0", 
        "numpy>=1.21.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "tqdm>=4.64.0",
        "python-dotenv>=0.21.0"
    ]
    
    # Optional packages (install if possible, skip if failed)
    optional_packages = [
        "pytorch-forecasting>=0.10.0",
        "stable-baselines3>=1.7.0", 
        "gymnasium>=0.26.0",
        "mlflow>=2.1.0"
    ]
    
    print("📦 Installing essential packages...")
    for package in essential_packages:
        success = run_command(f"pip install {package}", f"Installing {package}")
        if not success:
            print(f"⚠️  Failed to install {package} - this may cause issues")
    
    print("\n📦 Installing optional packages (failures are OK)...")
    for package in optional_packages:
        success = run_command(f"pip install {package}", f"Installing {package}")
        if not success:
            print(f"⏭️  Skipped {package} - will install in cloud")

def create_directories():
    """Create required directories"""
    dirs_to_create = ['data', 'models', 'logs', 'src/data', 'src/models', 'src/environment']
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

def main():
    """Main setup function"""
    print("🍎 MacBook Local Testing Setup")
    print("=" * 50)
    print("This script installs minimal dependencies for local CPU testing.")
    print("Full dependencies will be installed in the cloud environment.\n")
    
    # Check Python version
    if not check_python_version():
        print("❌ Please upgrade Python to 3.8+ and try again")
        return 1
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install packages
    print("\n📦 Installing packages...")
    install_packages()
    
    print("\n🎉 Local testing setup complete!")
    print("\n📋 Next steps:")
    print("   1. Run: python test_local_cpu.py")
    print("   2. Fix any issues found")
    print("   3. Upload data to cloud and train")
    
    return 0

if __name__ == "__main__":
    exit(main()) 