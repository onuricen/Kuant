#!/usr/bin/env python3
"""
Cloud-optimized training script for Nasdaq Futures Hybrid ML Trading System
Automatically detects cloud environment and optimizes accordingly
"""

import logging
import argparse
import sys
import os
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import torch
import psutil
import platform
import joblib
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import config
from src.data.data_loader import NasdaqFuturesDataLoader
from src.data.feature_engineering import FeatureEngineer
from src.models.tft_model import TradingTFT
from src.environment.trading_environment import HybridTradingEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import mlflow

# Cloud environment detection
def detect_cloud_environment():
    """Detect which cloud environment we're running in"""
    # Check for Google Colab
    if 'COLAB_GPU' in os.environ or 'google.colab' in sys.modules:
        return 'colab'
    
    # Check for Kaggle
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return 'kaggle'
    
    # Check for Runpod (multiple detection methods)
    if ('RUNPOD_POD_ID' in os.environ or 
        'RUNPOD_CPU_COUNT' in os.environ or
        'RUNPOD_DATACENTER' in os.environ or
        'RUNPOD_MACHINE_ID' in os.environ):
        return 'runpod'
    
    # Additional Runpod checks
    if (os.path.exists('/etc/runpod-release') or 
        'runpod' in platform.node().lower() or
        any('runpod' in str(p).lower() for p in Path('/').glob('*runpod*') if p.exists())):
        return 'runpod'
    
    # Check for AWS EC2
    try:
        import requests
        response = requests.get('http://169.254.169.254/latest/meta-data/', timeout=1)
        if response.status_code == 200:
            return 'aws'
    except:
        pass
    
    # Check for Google Cloud
    try:
        import requests
        response = requests.get('http://metadata.google.internal/computeMetadata/v1/', 
                              headers={'Metadata-Flavor': 'Google'}, timeout=1)
        if response.status_code == 200:
            return 'gcp'
    except:
        pass
    
    # Check for Paperspace
    if 'PS_API_KEY' in os.environ or 'PAPERSPACE_NOTEBOOK_REPO_ID' in os.environ:
        return 'paperspace'
    
    return 'unknown'

# Optimized Feature Engineering for Cloud
class FastFeatureEngineer:
    """Simplified feature engineering for cloud training"""
    
    def __init__(self):
        from src.data.feature_engineering import FeatureEngineer
        self.fe = FeatureEngineer()
        self.logger = logging.getLogger(__name__)
    
    def create_fast_feature_set(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feature set optimized for speed (skip volume profile)"""
        self.logger.info("Starting FAST feature engineering pipeline (cloud optimized)")
        
        # Apply core features only (skip expensive volume profile)
        df = self.fe.calculate_vwap_features(df)
        df = self.fe.calculate_price_action_features(df)
        df = self.fe.calculate_session_features(df)
        df = self.fe.calculate_target_labels(df)
        df = self.fe.normalize_features(df)
        
        # Remove rows with insufficient data
        df = df.dropna()
        
        # Fix categorical columns for TFT compatibility
        self.logger.info("Checking column data types before conversion...")
        categorical_columns = ['session_type', 'day_of_week', 'hour', 'is_asia_session']
        for col in categorical_columns:
            if col in df.columns:
                self.logger.info(f"Column {col}: dtype={df[col].dtype}, sample values={df[col].unique()[:5]}")
                df[col] = df[col].astype(str)
                self.logger.info(f"After conversion - {col}: dtype={df[col].dtype}, sample values={df[col].unique()[:5]}")
            else:
                self.logger.warning(f"Column {col} not found in dataframe!")
        
        # Convert boolean columns to strings for TFT
        boolean_columns = ['is_weekend']
        for col in boolean_columns:
            if col in df.columns:
                self.logger.info(f"Boolean column {col}: dtype={df[col].dtype}, sample values={df[col].unique()[:5]}")
                df[col] = df[col].astype(str)
                self.logger.info(f"After conversion - {col}: dtype={df[col].dtype}, sample values={df[col].unique()[:5]}")
        
        self.logger.info(f"All categorical columns final check:")
        for col in categorical_columns + boolean_columns:
            if col in df.columns:
                self.logger.info(f"{col}: {df[col].dtype}")
        
        self.logger.info(f"FAST feature engineering finished. Final dataset shape: {df.shape}")
        
        return df

# Cloud-specific configurations
class CloudConfig:
    """Cloud environment specific configurations"""
    
    @staticmethod
    def get_config(cloud_env):
        configs = {
            'colab': {
                'batch_size': 32,
                'max_epochs': 75,
                'total_timesteps': 750_000,
                'save_freq': 10000,
                'eval_freq': 5000,
                'checkpointing': True,
                'data_limit_months': 3
            },
            'kaggle': {
                'batch_size': 32,
                'max_epochs': 60,
                'total_timesteps': 600_000,
                'save_freq': 5000,
                'eval_freq': 2500,
                'checkpointing': True,
                'data_limit_months': 2  # Kaggle has time limits
            },
            'runpod': {
                'batch_size': 48,  # Optimized for A5000 24GB VRAM
                'max_epochs': 100,
                'total_timesteps': 1_000_000,
                'save_freq': 15000,
                'eval_freq': 7500,
                'checkpointing': True,
                'data_limit_months': None,
                'mixed_precision': True,  # A5000 supports this well
                'gradient_accumulation': 2,  # For effective batch size of 96
            },
            'aws': {
                'batch_size': 64,
                'max_epochs': 100,
                'total_timesteps': 1_000_000,
                'save_freq': 25000,
                'eval_freq': 10000,
                'checkpointing': True,
                'data_limit_months': None
            },
            'gcp': {
                'batch_size': 64,
                'max_epochs': 100,
                'total_timesteps': 1_000_000,
                'save_freq': 25000,
                'eval_freq': 10000,
                'checkpointing': True,
                'data_limit_months': None
            },
            'paperspace': {
                'batch_size': 48,
                'max_epochs': 80,
                'total_timesteps': 800_000,
                'save_freq': 15000,
                'eval_freq': 7500,
                'checkpointing': True,
                'data_limit_months': 4
            }
        }
        
        return configs.get(cloud_env, configs['aws'])  # Default to AWS config

# Setup logging
def setup_cloud_logging(cloud_env):
    log_level = logging.INFO
    if cloud_env in ['colab', 'kaggle']:
        # More verbose logging for interactive environments
        log_level = logging.DEBUG
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.data.LOGS_DIR / f'cloud_training_{cloud_env}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class CloudTrainer:
    """Cloud-optimized trainer with environment-specific adaptations"""
    
    def __init__(self, cloud_env='unknown'):
        self.cloud_env = cloud_env
        self.use_mixed_precision = False
        self.aggressive_checkpointing = False
        
        # Setup cloud configuration
        self.cloud_config = CloudConfig.get_config(cloud_env)
        
        # Initialize logger first (before any other operations that might need it)
        self.logger = setup_cloud_logging(cloud_env)
        self.logger.info(f"Initializing CloudTrainer for {cloud_env}")
        
        # Verify and fix pytorch-forecasting compatibility early
        self.logger.info("🔍 Verifying PyTorch Forecasting compatibility...")
        if not verify_and_fix_pytorch_forecasting():
            self.logger.warning("⚠️ PyTorch Forecasting compatibility issues detected but could not be auto-fixed")
            self.logger.info("💡 You may need to manually run: pip install pytorch-forecasting==1.0.0")
        
        # Initialize components
        self.data_loader = NasdaqFuturesDataLoader()
        self.feature_engineer = FeatureEngineer()
        self.tft_model = None
        self.rl_model = None
        self.processed_data = None
        
        # Setup cloud-specific configurations
        self.setup_cloud_optimizations()
        
        # MLflow setup
        mlflow.set_experiment(f"nasdaq_futures_cloud_{cloud_env}")
        
        self.logger.info(f"CloudTrainer initialization completed for {cloud_env}")
    
    def setup_cloud_optimizations(self):
        """Setup optimizations based on cloud environment"""
        # Apply cloud-specific config overrides
        config.tft.BATCH_SIZE = self.cloud_config['batch_size']
        config.tft.MAX_EPOCHS = self.cloud_config['max_epochs']
        config.rl.TOTAL_TIMESTEPS = self.cloud_config['total_timesteps']
        
        # GPU optimizations
        if torch.cuda.is_available():
            try:
                # Enable optimizations for cloud GPUs
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # Memory management
                torch.cuda.empty_cache()
                
                gpu_props = torch.cuda.get_device_properties(0)
                self.logger.info(f"Cloud GPU: {gpu_props.name}")
                self.logger.info(f"VRAM: {gpu_props.total_memory / 1024**3:.1f} GB")
                
                # Enable mixed precision for newer GPUs
                if gpu_props.major >= 7:  # Tensor cores available
                    self.use_mixed_precision = True
                    self.logger.info("Enabled mixed precision training")
            except Exception as e:
                self.logger.warning(f"GPU setup encountered an issue: {str(e)}")
                self.logger.info("Continuing with CPU or default GPU settings")
        else:
            self.logger.info("CUDA not available, using CPU")
        
        # Environment-specific setup
        if self.cloud_env == 'colab':
            self.setup_colab()
        elif self.cloud_env == 'kaggle':
            self.setup_kaggle()
        elif self.cloud_env == 'runpod':
            self.setup_runpod()
        elif self.cloud_env in ['aws', 'gcp']:
            self.setup_enterprise_cloud()
    
    def setup_colab(self):
        """Colab-specific setup"""
        self.logger.info("Setting up for Google Colab")
        # Mount Google Drive if available
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=True)
            self.drive_path = Path('/content/drive/MyDrive/nasdaq_trading')
            self.drive_path.mkdir(exist_ok=True)
            self.logger.info("Google Drive mounted")
        except:
            self.logger.warning("Could not mount Google Drive")
    
    def setup_kaggle(self):
        """Kaggle-specific setup"""
        self.logger.info("Setting up for Kaggle")
        # Kaggle has 9-hour time limits, enable aggressive checkpointing
        self.aggressive_checkpointing = True
        
        # Use Kaggle datasets path
        kaggle_data_path = Path('/kaggle/input')
        if kaggle_data_path.exists():
            self.logger.info(f"Kaggle input data found at {kaggle_data_path}")
    
    def setup_runpod(self):
        """Runpod-specific setup"""
        self.logger.info("Setting up for Runpod")
        
        # Runpod-specific optimizations
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            
            # A5000-specific optimizations
            if 'a5000' in gpu_name:
                self.logger.info("Detected RTX A5000 - applying optimizations")
                # Enable optimizations for A5000's 24GB VRAM
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Memory optimization for A5000
                torch.cuda.set_per_process_memory_fraction(0.9)
                
            # General Runpod GPU optimizations
            torch.backends.cudnn.benchmark = True
            
        # Check for persistent storage
        workspace_path = Path('/workspace')
        if workspace_path.exists():
            self.logger.info("Found /workspace directory - using for persistent storage")
            # Update config paths to use workspace
            config.data.PROCESSED_DATA_DIR = workspace_path / 'data' / 'processed'
            config.data.MODEL_DIR = workspace_path / 'models'
            config.data.LOGS_DIR = workspace_path / 'logs'
            
            # Create directories if they don't exist
            config.data.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            config.data.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            config.data.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Log Runpod environment info
        runpod_info = {
            'pod_id': os.environ.get('RUNPOD_POD_ID', 'unknown'),
            'datacenter': os.environ.get('RUNPOD_DATACENTER', 'unknown'),
            'machine_id': os.environ.get('RUNPOD_MACHINE_ID', 'unknown'),
        }
        self.logger.info(f"Runpod environment: {runpod_info}")
        
        # Enable aggressive checkpointing for Runpod's spot instances
        if 'spot' in os.environ.get('RUNPOD_POD_TYPE', '').lower():
            self.aggressive_checkpointing = True
            self.logger.info("Detected spot instance - enabling aggressive checkpointing")
    
    def setup_enterprise_cloud(self):
        """Setup for AWS/GCP enterprise environments"""
        self.logger.info(f"Setting up for enterprise cloud: {self.cloud_env}")
        # Enable distributed training if multiple GPUs available
        if torch.cuda.device_count() > 1:
            self.logger.info(f"Multiple GPUs detected: {torch.cuda.device_count()}")
            # Could enable data parallel training here
    
    def load_and_prepare_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load and prepare data with cloud optimizations and smart caching"""
        self.logger.info(f"Loading data for {self.cloud_env} environment with smart caching")
        
        # Import smart data loader
        from src.data.data_cache import SmartDataLoader
        
        # Create smart data loader with caching
        smart_loader = SmartDataLoader(
            base_data_loader=self.data_loader,
            cache_dir=config.data.PROCESSED_DATA_DIR / "cache"
        )
        
        # Prepare feature parameters for cache key
        feature_params = {
            'cloud_env': self.cloud_env,
            'batch_size': config.tft.BATCH_SIZE,
            'data_limit_months': self.cloud_config.get('data_limit_months'),
            'fast_features': True  # Using FastFeatureEngineer
        }
        
        # Apply data limits for time-constrained environments
        if not start_date or not end_date:
            available_dates = self.data_loader.get_available_dates()
            
            # Limit data size for time-constrained environments
            if self.cloud_config.get('data_limit_months'):
                limit = self.cloud_config['data_limit_months']
                dates_to_use = available_dates[:limit * 30]  # Approximate months
                if dates_to_use:
                    start_date = dates_to_use[0]
                    end_date = dates_to_use[-1]
                    self.logger.info(f"Limiting data to {limit} months: {start_date} to {end_date}")
        
        # Load with smart caching
        cache_result = smart_loader.load_and_process_with_cache(
            start_date=start_date,
            end_date=end_date,
            feature_params=feature_params,
            force_refresh=getattr(self, 'force_refresh', False)  # Use force_refresh flag if set
        )
        
        if 'processed_data' in cache_result:
            self.processed_data = cache_result['processed_data']
            
            # Store cache key for TFT dataset caching
            if 'cache_key' in cache_result:
                self.processed_data_cache_key = cache_result['cache_key']
            
            # Log cache efficiency
            if 'feature_metadata' in cache_result:
                meta = cache_result['feature_metadata']
                self.logger.info(f"✅ Using processed data: {meta['total_features']} features, {meta['data_range']['total_rows']} rows")
                
                if 'processing_time' in meta:
                    self.logger.info(f"📅 Data processed at: {meta['processing_time']}")
                    
        else:
            # Fallback to original method if caching fails
            self.logger.warning("Smart caching failed, using fallback method")
            
            if start_date and end_date:
                raw_data = self.data_loader.load_data_range(start_date, end_date)
            else:
                raw_data = self.data_loader.load_all_data()
            
            # Resample and engineer features
            resampled_data = self.data_loader.resample_to_5min(raw_data)
            
            # Use fast feature engineering for cloud environments
            fast_fe = FastFeatureEngineer()
            self.processed_data = fast_fe.create_fast_feature_set(resampled_data)
        
        # Cloud storage optimization
        self.save_cloud_data()
        
        return self.processed_data
    
    def save_cloud_data(self):
        """Save processed data to cloud storage"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_nasdaq_cloud_{self.cloud_env}_{timestamp}"
        
        # Save locally
        self.data_loader.save_processed_data(self.processed_data, filename)
        
        # Save to cloud storage if available
        if self.cloud_env == 'colab' and hasattr(self, 'drive_path'):
            cloud_path = self.drive_path / f"{filename}.parquet"
            self.processed_data.to_parquet(cloud_path)
            self.logger.info(f"Data saved to Google Drive: {cloud_path}")
    
    def train_tft_cloud(self, data: pd.DataFrame) -> TradingTFT:
        """Train TFT with cloud optimizations and dataset caching"""
        self.logger.info("Starting cloud-optimized TFT training with caching")
        
        # CRITICAL: Validate and force categorical columns before TFT training
        self.logger.info("=== PRE-TFT CATEGORICAL VALIDATION ===")
        expected_categoricals = ['session_type', 'day_of_week', 'hour', 'is_asia_session']
        
        for col in expected_categoricals:
            if col in data.columns:
                self.logger.info(f"PRE-TFT {col}: dtype={data[col].dtype}, unique_values={sorted(data[col].unique())}")
                # Force to string and categorical
                data[col] = data[col].astype(str).astype('category')
                self.logger.info(f"POST-FORCE {col}: dtype={data[col].dtype}, unique_values={sorted(data[col].unique())}")
            else:
                self.logger.error(f"MISSING COLUMN: {col} not found in dataframe!")
        
        # Also check boolean columns
        boolean_cols = ['is_weekend']
        for col in boolean_cols:
            if col in data.columns:
                self.logger.info(f"PRE-TFT {col}: dtype={data[col].dtype}, unique_values={sorted(data[col].unique())}")
                data[col] = data[col].astype(str).astype('category')
                self.logger.info(f"POST-FORCE {col}: dtype={data[col].dtype}, unique_values={sorted(data[col].unique())}")
        
        self.logger.info("=== TFT CATEGORICAL VALIDATION COMPLETE ===")
        
        with mlflow.start_run(run_name=f"TFT_Cloud_{self.cloud_env}", nested=True):
            # Log cloud configuration
            mlflow.log_params({
                "cloud_environment": self.cloud_env,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "batch_size": config.tft.BATCH_SIZE,
                "max_epochs": config.tft.MAX_EPOCHS
            })
            
            # Check for cached TFT datasets
            tft_cache_key = None
            cached_datasets = None
            
            # Try to load cached TFT datasets if we have a cache key from data loading
            if hasattr(self, 'processed_data_cache_key'):
                from src.data.data_cache import DataCache
                cache = DataCache(config.data.PROCESSED_DATA_DIR / "cache")
                
                try:
                    cached_data = cache.load_processed_data(self.processed_data_cache_key)
                    if 'tft_datasets' in cached_data:
                        cached_datasets = cached_data['tft_datasets']
                        self.logger.info("📦 Found cached TFT datasets")
                except Exception as e:
                    self.logger.warning(f"Could not load cached TFT datasets: {e}")
            
            # Initialize and train TFT
            self.tft_model = TradingTFT()
            
            if cached_datasets:
                # Use cached datasets
                self.logger.info("✅ Using cached TFT datasets")
                training_dataset, validation_dataset = cached_datasets
            else:
                # Create new datasets and cache them
                self.logger.info("🔄 Creating new TFT datasets")
                training_dataset, validation_dataset = self.tft_model.prepare_data(data)
                
                # Cache the datasets for future use
                if hasattr(self, 'processed_data_cache_key'):
                    try:
                        cache = DataCache(config.data.PROCESSED_DATA_DIR / "cache")
                        cache_paths = cache._get_cache_paths(self.processed_data_cache_key)
                        joblib.dump((training_dataset, validation_dataset), cache_paths['tft_datasets'])
                        self.logger.info("💾 Cached TFT datasets for future use")
                    except Exception as e:
                        self.logger.warning(f"Could not cache TFT datasets: {e}")
            
            model = self.tft_model.create_model(training_dataset)
            
            # Train with cloud optimizations
            training_metrics = self.tft_model.train(training_dataset, validation_dataset)
            mlflow.log_metrics(training_metrics)
            
            # Save model
            model_path = config.data.MODEL_DIR / f"tft_cloud_{self.cloud_env}.pt"
            self.tft_model.save_model(model_path)
            mlflow.log_artifact(str(model_path))
            
            return self.tft_model
    
    def train_rl_cloud(self, data: pd.DataFrame, tft_model: TradingTFT) -> PPO:
        """Train RL agent with cloud optimizations"""
        self.logger.info("Starting cloud-optimized RL training")
        
        with mlflow.start_run(run_name=f"RL_Cloud_{self.cloud_env}", nested=True):
            # Log RL parameters
            mlflow.log_params({
                "cloud_environment": self.cloud_env,
                "total_timesteps": config.rl.TOTAL_TIMESTEPS,
                "checkpointing": self.cloud_config['checkpointing']
            })
            
            # Create environment
            env = HybridTradingEnvironment(
                data=data,
                tft_model=tft_model,
                initial_balance=config.trading.ACCOUNT_SIZE
            )
            
            # Create RL model
            self.rl_model = PPO(
                config.rl.POLICY_TYPE,
                env,
                learning_rate=config.rl.LEARNING_RATE,
                n_steps=config.rl.N_STEPS,
                batch_size=config.rl.BATCH_SIZE,
                n_epochs=config.rl.N_EPOCHS,
                gamma=config.rl.GAMMA,
                gae_lambda=config.rl.GAE_LAMBDA,
                clip_range=config.rl.CLIP_RANGE,
                ent_coef=config.rl.ENT_COEF,
                vf_coef=config.rl.VF_COEF,
                max_grad_norm=config.rl.MAX_GRAD_NORM,
                verbose=1,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Setup callbacks
            callbacks = []
            
            # Checkpointing callback
            if self.cloud_config['checkpointing']:
                checkpoint_callback = CheckpointCallback(
                    save_freq=self.cloud_config['save_freq'],
                    save_path=str(config.data.MODEL_DIR / f"rl_checkpoints_{self.cloud_env}"),
                    name_prefix=f"rl_model_{self.cloud_env}"
                )
                callbacks.append(checkpoint_callback)
            
            # Evaluation callback
            eval_callback = EvalCallback(
                env,
                best_model_save_path=str(config.data.MODEL_DIR / f"rl_best_{self.cloud_env}"),
                log_path=str(config.data.LOGS_DIR / f"rl_eval_{self.cloud_env}"),
                eval_freq=self.cloud_config['eval_freq'],
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
            
            # Train the model
            self.rl_model.learn(
                total_timesteps=config.rl.TOTAL_TIMESTEPS,
                callback=callbacks,
                progress_bar=True
            )
            
            # Save final model
            model_path = config.data.MODEL_DIR / f"rl_final_{self.cloud_env}.zip"
            self.rl_model.save(str(model_path))
            mlflow.log_artifact(str(model_path))
            
            return self.rl_model
    
    def run_cloud_training(self, start_date: str = None, end_date: str = None):
        """Run complete cloud training pipeline"""
        self.logger.info(f"Starting cloud training pipeline on {self.cloud_env}")
        
        with mlflow.start_run(run_name=f"Cloud_Training_{self.cloud_env}"):
            # Log environment info
            mlflow.log_params({
                "cloud_provider": self.cloud_env,
                "python_version": platform.python_version(),
                "pytorch_version": torch.__version__,
                "system_ram_gb": psutil.virtual_memory().total / 1024**3
            })
            
            try:
                # Data preparation
                self.logger.info("=== Cloud Data Preparation ===")
                data = self.load_and_prepare_data(start_date, end_date)
                
                # TFT training
                self.logger.info("=== Cloud TFT Training ===")
                tft_model = self.train_tft_cloud(data)
                
                # RL training
                self.logger.info("=== Cloud RL Training ===")
                rl_model = self.train_rl_cloud(data, tft_model)
                
                # Final evaluation
                self.logger.info("=== Cloud Evaluation ===")
                env = HybridTradingEnvironment(
                    data=data,
                    tft_model=tft_model,
                    initial_balance=config.trading.ACCOUNT_SIZE
                )
                
                # Quick evaluation
                total_rewards = []
                for episode in range(10):
                    obs, info = env.reset()
                    episode_reward = 0
                    done = False
                    
                    while not done:
                        action, _ = rl_model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env.step(action)
                        episode_reward += reward
                        done = terminated or truncated
                    
                    total_rewards.append(episode_reward)
                
                final_metrics = {
                    "avg_episode_reward": np.mean(total_rewards),
                    "cloud_training_successful": 1
                }
                
                mlflow.log_metrics(final_metrics)
                
                self.logger.info("=== Cloud Training Completed Successfully ===")
                self.logger.info(f"Average Episode Reward: {final_metrics['avg_episode_reward']:.2f}")
                
                return {
                    "tft_model": tft_model,
                    "rl_model": rl_model,
                    "final_metrics": final_metrics
                }
                
            except Exception as e:
                self.logger.error(f"Cloud training failed: {str(e)}")
                mlflow.log_metrics({"cloud_training_successful": 0})
                raise

def verify_and_fix_pytorch_forecasting():
    """Verify pytorch-forecasting compatibility and auto-fix if needed"""
    logger = logging.getLogger(__name__)
    
    try:
        # Test if TFT is properly recognized as LightningModule
        import pytorch_lightning as pl
        from pytorch_forecasting import TemporalFusionTransformer
        
        is_lightning_module = issubclass(TemporalFusionTransformer, pl.LightningModule)
        
        if is_lightning_module:
            logger.info("✅ PyTorch Forecasting compatibility verified")
            return True
        
        logger.warning("❌ PyTorch Forecasting compatibility issue detected")
        logger.info("🔧 Attempting automatic fix...")
        
        # Apply the fix
        return apply_pytorch_forecasting_fix()
        
    except Exception as e:
        logger.error(f"❌ Error checking PyTorch Forecasting compatibility: {e}")
        logger.info("🔧 Attempting automatic fix anyway...")
        return apply_pytorch_forecasting_fix()

def apply_pytorch_forecasting_fix():
    """Apply the pytorch-forecasting import fix"""
    logger = logging.getLogger(__name__)
    
    try:
        import pytorch_forecasting
        from pathlib import Path
        
        # Find pytorch-forecasting installation path
        pf_path = Path(pytorch_forecasting.__file__).parent
        logger.info(f"📍 Found pytorch-forecasting at: {pf_path}")
        
        # Files that need fixing
        files_to_fix = [
            pf_path / "models" / "base_model.py",
            pf_path / "utils" / "_utils.py", 
            pf_path / "models" / "temporal_fusion_transformer" / "tuning.py"
        ]
        
        fixed_count = 0
        
        for file_path in files_to_fix:
            if file_path.exists():
                try:
                    # Read file
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Replace problematic imports
                    original_content = content
                    content = content.replace('lightning.pytorch', 'pytorch_lightning')
                    
                    if content != original_content:
                        # Create backup
                        backup_path = str(file_path) + '.backup'
                        with open(backup_path, 'w') as f:
                            f.write(original_content)
                        
                        # Write fixed content
                        with open(file_path, 'w') as f:
                            f.write(content)
                        
                        logger.info(f"✅ Fixed {file_path.name}")
                        fixed_count += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️ Could not fix {file_path}: {e}")
            else:
                logger.warning(f"⚠️ File not found: {file_path}")
        
        logger.info(f"🎉 Fix completed! Modified {fixed_count} files.")
        
        # Test the fix
        try:
            # Force reload the modules
            import importlib
            import pytorch_forecasting.models.base_model
            import pytorch_forecasting.models.temporal_fusion_transformer
            importlib.reload(pytorch_forecasting.models.base_model)
            importlib.reload(pytorch_forecasting.models.temporal_fusion_transformer)
            
            # Test again
            import pytorch_lightning as pl
            from pytorch_forecasting import TemporalFusionTransformer
            
            is_lightning_module = issubclass(TemporalFusionTransformer, pl.LightningModule)
            
            if is_lightning_module:
                logger.info("✅ Fix successful! TFT is now properly recognized as LightningModule")
                return True
            else:
                logger.error("❌ Fix may not have worked. TFT still not recognized as LightningModule")
                logger.info("💡 Try restarting the Python process or use: pip install pytorch-forecasting==1.0.0")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing fix: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to apply pytorch-forecasting fix: {e}")
        return False

def main():
    """Main cloud training script"""
    parser = argparse.ArgumentParser(description="Cloud Training for Hybrid Trading System")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--cloud-env", type=str, help="Force specific cloud environment")
    parser.add_argument("--force-refresh", action="store_true", help="Force fresh data processing (ignore cache)")
    parser.add_argument("--cache-info", action="store_true", help="Show cache information and exit")
    parser.add_argument("--clear-cache", action="store_true", help="Clear all cache and exit")
    parser.add_argument("--verify-pytorch-forecasting", action="store_true", help="Verify and fix PyTorch Forecasting compatibility")
    
    args = parser.parse_args()
    
    # PyTorch Forecasting verification command
    if args.verify_pytorch_forecasting:
        print("🔍 Verifying PyTorch Forecasting compatibility...")
        if verify_and_fix_pytorch_forecasting():
            print("✅ PyTorch Forecasting is working correctly!")
        else:
            print("❌ PyTorch Forecasting compatibility issues detected")
            print("💡 Try manually running: pip install pytorch-forecasting==1.0.0")
        return
    
    # Cache management commands
    if args.cache_info or args.clear_cache:
        from src.data.data_cache import DataCache
        cache = DataCache()
        
        if args.cache_info:
            info = cache.get_cache_info()
            print("\n📦 Cache Information:")
            print(f"   Total cached datasets: {info['total_cached_datasets']}")
            print(f"   Total cache size: {info['total_size_mb']} MB")
            
            if info['datasets']:
                print("\n   Cached datasets:")
                for cache_key, details in info['datasets'].items():
                    print(f"     🔑 {cache_key[:8]}...")
                    print(f"        📅 Created: {details['created_at']}")
                    print(f"        📊 Shape: {details['data_shape']}")
                    print(f"        📈 Date range: {details['date_range']}")
                    print(f"        💾 Size: {details['size_mb']} MB")
                    print()
            else:
                print("   No cached data found")
            return
        
        if args.clear_cache:
            print("🗑️  Clearing all cache...")
            cache.clear_cache()
            print("✅ Cache cleared successfully")
            return
    
    # Detect or use specified cloud environment
    cloud_env = args.cloud_env or detect_cloud_environment()
    
    print(f"🌩️  Detected cloud environment: {cloud_env.upper()}")
    print(f"🎯  Training configuration optimized for {cloud_env}")
    
    if args.force_refresh:
        print("🔄 Force refresh enabled - will ignore cache")
    
    # Initialize cloud trainer
    trainer = CloudTrainer(cloud_env)
    
    # Set force refresh flag
    if args.force_refresh:
        trainer.force_refresh = True
    
    try:
        # Run cloud training
        results = trainer.run_cloud_training(args.start_date, args.end_date)
        
        print("\n🎉 Cloud training completed successfully!")
        print(f"📊 Average Episode Reward: {results['final_metrics']['avg_episode_reward']:.2f}")
        print("\n📁 Models saved:")
        print(f"   TFT Model: models/tft_cloud_{cloud_env}.pt")
        print(f"   RL Model: models/rl_final_{cloud_env}.zip")
        
        # Show cache info after training
        from src.data.data_cache import DataCache
        cache = DataCache()
        info = cache.get_cache_info()
        print(f"\n💾 Cache: {info['total_cached_datasets']} datasets, {info['total_size_mb']:.1f} MB")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Cloud training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 