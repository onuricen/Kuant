#!/usr/bin/env python3
"""
Mobile GPU optimized training script for Nasdaq Futures Hybrid ML Trading System
Optimized for GTX 1060 Ti mobile and similar GPUs with limited VRAM
"""

import logging
import argparse
import sys
import os
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import torch
import psutil
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import get_mobile_gpu_config
from src.data.data_loader import NasdaqFuturesDataLoader
from src.data.feature_engineering import FeatureEngineer
from src.models.tft_model import TradingTFT
from src.environment.trading_environment import HybridTradingEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import mlflow

# Use mobile GPU config
config = get_mobile_gpu_config()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.data.LOGS_DIR / 'mobile_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MobileGPUTrainer:
    """
    Mobile GPU optimized trainer with memory management
    """
    
    def __init__(self):
        self.data_loader = NasdaqFuturesDataLoader()
        self.feature_engineer = FeatureEngineer()
        self.tft_model = None
        self.rl_model = None
        self.processed_data = None
        
        # GPU memory management
        self.setup_gpu_memory()
        
        # MLflow setup
        mlflow.set_experiment("nasdaq_futures_mobile_gpu")
        
    def setup_gpu_memory(self):
        """Setup GPU memory management for mobile GPUs"""
        if torch.cuda.is_available():
            # Enable memory optimization
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Set memory fraction (use 90% of available VRAM)
            if hasattr(torch.cuda, 'set_memory_fraction'):
                torch.cuda.set_memory_fraction(0.9)
            
            gpu_props = torch.cuda.get_device_properties(0)
            logger.info(f"GPU: {gpu_props.name}")
            logger.info(f"Total VRAM: {gpu_props.total_memory / 1024**3:.1f} GB")
            logger.info(f"Available VRAM: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")
        else:
            logger.warning("CUDA not available. Training will use CPU (much slower)")
    
    def monitor_memory(self, step_name: str):
        """Monitor memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"{step_name} - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        # System RAM
        ram_percent = psutil.virtual_memory().percent
        logger.info(f"{step_name} - System RAM: {ram_percent:.1f}% used")
    
    def clear_memory(self):
        """Clear memory caches"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def load_and_prepare_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load and prepare data with memory optimization"""
        logger.info("Loading and preparing data for mobile GPU...")
        self.monitor_memory("Before data loading")
        
        # Load raw data
        if start_date and end_date:
            raw_data = self.data_loader.load_data_range(start_date, end_date)
        else:
            # Limit data size for mobile GPU
            available_dates = self.data_loader.get_available_dates()
            if len(available_dates) > 60:  # Limit to ~2 months for mobile GPU
                start_date = available_dates[0]
                end_date = available_dates[59]
                logger.info(f"Limiting dataset to {start_date} - {end_date} for mobile GPU")
                raw_data = self.data_loader.load_data_range(start_date, end_date)
            else:
                raw_data = self.data_loader.load_all_data()
        
        # Resample to 5-minute bars
        resampled_data = self.data_loader.resample_to_5min(raw_data)
        
        # Apply feature engineering
        logger.info("Applying feature engineering...")
        self.processed_data = self.feature_engineer.create_complete_feature_set(resampled_data)
        
        # Memory optimization: convert to float32
        float_cols = self.processed_data.select_dtypes(include=[np.float64]).columns
        self.processed_data[float_cols] = self.processed_data[float_cols].astype(np.float32)
        
        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_nasdaq_mobile_{timestamp}"
        self.data_loader.save_processed_data(self.processed_data, filename)
        
        # Log data summary
        summary = self.data_loader.get_data_summary(self.processed_data)
        logger.info(f"Data preparation complete. Shape: {self.processed_data.shape}")
        logger.info(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        
        self.monitor_memory("After data preparation")
        return self.processed_data
    
    def train_tft_model_mobile(self, data: pd.DataFrame, validation_split: float = 0.2) -> TradingTFT:
        """Train TFT model with mobile GPU optimizations"""
        logger.info("Starting mobile GPU optimized TFT training...")
        
        with mlflow.start_run(run_name="TFT_Mobile_Training", nested=True):
            # Log mobile configuration
            mlflow.log_params({
                "mobile_gpu_mode": True,
                "tft_hidden_size": config.tft.HIDDEN_SIZE,
                "tft_attention_heads": config.tft.ATTENTION_HEAD_SIZE,
                "tft_lookback_periods": config.tft.LOOKBACK_PERIODS,
                "tft_batch_size": config.tft.BATCH_SIZE,
                "tft_max_epochs": config.tft.MAX_EPOCHS
            })
            
            self.clear_memory()
            self.monitor_memory("Before TFT training")
            
            # Initialize TFT model
            self.tft_model = TradingTFT(config.tft)
            
            # Prepare data for TFT
            training_dataset, validation_dataset = self.tft_model.prepare_data(data)
            
            # Create and train model
            model = self.tft_model.create_model(training_dataset)
            
            # Training with gradient accumulation for small batches
            training_metrics = self.tft_model.train(training_dataset, validation_dataset)
            
            # Log training metrics
            mlflow.log_metrics(training_metrics)
            
            # Clear memory after training
            self.clear_memory()
            self.monitor_memory("After TFT training")
            
            # Save model
            model_path = config.data.MODEL_DIR / "tft_mobile_model.pt"
            self.tft_model.save_model(model_path)
            mlflow.log_artifact(str(model_path))
            
            logger.info(f"Mobile TFT training completed. Final validation loss: {training_metrics.get('final_val_loss', 0):.6f}")
            
            return self.tft_model
    
    def train_rl_agent_mobile(self, data: pd.DataFrame, tft_model: TradingTFT) -> PPO:
        """Train RL agent with mobile optimizations"""
        logger.info("Starting mobile GPU optimized RL training...")
        
        with mlflow.start_run(run_name="RL_Mobile_Training", nested=True):
            # Log mobile RL parameters
            mlflow.log_params({
                "mobile_gpu_mode": True,
                "rl_total_timesteps": config.rl.TOTAL_TIMESTEPS,
                "rl_n_steps": config.rl.N_STEPS,
                "rl_batch_size": config.rl.BATCH_SIZE
            })
            
            self.clear_memory()
            self.monitor_memory("Before RL training")
            
            # Create trading environment
            logger.info("Creating mobile-optimized trading environment...")
            env = HybridTradingEnvironment(
                data=data,
                tft_model=tft_model,
                initial_balance=config.trading.ACCOUNT_SIZE
            )
            
            # Single environment for mobile GPU (no vectorization)
            logger.info("Creating RL agent...")
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
            
            # Setup evaluation callback
            eval_callback = EvalCallback(
                env,
                best_model_save_path=str(config.data.MODEL_DIR / "rl_mobile_best"),
                log_path=str(config.data.LOGS_DIR / "rl_mobile_eval"),
                eval_freq=5000,  # More frequent evaluation
                deterministic=True,
                render=False
            )
            
            # Train with mobile optimizations
            logger.info("Starting mobile RL training...")
            self.rl_model.learn(
                total_timesteps=config.rl.TOTAL_TIMESTEPS,
                callback=eval_callback,
                progress_bar=True
            )
            
            # Save the final model
            model_path = config.data.MODEL_DIR / "rl_mobile_final.zip"
            self.rl_model.save(str(model_path))
            mlflow.log_artifact(str(model_path))
            
            self.clear_memory()
            self.monitor_memory("After RL training")
            
            logger.info("Mobile RL training completed successfully")
            
            return self.rl_model
    
    def run_mobile_training(self, start_date: str = None, end_date: str = None):
        """Run mobile GPU optimized training pipeline"""
        logger.info("Starting Mobile GPU Hybrid Training Pipeline")
        logger.info(f"Configuration: Hidden={config.tft.HIDDEN_SIZE}, Lookback={config.tft.LOOKBACK_PERIODS}, Batch={config.tft.BATCH_SIZE}")
        
        with mlflow.start_run(run_name="Mobile_Hybrid_Training"):
            # Log hardware info
            if torch.cuda.is_available():
                gpu_props = torch.cuda.get_device_properties(0)
                mlflow.log_params({
                    "gpu_name": gpu_props.name,
                    "gpu_memory_gb": gpu_props.total_memory / 1024**3,
                    "mobile_optimized": True
                })
            
            try:
                # Step 1: Data preparation
                logger.info("=== Step 1: Mobile Data Preparation ===")
                data = self.load_and_prepare_data(start_date, end_date)
                
                # Step 2: Train TFT
                logger.info("=== Step 2: Mobile TFT Training ===")
                tft_model = self.train_tft_model_mobile(data)
                
                # Step 3: Train RL
                logger.info("=== Step 3: Mobile RL Training ===")
                rl_model = self.train_rl_agent_mobile(data, tft_model)
                
                # Step 4: Quick evaluation
                logger.info("=== Step 4: Mobile Evaluation ===")
                env = HybridTradingEnvironment(
                    data=data,
                    tft_model=tft_model,
                    initial_balance=config.trading.ACCOUNT_SIZE
                )
                
                # Quick evaluation (fewer episodes for mobile)
                eval_episodes = 5
                total_rewards = []
                
                for episode in range(eval_episodes):
                    obs, info = env.reset()
                    episode_reward = 0
                    done = False
                    
                    while not done:
                        action, _ = rl_model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env.step(action)
                        episode_reward += reward
                        done = terminated or truncated
                    
                    total_rewards.append(episode_reward)
                    logger.info(f"Episode {episode + 1}: Reward={episode_reward:.2f}")
                
                final_metrics = {
                    "avg_episode_reward": np.mean(total_rewards),
                    "total_episodes_evaluated": eval_episodes
                }
                
                mlflow.log_metrics(final_metrics)
                mlflow.log_metrics({"mobile_training_successful": 1})
                
                logger.info("=== Mobile Training Pipeline Completed ===")
                logger.info(f"Average Episode Reward: {final_metrics['avg_episode_reward']:.2f}")
                
                return {
                    "tft_model": tft_model,
                    "rl_model": rl_model,
                    "final_metrics": final_metrics
                }
                
            except Exception as e:
                logger.error(f"Mobile training failed: {str(e)}")
                mlflow.log_metrics({"mobile_training_successful": 0})
                raise

def main():
    """Main mobile training script"""
    parser = argparse.ArgumentParser(description="Mobile GPU Hybrid Trading System Training")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU training")
    
    args = parser.parse_args()
    
    # Force CPU if requested
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        logger.info("Forcing CPU-only training")
    
    # Check GPU availability
    if torch.cuda.is_available() and not args.cpu_only:
        gpu_props = torch.cuda.get_device_properties(0)
        vram_gb = gpu_props.total_memory / 1024**3
        logger.info(f"Detected GPU: {gpu_props.name} with {vram_gb:.1f}GB VRAM")
        
        if vram_gb < 4:
            logger.warning("GPU has less than 4GB VRAM. Consider using --cpu-only for stability")
    else:
        logger.info("Using CPU training (will be slower)")
    
    # Initialize mobile trainer
    trainer = MobileGPUTrainer()
    
    try:
        # Run mobile training
        results = trainer.run_mobile_training(args.start_date, args.end_date)
        
        logger.info("Mobile training completed successfully!")
        logger.info("Models saved to:")
        logger.info(f"  TFT Model: {config.data.MODEL_DIR / 'tft_mobile_model.pt'}")
        logger.info(f"  RL Model: {config.data.MODEL_DIR / 'rl_mobile_final.zip'}")
        
        # Memory cleanup
        trainer.clear_memory()
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Mobile training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 