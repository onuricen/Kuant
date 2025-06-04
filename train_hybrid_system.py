#!/usr/bin/env python3
"""
Main training script for Nasdaq Futures Hybrid ML Trading System
Combines Temporal Fusion Transformer (TFT) and Reinforcement Learning (RL)
"""

import logging
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
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
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import mlflow
import mlflow.pytorch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.data.LOGS_DIR / 'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HybridTradingSystemTrainer:
    """
    Orchestrates the training of the hybrid TFT+RL trading system
    """
    
    def __init__(self):
        self.data_loader = NasdaqFuturesDataLoader()
        self.feature_engineer = FeatureEngineer()
        self.tft_model = None
        self.rl_model = None
        self.processed_data = None
        
        # MLflow setup
        if config.system.MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(config.system.MLFLOW_TRACKING_URI)
        
        mlflow.set_experiment("nasdaq_futures_hybrid_ml")
        
    def load_and_prepare_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load and prepare data for training"""
        logger.info("Loading and preparing data...")
        
        # Load raw data
        if start_date and end_date:
            raw_data = self.data_loader.load_data_range(start_date, end_date)
        else:
            raw_data = self.data_loader.load_all_data()
        
        # Resample to 5-minute bars
        resampled_data = self.data_loader.resample_to_5min(raw_data)
        
        # Apply feature engineering
        logger.info("Applying feature engineering...")
        self.processed_data = self.feature_engineer.create_complete_feature_set(resampled_data)
        
        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_nasdaq_futures_{timestamp}"
        self.data_loader.save_processed_data(self.processed_data, filename)
        
        # Log data summary
        summary = self.data_loader.get_data_summary(self.processed_data)
        logger.info(f"Data preparation complete. Shape: {self.processed_data.shape}")
        logger.info(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        
        return self.processed_data
    
    def train_tft_model(self, data: pd.DataFrame, validation_split: float = 0.2) -> TradingTFT:
        """Train the Temporal Fusion Transformer model"""
        logger.info("Starting TFT model training...")
        
        with mlflow.start_run(run_name="TFT_Training", nested=True):
            # Log TFT parameters
            mlflow.log_params({
                "tft_hidden_size": config.tft.HIDDEN_SIZE,
                "tft_attention_heads": config.tft.ATTENTION_HEAD_SIZE,
                "tft_lookback_periods": config.tft.LOOKBACK_PERIODS,
                "tft_learning_rate": config.tft.LEARNING_RATE,
                "tft_batch_size": config.tft.BATCH_SIZE,
                "tft_max_epochs": config.tft.MAX_EPOCHS
            })
            
            # Split data
            split_idx = int(len(data) * (1 - validation_split))
            train_data = data[:split_idx]
            val_data = data[split_idx:]
            
            logger.info(f"Training data: {len(train_data)} samples")
            logger.info(f"Validation data: {len(val_data)} samples")
            
            # Initialize TFT model
            self.tft_model = TradingTFT()
            
            # Prepare data for TFT
            training_dataset, validation_dataset = self.tft_model.prepare_data(data)
            
            # Create and train model
            model = self.tft_model.create_model(training_dataset)
            training_metrics = self.tft_model.train(training_dataset, validation_dataset)
            
            # Log training metrics
            mlflow.log_metrics(training_metrics)
            
            # Evaluate model
            eval_metrics = self.tft_model.evaluate_model(validation_dataset)
            mlflow.log_metrics(eval_metrics)
            
            # Save model
            model_path = config.data.MODEL_DIR / "tft_model.pt"
            self.tft_model.save_model(model_path)
            mlflow.log_artifact(str(model_path))
            
            logger.info(f"TFT training completed. Final validation loss: {training_metrics.get('final_val_loss', 0):.6f}")
            
            return self.tft_model
    
    def train_rl_agent(self, data: pd.DataFrame, tft_model: TradingTFT, 
                      train_episodes: int = 1000) -> PPO:
        """Train the RL agent using the trained TFT model"""
        logger.info("Starting RL agent training...")
        
        with mlflow.start_run(run_name="RL_Training", nested=True):
            # Log RL parameters
            mlflow.log_params({
                "rl_algorithm": config.rl.ALGORITHM,
                "rl_learning_rate": config.rl.LEARNING_RATE,
                "rl_batch_size": config.rl.BATCH_SIZE,
                "rl_n_steps": config.rl.N_STEPS,
                "rl_total_timesteps": config.rl.TOTAL_TIMESTEPS,
                "train_episodes": train_episodes
            })
            
            # Create trading environment
            logger.info("Creating trading environment...")
            env = HybridTradingEnvironment(
                data=data,
                tft_model=tft_model,
                initial_balance=config.trading.ACCOUNT_SIZE
            )
            
            # Create vectorized environment for parallel training
            def make_env():
                return HybridTradingEnvironment(
                    data=data,
                    tft_model=tft_model,
                    initial_balance=config.trading.ACCOUNT_SIZE
                )
            
            vec_env = make_vec_env(make_env, n_envs=1)
            
            # Create RL model
            logger.info("Creating RL agent...")
            self.rl_model = PPO(
                config.rl.POLICY_TYPE,
                vec_env,
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
                tensorboard_log=str(config.data.LOGS_DIR / "rl_tensorboard")
            )
            
            # Setup callbacks
            eval_env = make_env()
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(config.data.MODEL_DIR / "rl_best_model"),
                log_path=str(config.data.LOGS_DIR / "rl_eval"),
                eval_freq=10000,
                deterministic=True,
                render=False
            )
            
            # Train the model
            logger.info("Starting RL training...")
            self.rl_model.learn(
                total_timesteps=config.rl.TOTAL_TIMESTEPS,
                callback=eval_callback,
                progress_bar=True
            )
            
            # Save the final model
            model_path = config.data.MODEL_DIR / "rl_final_model.zip"
            self.rl_model.save(str(model_path))
            mlflow.log_artifact(str(model_path))
            
            # Evaluate trained agent
            logger.info("Evaluating trained RL agent...")
            eval_metrics = self.evaluate_rl_agent(env, episodes=10)
            mlflow.log_metrics(eval_metrics)
            
            logger.info("RL training completed successfully")
            
            return self.rl_model
    
    def evaluate_rl_agent(self, env: HybridTradingEnvironment, episodes: int = 10) -> dict:
        """Evaluate the trained RL agent"""
        logger.info(f"Evaluating RL agent over {episodes} episodes...")
        
        total_rewards = []
        episode_metrics = []
        
        for episode in range(episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.rl_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            total_rewards.append(episode_reward)
            metrics = env.get_trading_metrics()
            episode_metrics.append(metrics)
            
            logger.info(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
                       f"Return={metrics.get('total_return', 0):.2%}, "
                       f"Win Rate={metrics.get('win_rate', 0):.2%}")
        
        # Aggregate metrics
        avg_metrics = {}
        if episode_metrics:
            for key in episode_metrics[0].keys():
                values = [m.get(key, 0) for m in episode_metrics if key in m]
                if values:
                    avg_metrics[f"avg_{key}"] = np.mean(values)
                    avg_metrics[f"std_{key}"] = np.std(values)
        
        avg_metrics.update({
            "avg_episode_reward": np.mean(total_rewards),
            "std_episode_reward": np.std(total_rewards),
            "min_episode_reward": np.min(total_rewards),
            "max_episode_reward": np.max(total_rewards)
        })
        
        return avg_metrics
    
    def run_hybrid_training(self, start_date: str = None, end_date: str = None):
        """Run the complete hybrid training pipeline"""
        logger.info("Starting Hybrid TFT+RL Trading System Training")
        
        with mlflow.start_run(run_name="Hybrid_System_Training"):
            # Log system configuration
            mlflow.log_params({
                "account_size": config.trading.ACCOUNT_SIZE,
                "risk_per_trade": config.trading.DEFAULT_RISK_PER_TRADE,
                "target_rr_ratio": config.trading.TARGET_RR_RATIO,
                "max_drawdown_threshold": config.trading.MAX_DRAWDOWN_THRESHOLD,
                "lookback_periods": config.tft.LOOKBACK_PERIODS,
                "total_timesteps": config.rl.TOTAL_TIMESTEPS
            })
            
            try:
                # Step 1: Load and prepare data
                logger.info("=== Step 1: Data Preparation ===")
                data = self.load_and_prepare_data(start_date, end_date)
                mlflow.log_metrics({
                    "total_samples": len(data),
                    "total_features": len(data.columns)
                })
                
                # Step 2: Train TFT model
                logger.info("=== Step 2: TFT Model Training ===")
                tft_model = self.train_tft_model(data)
                
                # Step 3: Train RL agent
                logger.info("=== Step 3: RL Agent Training ===")
                rl_model = self.train_rl_agent(data, tft_model)
                
                # Step 4: Final evaluation
                logger.info("=== Step 4: Final System Evaluation ===")
                env = HybridTradingEnvironment(
                    data=data,
                    tft_model=tft_model,
                    initial_balance=config.trading.ACCOUNT_SIZE
                )
                
                final_metrics = self.evaluate_rl_agent(env, episodes=20)
                mlflow.log_metrics(final_metrics)
                
                # Log success
                mlflow.log_metrics({"training_successful": 1})
                
                logger.info("=== Training Pipeline Completed Successfully ===")
                logger.info(f"Final Performance Metrics:")
                for key, value in final_metrics.items():
                    if "avg_" in key:
                        logger.info(f"  {key}: {value:.4f}")
                
                return {
                    "tft_model": tft_model,
                    "rl_model": rl_model,
                    "final_metrics": final_metrics
                }
                
            except Exception as e:
                logger.error(f"Training pipeline failed: {str(e)}")
                mlflow.log_metrics({"training_successful": 0})
                raise
    
    def load_pretrained_models(self, tft_path: str = None, rl_path: str = None):
        """Load pre-trained models"""
        if tft_path:
            logger.info(f"Loading TFT model from {tft_path}")
            self.tft_model = TradingTFT()
            self.tft_model.load_model(Path(tft_path))
        
        if rl_path:
            logger.info(f"Loading RL model from {rl_path}")
            self.rl_model = PPO.load(rl_path)
        
        return self.tft_model, self.rl_model

def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description="Train Hybrid TFT+RL Trading System")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--tft-only", action="store_true", help="Train only TFT model")
    parser.add_argument("--rl-only", action="store_true", help="Train only RL agent")
    parser.add_argument("--tft-path", type=str, help="Path to pre-trained TFT model")
    parser.add_argument("--rl-path", type=str, help="Path to pre-trained RL model")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate pre-trained models")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = HybridTradingSystemTrainer()
    
    try:
        if args.eval_only:
            # Evaluation mode
            logger.info("Running in evaluation mode")
            trainer.load_pretrained_models(args.tft_path, args.rl_path)
            
            # Load data for evaluation
            data = trainer.load_and_prepare_data(args.start_date, args.end_date)
            
            # Create environment and evaluate
            env = HybridTradingEnvironment(
                data=data,
                tft_model=trainer.tft_model,
                initial_balance=config.trading.ACCOUNT_SIZE
            )
            
            metrics = trainer.evaluate_rl_agent(env, episodes=50)
            logger.info("Evaluation Results:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")
        
        elif args.tft_only:
            # Train only TFT
            logger.info("Training TFT model only")
            data = trainer.load_and_prepare_data(args.start_date, args.end_date)
            trainer.train_tft_model(data)
        
        elif args.rl_only:
            # Train only RL (requires pre-trained TFT)
            logger.info("Training RL agent only")
            if not args.tft_path:
                raise ValueError("TFT model path required for RL-only training")
            
            trainer.load_pretrained_models(args.tft_path)
            data = trainer.load_and_prepare_data(args.start_date, args.end_date)
            trainer.train_rl_agent(data, trainer.tft_model)
        
        else:
            # Full hybrid training pipeline
            logger.info("Running full hybrid training pipeline")
            results = trainer.run_hybrid_training(args.start_date, args.end_date)
            
            logger.info("Training completed successfully!")
            logger.info("Models saved to:")
            logger.info(f"  TFT Model: {config.data.MODEL_DIR / 'tft_model.pt'}")
            logger.info(f"  RL Model: {config.data.MODEL_DIR / 'rl_final_model.zip'}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 