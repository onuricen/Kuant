"""
Temporal Fusion Transformer (TFT) implementation for trading pattern recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_forecasting as ptf
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss, MultiLoss
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

from ..config import config

logger = logging.getLogger(__name__)

class TradingTFT:
    """
    Temporal Fusion Transformer for trading pattern recognition with uncertainty quantification
    """
    
    def __init__(self, config_tft=None):
        self.config = config_tft or config.tft
        self.model = None
        self.dataset = None
        self.trainer = None
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = None
        
    def prepare_data(self, df: pd.DataFrame) -> TimeSeriesDataSet:
        """Prepare data for TFT training"""
        logger.info("Preparing data for TFT training")
        
        # Create time index (assuming the dataframe index is datetime)
        df = df.copy()
        df.reset_index(inplace=True)
        df['time_idx'] = range(len(df))
        
        # Add group identifier (for multi-series support, we'll use a single group)
        df['group'] = 'nasdaq_futures'
        
        # Prepare categorical features
        categorical_features = []
        for col in self.config.STATIC_CATEGORICALS + self.config.TIME_VARYING_KNOWN_CATEGORICALS:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].fillna(0))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].fillna(0))
                categorical_features.append(col)
        
        # Prepare continuous features
        continuous_features = []
        for col in self.config.TIME_VARYING_UNKNOWN_REALS:
            if col in df.columns:
                # Handle missing values
                df[col] = df[col].fillna(df[col].median())
                continuous_features.append(col)
        
        # Create target variables for multi-task learning
        target_variables = []
        
        # Classification targets
        if 'can_long_2_to_1' in df.columns:
            target_variables.append('can_long_2_to_1')
        if 'can_short_2_to_1' in df.columns:
            target_variables.append('can_short_2_to_1')
        
        # Regression targets for quantile prediction
        for horizon in [1, 5, 15, 50]:
            target_col = f'future_return_{horizon}'
            if target_col in df.columns:
                df[target_col] = df[target_col].fillna(0)
                target_variables.append(target_col)
        
        # Use the first available target as the main target
        main_target = target_variables[0] if target_variables else 'close'
        
        # Create TimeSeriesDataSet
        max_encoder_length = self.config.LOOKBACK_PERIODS
        max_prediction_length = max(self.config.PREDICTION_HORIZONS)
        
        training_data = TimeSeriesDataSet(
            df[:-max_prediction_length],
            time_idx="time_idx",
            target=main_target,
            group_ids=["group"],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=self.config.STATIC_CATEGORICALS,
            static_reals=[],
            time_varying_known_categoricals=self.config.TIME_VARYING_KNOWN_CATEGORICALS,
            time_varying_known_reals=[],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=continuous_features,
            target_normalizer=GroupNormalizer(
                groups=["group"], 
                transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=False,
        )
        
        # Create validation dataset
        validation_data = TimeSeriesDataSet.from_dataset(
            training_data, 
            df,
            predict=True, 
            stop_randomization=True
        )
        
        self.dataset = training_data
        logger.info(f"Created TFT dataset with {len(training_data)} training samples")
        
        return training_data, validation_data
    
    def create_model(self, training_data: TimeSeriesDataSet) -> TemporalFusionTransformer:
        """Create TFT model"""
        logger.info("Creating TFT model")
        
        # Model configuration
        model_config = {
            "learning_rate": self.config.LEARNING_RATE,
            "hidden_size": self.config.HIDDEN_SIZE,
            "attention_head_size": self.config.ATTENTION_HEAD_SIZE,
            "dropout": self.config.DROPOUT,
            "hidden_continuous_size": self.config.HIDDEN_SIZE // 2,
            "output_size": len(self.config.QUANTILES),
            "loss": QuantileLoss(quantiles=self.config.QUANTILES),
            "log_interval": 10,
            "reduce_on_plateau_patience": 4,
            "optimizer": "AdamW",
            "weight_decay": self.config.WEIGHT_DECAY,
        }
        
        self.model = TemporalFusionTransformer.from_dataset(
            training_data,
            **model_config
        )
        
        logger.info(f"Created TFT model with {sum(p.numel() for p in self.model.parameters())} parameters")
        return self.model
    
    def train(self, training_data: TimeSeriesDataSet, validation_data: TimeSeriesDataSet) -> Dict:
        """Train the TFT model"""
        logger.info("Starting TFT training")
        
        # Create data loaders
        train_dataloader = training_data.to_dataloader(
            train=True, 
            batch_size=self.config.BATCH_SIZE, 
            num_workers=0
        )
        val_dataloader = validation_data.to_dataloader(
            train=False, 
            batch_size=self.config.BATCH_SIZE * 2, 
            num_workers=0
        )
        
        # Create trainer
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
        from pytorch_lightning.loggers import TensorBoardLogger
        
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=10,
            verbose=False,
            mode="min"
        )
        
        lr_logger = LearningRateMonitor(logging_interval="step")
        
        logger_tb = TensorBoardLogger(
            save_dir=str(config.data.LOGS_DIR),
            name="tft_training"
        )
        
        self.trainer = Trainer(
            max_epochs=self.config.MAX_EPOCHS,
            accelerator="auto",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback, lr_logger],
            logger=logger_tb,
            log_every_n_steps=10
        )
        
        # Train the model
        self.trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        # Get training metrics
        training_metrics = {
            'final_train_loss': float(self.trainer.callback_metrics.get('train_loss', 0)),
            'final_val_loss': float(self.trainer.callback_metrics.get('val_loss', 0)),
            'epochs_trained': self.trainer.current_epoch
        }
        
        logger.info(f"TFT training completed. Final validation loss: {training_metrics['final_val_loss']:.6f}")
        
        return training_metrics
    
    def predict(self, df: pd.DataFrame, return_attention: bool = True) -> Dict[str, np.ndarray]:
        """Make predictions with the trained TFT model"""
        if self.model is None or self.dataset is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare data in the same format as training
        df_pred = df.copy()
        df_pred.reset_index(inplace=True)
        df_pred['time_idx'] = range(len(df_pred))
        df_pred['group'] = 'nasdaq_futures'
        
        # Apply same preprocessing as training
        for col in self.config.STATIC_CATEGORICALS + self.config.TIME_VARYING_KNOWN_CATEGORICALS:
            if col in df_pred.columns and col in self.label_encoders:
                df_pred[col] = self.label_encoders[col].transform(df_pred[col].fillna(0))
        
        for col in self.config.TIME_VARYING_UNKNOWN_REALS:
            if col in df_pred.columns:
                df_pred[col] = df_pred[col].fillna(df_pred[col].median())
        
        # Create prediction dataset
        predict_data = TimeSeriesDataSet.from_dataset(
            self.dataset, 
            df_pred, 
            predict=True, 
            stop_randomization=True
        )
        
        predict_dataloader = predict_data.to_dataloader(
            train=False, 
            batch_size=self.config.BATCH_SIZE * 2, 
            num_workers=0
        )
        
        # Make predictions
        predictions = self.trainer.predict(self.model, dataloaders=predict_dataloader)
        
        # Process predictions
        if predictions:
            # Combine all batch predictions
            pred_tensor = torch.cat([p for p in predictions], dim=0)
            
            # Extract quantile predictions
            quantile_predictions = pred_tensor.cpu().numpy()
            
            # Calculate classification probabilities (if applicable)
            # This is a simplified version - you might need to adjust based on your exact targets
            long_probs = np.mean(quantile_predictions > 0, axis=1)
            short_probs = np.mean(quantile_predictions < 0, axis=1)
            confidence_scores = np.std(quantile_predictions, axis=1)
            
            results = {
                'quantile_predictions': quantile_predictions,
                'long_probability': long_probs,
                'short_probability': short_probs,
                'confidence_scores': confidence_scores,
                'uncertainty_spread': np.std(quantile_predictions, axis=1)
            }
            
            # Add attention weights if requested
            if return_attention:
                try:
                    # This requires modification of the forward method to return attention
                    # For now, we'll return dummy attention weights
                    attention_weights = np.random.random((len(quantile_predictions), self.config.LOOKBACK_PERIODS))
                    results['attention_weights'] = attention_weights
                except:
                    logger.warning("Could not extract attention weights")
            
            return results
        
        else:
            logger.error("No predictions generated")
            return {}
    
    def calculate_pattern_confidence(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate pattern confidence based on TFT outputs"""
        if 'confidence_scores' in predictions:
            # Use inverse of uncertainty as confidence measure
            uncertainty = predictions['uncertainty_spread']
            confidence = 1.0 / (1.0 + uncertainty)
            
            # Normalize to 0-1 range
            confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())
            
            return confidence
        else:
            return np.ones(len(predictions['quantile_predictions'])) * 0.5
    
    def save_model(self, model_path: Path):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.hparams,
            'scalers': self.scalers,
            'label_encoders': self.label_encoders,
            'config': self.config.__dict__
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: Path):
        """Load a trained model"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Restore configuration
        self.config.__dict__.update(checkpoint['config'])
        self.scalers = checkpoint['scalers']
        self.label_encoders = checkpoint['label_encoders']
        
        # Create model with saved configuration
        # Note: This requires having the dataset structure available
        # In practice, you might need to save and restore the dataset metadata as well
        
        logger.info(f"Model loaded from {model_path}")
    
    def evaluate_model(self, validation_data: TimeSeriesDataSet) -> Dict[str, float]:
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        val_dataloader = validation_data.to_dataloader(
            train=False, 
            batch_size=self.config.BATCH_SIZE * 2, 
            num_workers=0
        )
        
        # Calculate validation metrics
        val_results = self.trainer.test(self.model, dataloaders=val_dataloader)
        
        if val_results:
            metrics = {
                'val_loss': float(val_results[0].get('test_loss', 0)),
                'quantile_loss': float(val_results[0].get('test_QuantileLoss', 0))
            }
        else:
            metrics = {'val_loss': 0, 'quantile_loss': 0}
        
        return metrics

class TFTOutputProcessor:
    """Process TFT outputs for RL agent consumption"""
    
    def __init__(self, config_tft=None):
        self.config = config_tft or config.tft
    
    def process_for_rl(self, tft_predictions: Dict[str, np.ndarray], 
                      current_market_data: pd.Series) -> Dict[str, float]:
        """
        Process TFT outputs into a format suitable for RL agent state space
        """
        
        if not tft_predictions:
            # Return default values if no predictions
            return {
                'long_probability': 0.5,
                'short_probability': 0.5,
                'pattern_confidence': 0.0,
                'price_uncertainty': 1.0,
                'quantile_p10': current_market_data['close'],
                'quantile_p50': current_market_data['close'],
                'quantile_p90': current_market_data['close'],
                'uncertainty_spread': 1.0,
                'attention_focus': 0.5
            }
        
        # Extract key metrics for RL state
        result = {}
        
        # Classification probabilities
        result['long_probability'] = float(tft_predictions.get('long_probability', [0.5])[-1])
        result['short_probability'] = float(tft_predictions.get('short_probability', [0.5])[-1])
        
        # Pattern confidence (higher is better)
        if 'confidence_scores' in tft_predictions:
            result['pattern_confidence'] = float(tft_predictions['confidence_scores'][-1])
        else:
            result['pattern_confidence'] = 0.5
        
        # Uncertainty measures
        if 'uncertainty_spread' in tft_predictions:
            result['uncertainty_spread'] = float(tft_predictions['uncertainty_spread'][-1])
            result['price_uncertainty'] = min(1.0, result['uncertainty_spread'] / 0.1)  # Normalize
        else:
            result['uncertainty_spread'] = 0.5
            result['price_uncertainty'] = 0.5
        
        # Quantile predictions for risk assessment
        if 'quantile_predictions' in tft_predictions:
            quantiles = tft_predictions['quantile_predictions'][-1]
            if len(quantiles) >= 3:
                result['quantile_p10'] = float(quantiles[0])
                result['quantile_p50'] = float(quantiles[1])
                result['quantile_p90'] = float(quantiles[2])
            else:
                current_price = current_market_data['close']
                result['quantile_p10'] = current_price * 0.99
                result['quantile_p50'] = current_price
                result['quantile_p90'] = current_price * 1.01
        else:
            current_price = current_market_data['close']
            result['quantile_p10'] = current_price * 0.99
            result['quantile_p50'] = current_price
            result['quantile_p90'] = current_price * 1.01
        
        # Attention focus (average attention on recent periods)
        if 'attention_weights' in tft_predictions:
            recent_attention = tft_predictions['attention_weights'][-1][-20:]  # Last 20 periods
            result['attention_focus'] = float(np.mean(recent_attention))
        else:
            result['attention_focus'] = 0.5
        
        return result
    
    def validate_2_to_1_trade_potential(self, tft_output: Dict[str, float], 
                                      current_price: float) -> Dict[str, bool]:
        """
        Validate if TFT predictions suggest viable 2:1 RR trades
        """
        
        # Check confidence thresholds
        confidence_check = tft_output['pattern_confidence'] >= self.config.MIN_CLASSIFICATION_CONFIDENCE
        uncertainty_check = tft_output['uncertainty_spread'] <= self.config.MAX_UNCERTAINTY_SPREAD
        
        # Check if quantiles suggest sufficient price movement for 2:1 RR
        p10 = tft_output['quantile_p10']
        p90 = tft_output['quantile_p90']
        
        # For long trade: check if upside potential is 2x downside risk
        downside_risk = current_price - p10
        upside_potential = p90 - current_price
        
        long_rr_check = False
        short_rr_check = False
        
        if downside_risk > 0:
            long_rr = upside_potential / downside_risk
            long_rr_check = long_rr >= config.trading.TARGET_RR_RATIO
        
        if upside_potential > 0:
            short_rr = downside_risk / upside_potential
            short_rr_check = short_rr >= config.trading.TARGET_RR_RATIO
        
        return {
            'confidence_valid': confidence_check,
            'uncertainty_valid': uncertainty_check,
            'long_rr_valid': long_rr_check,
            'short_rr_valid': short_rr_check,
            'overall_long_valid': confidence_check and uncertainty_check and long_rr_check,
            'overall_short_valid': confidence_check and uncertainty_check and short_rr_check
        } 