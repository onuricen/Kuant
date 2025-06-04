"""
Configuration settings for the Nasdaq Futures Hybrid ML Trading System
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

@dataclass
class TradingConfig:
    """Trading-specific configuration"""
    # Account Configuration
    ACCOUNT_SIZE: float = 100_000.0  # $100,000
    DEFAULT_RISK_PER_TRADE: float = 0.01  # 1% of account
    MAX_DRAWDOWN_THRESHOLD: float = 0.10  # 10% maximum drawdown
    
    # Risk-Reward Configuration
    TARGET_RR_RATIO: float = 2.0  # 2:1 risk-reward ratio
    MIN_RR_RATIO: float = 1.8  # Minimum acceptable RR
    
    # Position Management
    MAX_POSITION_SIZE: float = 10.0  # Maximum contracts
    MIN_POSITION_SIZE: float = 0.1   # Minimum contracts
    SINGLE_POSITION_LIMIT: bool = True  # Only one position at a time
    
    # Session Management (NY Timezone)
    ASIA_SESSION_START: int = 23  # 11 PM NY time
    ASIA_SESSION_END: int = 8     # 8 AM NY time
    
    # Stop Loss Management
    ENABLE_DYNAMIC_STOPS: bool = True
    STOP_ADJUSTMENT_AT_50_PERCENT: bool = True
    
    # Reverse Martingale System
    ENABLE_REVERSE_MARTINGALE: bool = True
    MAX_CONSECUTIVE_LOSSES: int = 3
    RISK_REDUCTION_FACTOR: float = 0.5

@dataclass
class TFTConfig:
    """Temporal Fusion Transformer configuration"""
    # Model Architecture
    HIDDEN_SIZE: int = 128
    LSTM_LAYERS: int = 2
    ATTENTION_HEAD_SIZE: int = 4
    DROPOUT: float = 0.1
    
    # Data Configuration
    LOOKBACK_PERIODS: int = 200  # Extended for attention mechanisms
    PREDICTION_HORIZONS: list = None  # [1, 5, 15, 50] periods ahead
    
    # Feature Configuration
    STATIC_CATEGORICALS: list = None
    STATIC_REALS: list = None
    TIME_VARYING_KNOWN_CATEGORICALS: list = None
    TIME_VARYING_KNOWN_REALS: list = None
    TIME_VARYING_UNKNOWN_REALS: list = None
    
    # Training Configuration
    MAX_EPOCHS: int = 100
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 1e-5
    
    # Quantile Configuration
    QUANTILES: list = None  # [0.1, 0.5, 0.9] for uncertainty estimation
    
    # Confidence Thresholds
    MIN_CLASSIFICATION_CONFIDENCE: float = 0.7
    MAX_UNCERTAINTY_SPREAD: float = 0.3
    
    # Mobile GPU Optimization Profile
    MOBILE_GPU_MODE: bool = False
    
    def __post_init__(self):
        if self.PREDICTION_HORIZONS is None:
            self.PREDICTION_HORIZONS = [1, 5, 15, 50]
        if self.QUANTILES is None:
            self.QUANTILES = [0.1, 0.5, 0.9]
        if self.STATIC_CATEGORICALS is None:
            self.STATIC_CATEGORICALS = ['session_type', 'day_of_week']
        if self.TIME_VARYING_KNOWN_CATEGORICALS is None:
            self.TIME_VARYING_KNOWN_CATEGORICALS = ['hour', 'is_asia_session']
        if self.TIME_VARYING_UNKNOWN_REALS is None:
            self.TIME_VARYING_UNKNOWN_REALS = [
                'open', 'high', 'low', 'close', 'volume',
                'vwap_hourly', 'vwap_daily', 'vwap_weekly', 'vwap_monthly',
                'vwap_dist_hourly', 'vwap_dist_daily', 'vwap_dist_weekly', 'vwap_dist_monthly',
                'poc', 'vah', 'val', 'volume_profile_percentile'
            ]
        
        # Apply mobile GPU optimizations
        if self.MOBILE_GPU_MODE:
            self.HIDDEN_SIZE = 64  # Reduce from 128
            self.LOOKBACK_PERIODS = 100  # Reduce from 200
            self.BATCH_SIZE = 16  # Reduce from 64
            self.ATTENTION_HEAD_SIZE = 2  # Reduce from 4

@dataclass
class RLConfig:
    """Reinforcement Learning configuration"""
    # Algorithm Configuration
    ALGORITHM: str = "PPO"  # Proximal Policy Optimization
    POLICY_TYPE: str = "MlpPolicy"
    
    # Network Architecture
    NET_ARCH: list = None  # [256, 128, 64] for both actor and critic
    ACTIVATION_FN: str = "tanh"
    
    # Training Configuration
    TOTAL_TIMESTEPS: int = 1_000_000
    LEARNING_RATE: float = 3e-4
    N_STEPS: int = 2048
    BATCH_SIZE: int = 64
    N_EPOCHS: int = 10
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_RANGE: float = 0.2
    ENT_COEF: float = 0.01
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    
    # Action Space Configuration
    N_ACTIONS: int = 6  # Hold, Long, Short, Close, Adjust Stop, Reduce Position
    
    # State Space Configuration
    TFT_OUTPUT_DIM: int = 10  # Dimension of TFT output features
    MARKET_CONTEXT_DIM: int = 15  # Market context features
    POSITION_STATE_DIM: int = 8   # Position and risk state features
    
    # Reward Configuration
    TRADE_PNL_WEIGHT: float = 100.0
    PATTERN_ALIGNMENT_BONUS: float = 10.0
    PATTERN_PENALTY: float = -5.0
    RISK_COMPLIANCE_BONUS: float = 5.0
    DRAWDOWN_PENALTY: float = -50.0
    TIMING_BONUS: float = 3.0
    
    def __post_init__(self):
        if self.NET_ARCH is None:
            self.NET_ARCH = [256, 128, 64]

@dataclass
class DataConfig:
    """Data processing configuration"""
    # File Paths
    DATA_DIR: Path = Path("data")
    PROCESSED_DATA_DIR: Path = Path("data/processed")
    MODEL_DIR: Path = Path("models")
    LOGS_DIR: Path = Path("logs")
    
    # Data Processing
    RESAMPLE_FREQ: str = "5T"  # 5-minute bars for decision making
    VOLUME_WINDOW: int = 50    # Rolling window for volume normalization
    PRICE_NORMALIZATION: str = "zscore"  # zscore, minmax, robust
    
    # Feature Engineering
    VWAP_PERIODS: dict = None  # Hourly, Daily, Weekly, Monthly periods
    VOLUME_PROFILE_LOOKBACK: int = 7  # Days for volume profile calculation
    TPO_PERIODS: int = 30  # Periods for TPO chart calculation
    
    # Data Validation
    MIN_VOLUME_THRESHOLD: float = 1.0
    MAX_PRICE_DEVIATION: float = 0.10  # 10% maximum price deviation filter
    
    def __post_init__(self):
        if self.VWAP_PERIODS is None:
            self.VWAP_PERIODS = {
                'hourly': 12,    # 12 periods of 5-min = 1 hour
                'daily': 288,    # 288 periods of 5-min = 1 day
                'weekly': 2016,  # 2016 periods of 5-min = 1 week
                'monthly': 8640  # 8640 periods of 5-min = 1 month
            }

@dataclass
class SystemConfig:
    """System-wide configuration"""
    # Environment
    ENVIRONMENT: str = "development"  # development, production
    DEBUG: bool = True
    
    # Database Configuration
    DATABASE_URL: Optional[str] = None
    REDIS_URL: Optional[str] = None
    
    # API Configuration
    API_HOST: str = "localhost"
    API_PORT: int = 8000
    
    # Monitoring
    MLFLOW_TRACKING_URI: Optional[str] = None
    WANDB_PROJECT: Optional[str] = None
    
    # Real-time Trading
    BROKER_API: Optional[str] = None
    LIVE_TRADING: bool = False
    PAPER_TRADING: bool = True
    
    # Performance
    N_JOBS: int = -1  # Use all available cores
    RANDOM_SEED: int = 42

# Main configuration class
@dataclass
class Config:
    """Main configuration class combining all sub-configurations"""
    trading: TradingConfig = field(default_factory=TradingConfig)
    tft: TFTConfig = field(default_factory=TFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self):
        # Create directories if they don't exist
        for dir_path in [self.data.PROCESSED_DATA_DIR, self.data.MODEL_DIR, self.data.LOGS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load environment variables
        self._load_env_variables()
    
    def _load_env_variables(self):
        """Load configuration from environment variables"""
        self.system.DATABASE_URL = os.getenv("DATABASE_URL")
        self.system.REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.system.MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
        self.system.WANDB_PROJECT = os.getenv("WANDB_PROJECT", "nasdaq-futures-hybrid-ml")
        self.system.BROKER_API = os.getenv("BROKER_API")
        
        # Override with environment variables if available
        if os.getenv("LIVE_TRADING"):
            self.system.LIVE_TRADING = os.getenv("LIVE_TRADING").lower() == "true"
        if os.getenv("DEBUG"):
            self.system.DEBUG = os.getenv("DEBUG").lower() == "true"

# Global configuration instance
config = Config()

# Mobile GPU configuration
class MobileGPUConfig(Config):
    """Optimized configuration for mobile GPUs like GTX 1060 Ti"""
    
    def __init__(self):
        super().__init__()
        # Enable mobile GPU optimizations
        self.tft.MOBILE_GPU_MODE = True
        self.tft.HIDDEN_SIZE = 64
        self.tft.LOOKBACK_PERIODS = 100
        self.tft.BATCH_SIZE = 16
        self.tft.ATTENTION_HEAD_SIZE = 2
        self.tft.MAX_EPOCHS = 50  # Reduce training time
        
        # Adjust RL for faster training
        self.rl.TOTAL_TIMESTEPS = 500_000  # Reduce from 1M
        self.rl.N_STEPS = 1024  # Reduce from 2048
        self.rl.BATCH_SIZE = 32  # Reduce from 64

# Function to get mobile config
def get_mobile_gpu_config():
    """Get optimized configuration for mobile GPUs"""
    return MobileGPUConfig() 