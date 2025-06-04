"""
Hybrid RL Trading Environment that integrates TFT predictions for decision making
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from enum import IntEnum

from ..config import config
from ..models.tft_model import TFTOutputProcessor

logger = logging.getLogger(__name__)

class TradingAction(IntEnum):
    """Trading actions available to the RL agent"""
    HOLD = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3
    ADJUST_STOP = 4
    REDUCE_POSITION = 5

class PositionType(IntEnum):
    """Position types"""
    NONE = 0
    LONG = 1
    SHORT = 2

class HybridTradingEnvironment(gym.Env):
    """
    Hybrid RL Trading Environment that incorporates TFT intelligence
    for Nasdaq Futures trading with 2:1 RR enforcement
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 tft_model=None,
                 initial_balance: float = None,
                 transaction_cost: float = 2.0,
                 leverage: float = 1.0):
        
        super().__init__()
        
        # Configuration
        self.trading_config = config.trading
        self.rl_config = config.rl
        
        # Data and model setup
        self.data = data.copy()
        self.tft_model = tft_model
        self.tft_processor = TFTOutputProcessor()
        
        # Trading parameters
        self.initial_balance = initial_balance or self.trading_config.ACCOUNT_SIZE
        self.current_balance = self.initial_balance
        self.transaction_cost = transaction_cost
        self.leverage = leverage
        
        # State tracking
        self.current_step = 0
        self.max_steps = len(self.data) - 1
        self.position_type = PositionType.NONE
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.position_value = 0.0
        self.unrealized_pnl = 0.0
        
        # Risk management
        self.consecutive_losses = 0
        self.current_risk_per_trade = self.trading_config.DEFAULT_RISK_PER_TRADE
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance
        
        # Episode tracking
        self.trade_history = []
        self.reward_history = []
        self.total_trades = 0
        self.winning_trades = 0
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(TradingAction))
        self._setup_observation_space()
        
        logger.info(f"Initialized HybridTradingEnvironment with {len(self.data)} steps")
    
    def _setup_observation_space(self):
        """Setup the observation space combining TFT outputs and market state"""
        
        # TFT outputs (processed by TFTOutputProcessor)
        tft_features = 9  # long_prob, short_prob, confidence, uncertainty, quantiles, attention
        
        # Market context features
        market_features = 15  # OHLCV, VWAP, volume profile, volatility measures
        
        # Position state features
        position_features = 8  # position info, unrealized P&L, risk state
        
        # Session and time features
        session_features = 5  # session type, time features
        
        total_features = tft_features + market_features + position_features + session_features
        
        # All features are continuous and normalized to reasonable ranges
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(total_features,),
            dtype=np.float32
        )
        
        logger.info(f"Observation space: {total_features} features")
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment for a new episode"""
        super().reset(seed=seed)
        
        # Reset state
        self.current_step = config.tft.LOOKBACK_PERIODS  # Start after lookback period
        self.current_balance = self.initial_balance
        self.position_type = PositionType.NONE
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.position_value = 0.0
        self.unrealized_pnl = 0.0
        
        # Reset risk management
        self.consecutive_losses = 0
        self.current_risk_per_trade = self.trading_config.DEFAULT_RISK_PER_TRADE
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance
        
        # Reset tracking
        self.trade_history = []
        self.reward_history = []
        self.total_trades = 0
        self.winning_trades = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        
        # Validate action
        if action not in range(len(TradingAction)):
            action = TradingAction.HOLD
        
        action = TradingAction(action)
        
        # Get current market data
        current_data = self.data.iloc[self.current_step]
        current_price = current_data['close']
        
        # Execute action
        reward = self._execute_action(action, current_data)
        
        # Update position and calculate unrealized P&L
        self._update_position_status(current_price)
        
        # Update risk management
        self._update_risk_management()
        
        # Check for termination conditions
        terminated = self._check_termination_conditions()
        truncated = self.current_step >= self.max_steps - 1
        
        # Move to next step
        self.current_step += 1
        
        # Get new observation
        observation = self._get_observation()
        info = self._get_info()
        
        # Track reward
        self.reward_history.append(reward)
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, action: TradingAction, current_data: pd.Series) -> float:
        """Execute the trading action and return reward"""
        
        current_price = current_data['close']
        reward = 0.0
        
        # Get TFT predictions for current state
        tft_output = self._get_tft_prediction(current_data)
        
        if action == TradingAction.HOLD:
            # No action - small penalty for inaction when good opportunities exist
            if tft_output and (tft_output.get('pattern_confidence', 0) > 0.8):
                reward = -1.0  # Opportunity cost
            else:
                reward = 0.0
        
        elif action == TradingAction.LONG:
            reward = self._execute_long_trade(current_price, tft_output)
        
        elif action == TradingAction.SHORT:
            reward = self._execute_short_trade(current_price, tft_output)
        
        elif action == TradingAction.CLOSE:
            reward = self._close_position(current_price)
        
        elif action == TradingAction.ADJUST_STOP:
            reward = self._adjust_stop_loss(current_price)
        
        elif action == TradingAction.REDUCE_POSITION:
            reward = self._reduce_position(current_price)
        
        return reward
    
    def _execute_long_trade(self, current_price: float, tft_output: Dict) -> float:
        """Execute a long trade"""
        
        # Can't open new position if already in a position
        if self.position_type != PositionType.NONE:
            return -5.0  # Penalty for invalid action
        
        # Check session restrictions
        if self._is_asia_session():
            return -10.0  # Heavy penalty for trading in Asia session
        
        # Validate TFT signals
        if tft_output:
            validation = self.tft_processor.validate_2_to_1_trade_potential(
                tft_output, current_price
            )
            
            if not validation['overall_long_valid']:
                return -5.0  # Penalty for ignoring TFT signals
        
        # Calculate position size based on risk management
        risk_amount = self.current_balance * self.current_risk_per_trade
        
        # For 2:1 RR, we need to determine stop loss distance
        # Using ATR or recent volatility as a proxy for stop distance
        atr = self.data.iloc[self.current_step].get('atr_5', current_price * 0.01)
        stop_distance = max(atr, current_price * 0.005)  # Minimum 0.5% stop
        
        # Check if 2:1 RR is achievable
        take_profit_distance = stop_distance * self.trading_config.TARGET_RR_RATIO
        
        # Position sizing: risk_amount = position_size * stop_distance
        position_size = risk_amount / stop_distance
        
        # Apply leverage and check limits
        position_size *= self.leverage
        position_size = min(position_size, self.trading_config.MAX_POSITION_SIZE)
        position_size = max(position_size, self.trading_config.MIN_POSITION_SIZE)
        
        # Execute trade
        self.position_type = PositionType.LONG
        self.position_size = position_size
        self.entry_price = current_price
        self.stop_loss = current_price - stop_distance
        self.take_profit = current_price + take_profit_distance
        self.position_value = position_size * current_price
        
        # Transaction costs
        transaction_cost = self.transaction_cost * position_size
        self.current_balance -= transaction_cost
        
        # Track trade
        self.total_trades += 1
        
        # Reward calculation
        reward = 0.0  # Neutral for opening position
        
        # Bonus for following TFT signals
        if tft_output and tft_output.get('long_probability', 0) > 0.7:
            reward += self.rl_config.PATTERN_ALIGNMENT_BONUS
        
        # Bonus for good risk management
        if position_size <= self.trading_config.MAX_POSITION_SIZE * 0.8:
            reward += self.rl_config.RISK_COMPLIANCE_BONUS
        
        logger.debug(f"Opened LONG position: size={position_size:.2f}, entry={current_price:.2f}, "
                    f"stop={self.stop_loss:.2f}, target={self.take_profit:.2f}")
        
        return reward
    
    def _execute_short_trade(self, current_price: float, tft_output: Dict) -> float:
        """Execute a short trade"""
        
        # Can't open new position if already in a position
        if self.position_type != PositionType.NONE:
            return -5.0  # Penalty for invalid action
        
        # Check session restrictions
        if self._is_asia_session():
            return -10.0  # Heavy penalty for trading in Asia session
        
        # Validate TFT signals
        if tft_output:
            validation = self.tft_processor.validate_2_to_1_trade_potential(
                tft_output, current_price
            )
            
            if not validation['overall_short_valid']:
                return -5.0  # Penalty for ignoring TFT signals
        
        # Calculate position size (similar to long trade)
        risk_amount = self.current_balance * self.current_risk_per_trade
        atr = self.data.iloc[self.current_step].get('atr_5', current_price * 0.01)
        stop_distance = max(atr, current_price * 0.005)
        
        take_profit_distance = stop_distance * self.trading_config.TARGET_RR_RATIO
        position_size = (risk_amount / stop_distance) * self.leverage
        position_size = min(position_size, self.trading_config.MAX_POSITION_SIZE)
        position_size = max(position_size, self.trading_config.MIN_POSITION_SIZE)
        
        # Execute trade
        self.position_type = PositionType.SHORT
        self.position_size = position_size
        self.entry_price = current_price
        self.stop_loss = current_price + stop_distance
        self.take_profit = current_price - take_profit_distance
        self.position_value = position_size * current_price
        
        # Transaction costs
        transaction_cost = self.transaction_cost * position_size
        self.current_balance -= transaction_cost
        
        # Track trade
        self.total_trades += 1
        
        # Reward calculation
        reward = 0.0
        
        # Bonus for following TFT signals
        if tft_output and tft_output.get('short_probability', 0) > 0.7:
            reward += self.rl_config.PATTERN_ALIGNMENT_BONUS
        
        # Bonus for good risk management
        if position_size <= self.trading_config.MAX_POSITION_SIZE * 0.8:
            reward += self.rl_config.RISK_COMPLIANCE_BONUS
        
        logger.debug(f"Opened SHORT position: size={position_size:.2f}, entry={current_price:.2f}, "
                    f"stop={self.stop_loss:.2f}, target={self.take_profit:.2f}")
        
        return reward
    
    def _close_position(self, current_price: float) -> float:
        """Close current position"""
        
        if self.position_type == PositionType.NONE:
            return -2.0  # Penalty for trying to close non-existent position
        
        # Calculate P&L
        if self.position_type == PositionType.LONG:
            pnl = (current_price - self.entry_price) * self.position_size
        else:  # SHORT
            pnl = (self.entry_price - current_price) * self.position_size
        
        # Apply transaction costs
        transaction_cost = self.transaction_cost * self.position_size
        net_pnl = pnl - transaction_cost
        
        # Update balance
        self.current_balance += net_pnl
        
        # Track trade outcome
        trade_record = {
            'entry_price': self.entry_price,
            'exit_price': current_price,
            'position_type': self.position_type.name,
            'position_size': self.position_size,
            'pnl': net_pnl,
            'risk_reward_ratio': abs(pnl) / (abs(self.entry_price - self.stop_loss) * self.position_size) if abs(self.entry_price - self.stop_loss) > 0 else 0,
            'step': self.current_step
        }
        self.trade_history.append(trade_record)
        
        # Update risk management
        if net_pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
            # Reset risk if in martingale recovery
            if self.current_risk_per_trade < self.trading_config.DEFAULT_RISK_PER_TRADE:
                self.current_risk_per_trade = min(
                    self.current_risk_per_trade * 2,
                    self.trading_config.DEFAULT_RISK_PER_TRADE
                )
        else:
            self.consecutive_losses += 1
            # Apply reverse martingale
            if self.trading_config.ENABLE_REVERSE_MARTINGALE:
                self.current_risk_per_trade *= self.trading_config.RISK_REDUCTION_FACTOR
        
        # Calculate reward
        risk_amount = self.initial_balance * self.trading_config.DEFAULT_RISK_PER_TRADE
        reward = (net_pnl / risk_amount) * self.rl_config.TRADE_PNL_WEIGHT
        
        # Additional rewards/penalties
        if trade_record['risk_reward_ratio'] >= self.trading_config.TARGET_RR_RATIO:
            reward += self.rl_config.TIMING_BONUS
        
        # Reset position
        self.position_type = PositionType.NONE
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.position_value = 0.0
        self.unrealized_pnl = 0.0
        
        logger.debug(f"Closed position: P&L={net_pnl:.2f}, RR={trade_record['risk_reward_ratio']:.2f}")
        
        return reward
    
    def _adjust_stop_loss(self, current_price: float) -> float:
        """Adjust stop loss (trailing stop or breakeven)"""
        
        if self.position_type == PositionType.NONE:
            return -1.0  # Small penalty for invalid action
        
        # Check if we're at 50% of profit target (as per plan)
        if self.position_type == PositionType.LONG:
            profit_target_distance = self.take_profit - self.entry_price
            current_profit = current_price - self.entry_price
            
            if current_profit >= profit_target_distance * 0.5:
                # Move stop to breakeven or trail
                new_stop = max(self.stop_loss, self.entry_price)
                if new_stop > self.stop_loss:
                    self.stop_loss = new_stop
                    return 2.0  # Reward for good risk management
        
        else:  # SHORT
            profit_target_distance = self.entry_price - self.take_profit
            current_profit = self.entry_price - current_price
            
            if current_profit >= profit_target_distance * 0.5:
                new_stop = min(self.stop_loss, self.entry_price)
                if new_stop < self.stop_loss:
                    self.stop_loss = new_stop
                    return 2.0  # Reward for good risk management
        
        return 0.0  # No change
    
    def _reduce_position(self, current_price: float) -> float:
        """Reduce position size (partial profit taking)"""
        
        if self.position_type == PositionType.NONE or self.position_size <= self.trading_config.MIN_POSITION_SIZE:
            return -1.0  # Penalty for invalid action
        
        # Reduce position by 50%
        reduction_size = self.position_size * 0.5
        
        # Calculate P&L on reduced portion
        if self.position_type == PositionType.LONG:
            pnl = (current_price - self.entry_price) * reduction_size
        else:  # SHORT
            pnl = (self.entry_price - current_price) * reduction_size
        
        # Apply transaction costs
        transaction_cost = self.transaction_cost * reduction_size
        net_pnl = pnl - transaction_cost
        
        # Update balance and position
        self.current_balance += net_pnl
        self.position_size -= reduction_size
        self.position_value = self.position_size * current_price
        
        # Calculate reward
        risk_amount = self.initial_balance * self.trading_config.DEFAULT_RISK_PER_TRADE
        reward = (net_pnl / risk_amount) * self.rl_config.TRADE_PNL_WEIGHT * 0.5  # Partial reward
        
        return reward
    
    def _update_position_status(self, current_price: float):
        """Update position status and check for stop loss/take profit"""
        
        if self.position_type == PositionType.NONE:
            return
        
        # Calculate unrealized P&L
        if self.position_type == PositionType.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.position_size
            
            # Check stop loss and take profit
            if current_price <= self.stop_loss:
                self._close_position(self.stop_loss)
            elif current_price >= self.take_profit:
                self._close_position(self.take_profit)
        
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.position_size
            
            # Check stop loss and take profit
            if current_price >= self.stop_loss:
                self._close_position(self.stop_loss)
            elif current_price <= self.take_profit:
                self._close_position(self.take_profit)
        
        # Update position value
        self.position_value = self.position_size * current_price
    
    def _update_risk_management(self):
        """Update risk management metrics"""
        
        # Update peak balance and drawdown
        total_equity = self.current_balance + self.unrealized_pnl
        
        if total_equity > self.peak_balance:
            self.peak_balance = total_equity
        
        current_drawdown = (self.peak_balance - total_equity) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def _check_termination_conditions(self) -> bool:
        """Check if episode should terminate"""
        
        # Check drawdown limit
        if self.max_drawdown >= self.trading_config.MAX_DRAWDOWN_THRESHOLD:
            logger.warning(f"Episode terminated due to excessive drawdown: {self.max_drawdown:.2%}")
            return True
        
        # Check if balance is too low
        if self.current_balance <= self.initial_balance * 0.5:
            logger.warning(f"Episode terminated due to low balance: ${self.current_balance:.2f}")
            return True
        
        return False
    
    def _get_tft_prediction(self, current_data: pd.Series) -> Optional[Dict]:
        """Get TFT prediction for current market state"""
        
        if self.tft_model is None:
            return None
        
        try:
            # Get recent data window for TFT
            start_idx = max(0, self.current_step - config.tft.LOOKBACK_PERIODS)
            end_idx = self.current_step + 1
            window_data = self.data.iloc[start_idx:end_idx]
            
            # Make TFT prediction
            tft_predictions = self.tft_model.predict(window_data, return_attention=True)
            
            # Process for RL consumption
            tft_output = self.tft_processor.process_for_rl(tft_predictions, current_data)
            
            return tft_output
            
        except Exception as e:
            logger.warning(f"Could not get TFT prediction: {str(e)}")
            return None
    
    def _is_asia_session(self) -> bool:
        """Check if current time is in Asia session"""
        
        current_data = self.data.iloc[self.current_step]
        
        # Check if is_asia_session column exists
        if 'is_asia_session' in current_data:
            return bool(current_data['is_asia_session'])
        
        # Fallback to hour-based check
        if 'hour_ny' in current_data:
            hour_ny = current_data['hour_ny']
            return (hour_ny >= self.trading_config.ASIA_SESSION_START) or \
                   (hour_ny <= self.trading_config.ASIA_SESSION_END)
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation for the RL agent"""
        
        current_data = self.data.iloc[self.current_step]
        observation = []
        
        # Get TFT features
        tft_output = self._get_tft_prediction(current_data)
        if tft_output:
            observation.extend([
                tft_output.get('long_probability', 0.5),
                tft_output.get('short_probability', 0.5),
                tft_output.get('pattern_confidence', 0.0),
                tft_output.get('price_uncertainty', 1.0),
                tft_output.get('quantile_p10', current_data['close']) / current_data['close'] - 1,
                tft_output.get('quantile_p50', current_data['close']) / current_data['close'] - 1,
                tft_output.get('quantile_p90', current_data['close']) / current_data['close'] - 1,
                tft_output.get('uncertainty_spread', 0.5),
                tft_output.get('attention_focus', 0.5)
            ])
        else:
            # Default TFT features if not available
            observation.extend([0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5])
        
        # Market context features (normalized)
        price_norm = current_data['close'] / current_data.get('price_ma_20', current_data['close'])
        observation.extend([
            (current_data['close'] - current_data['open']) / current_data['open'],  # Price change
            current_data.get('volume_ratio', 1.0),  # Volume ratio
            current_data.get('atr_5', 0.01) / current_data['close'],  # Volatility
            current_data.get('vwap_dist_daily', 0.0),  # VWAP distance
            current_data.get('bb_position', 0.5),  # Bollinger Band position
            current_data.get('momentum_5', 0.0),  # Short-term momentum
            current_data.get('momentum_20', 0.0),  # Long-term momentum
            price_norm - 1,  # Price vs MA
            current_data.get('realized_volatility', 0.2),  # Realized volatility
            current_data.get('volume_profile_percentile', 0.5),  # Volume profile position
            min(max((current_data.get('poc', current_data['close']) / current_data['close'] - 1), -0.1), 0.1),  # POC distance
            min(max((current_data.get('vah', current_data['close']) / current_data['close'] - 1), -0.1), 0.1),  # VAH distance
            min(max((current_data.get('val', current_data['close']) / current_data['close'] - 1), -0.1), 0.1),  # VAL distance
            current_data.get('intrabar_range_pct', 0.01),  # Intrabar range
            current_data.get('bb_width', 0.05)  # Bollinger Band width
        ])
        
        # Position state features
        observation.extend([
            float(self.position_type),  # Position type
            self.position_size / self.trading_config.MAX_POSITION_SIZE,  # Normalized position size
            (self.entry_price / current_data['close'] - 1) if self.entry_price > 0 else 0,  # Entry price distance
            self.unrealized_pnl / (self.current_balance * 0.01),  # Unrealized P&L vs 1% risk
            (self.current_balance / self.initial_balance - 1),  # Account performance
            self.max_drawdown,  # Max drawdown
            self.current_risk_per_trade / self.trading_config.DEFAULT_RISK_PER_TRADE,  # Risk adjustment
            float(self.consecutive_losses) / 5.0  # Consecutive losses (normalized)
        ])
        
        # Session features
        observation.extend([
            current_data.get('session_type', 0) / 3.0,  # Session type
            current_data.get('hour', 12) / 24.0,  # Hour of day
            current_data.get('day_of_week', 2) / 6.0,  # Day of week
            float(self._is_asia_session()),  # Asia session flag
            current_data.get('minutes_from_open', 240) / 480.0  # Minutes from market open
        ])
        
        # Ensure observation is the correct shape and convert to float32
        observation = np.array(observation, dtype=np.float32)
        
        # Clip extreme values
        observation = np.clip(observation, -10.0, 10.0)
        
        return observation
    
    def _get_info(self) -> Dict:
        """Get additional information about the current state"""
        
        current_data = self.data.iloc[self.current_step]
        
        info = {
            'step': self.current_step,
            'current_price': float(current_data['close']),
            'balance': self.current_balance,
            'position_type': self.position_type.name,
            'position_size': self.position_size,
            'unrealized_pnl': self.unrealized_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'max_drawdown': self.max_drawdown,
            'current_risk': self.current_risk_per_trade,
            'consecutive_losses': self.consecutive_losses,
            'is_asia_session': self._is_asia_session()
        }
        
        return info
    
    def render(self, mode='human'):
        """Render the environment state"""
        
        current_data = self.data.iloc[self.current_step]
        info = self._get_info()
        
        print(f"\n=== Step {self.current_step} ===")
        print(f"Price: ${current_data['close']:.2f}")
        print(f"Balance: ${info['balance']:.2f}")
        print(f"Position: {info['position_type']} (Size: {info['position_size']:.2f})")
        print(f"Unrealized P&L: ${info['unrealized_pnl']:.2f}")
        print(f"Trades: {info['total_trades']} (Win Rate: {info['win_rate']:.1%})")
        print(f"Max Drawdown: {info['max_drawdown']:.1%}")
        print(f"Risk Level: {info['current_risk']:.1%}")
        
    def get_trading_metrics(self) -> Dict:
        """Get comprehensive trading performance metrics"""
        
        if not self.trade_history:
            return {}
        
        trades_df = pd.DataFrame(self.trade_history)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        # P&L metrics
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        returns = [trade['pnl'] / self.initial_balance for trade in self.trade_history]
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_return': (self.current_balance / self.initial_balance - 1),
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss < 0 and losing_trades > 0 else 0,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0,
            'avg_rr_ratio': trades_df['risk_reward_ratio'].mean() if 'risk_reward_ratio' in trades_df else 0
        }
        
        return metrics 