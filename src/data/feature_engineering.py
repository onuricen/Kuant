"""
Feature engineering module for creating trading features from OHLCV data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timezone
import warnings

from ..config import config

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering for Nasdaq Futures trading data"""
    
    def __init__(self):
        self.vwap_periods = config.data.VWAP_PERIODS
        self.volume_window = config.data.VOLUME_WINDOW
        
    def calculate_vwap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP features for multiple timeframes"""
        df = df.copy()
        
        # Calculate typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate cumulative volume and volume * typical price
        df['cum_volume'] = df['volume'].cumsum()
        df['cum_vol_price'] = (df['volume'] * df['typical_price']).cumsum()
        
        # VWAP for different periods
        for period_name, periods in self.vwap_periods.items():
            # Rolling VWAP calculation
            vol_sum = df['volume'].rolling(window=periods, min_periods=1).sum()
            vol_price_sum = (df['volume'] * df['typical_price']).rolling(window=periods, min_periods=1).sum()
            
            vwap_col = f'vwap_{period_name}'
            dist_col = f'vwap_dist_{period_name}'
            
            df[vwap_col] = vol_price_sum / vol_sum
            df[dist_col] = (df['close'] - df[vwap_col]) / df[vwap_col]
            
            # VWAP slope (momentum)
            df[f'vwap_slope_{period_name}'] = df[vwap_col].diff(5) / df[vwap_col].shift(5)
        
        # Remove intermediate calculation columns
        df.drop(['typical_price', 'cum_volume', 'cum_vol_price'], axis=1, inplace=True)
        
        logger.info("Calculated VWAP features for all timeframes")
        return df
    
    def calculate_volume_profile_features(self, df: pd.DataFrame, lookback_days: int = 7) -> pd.DataFrame:
        """Calculate volume profile features"""
        df = df.copy()
        
        # Calculate volume profile for each bar based on lookback period
        lookback_periods = lookback_days * 288  # 288 5-min periods per day
        
        poc_list = []
        vah_list = []
        val_list = []
        vol_percentile_list = []
        
        for i in range(len(df)):
            start_idx = max(0, i - lookback_periods)
            window_data = df.iloc[start_idx:i+1]
            
            if len(window_data) < 10:  # Need minimum data
                poc_list.append(np.nan)
                vah_list.append(np.nan)
                val_list.append(np.nan)
                vol_percentile_list.append(0.5)
                continue
            
            # Create price bins
            price_range = window_data['high'].max() - window_data['low'].min()
            n_bins = min(50, max(10, int(price_range / 0.25)))  # Adaptive binning
            
            price_bins = np.linspace(
                window_data['low'].min(), 
                window_data['high'].max(), 
                n_bins
            )
            
            # Calculate volume at each price level
            volume_at_price = np.zeros(len(price_bins) - 1)
            
            for _, row in window_data.iterrows():
                # Distribute volume across price range of the bar
                low, high, volume = row['low'], row['high'], row['volume']
                
                # Find bins that overlap with this bar's price range
                bin_indices = np.where(
                    (price_bins[:-1] <= high) & (price_bins[1:] >= low)
                )[0]
                
                if len(bin_indices) > 0:
                    # Distribute volume equally across overlapping bins
                    volume_per_bin = volume / len(bin_indices)
                    volume_at_price[bin_indices] += volume_per_bin
            
            # Point of Control (POC) - price level with highest volume
            if len(volume_at_price) > 0 and volume_at_price.sum() > 0:
                poc_idx = np.argmax(volume_at_price)
                poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
                
                # Value Area High (VAH) and Low (VAL) - 70% of volume
                total_volume = volume_at_price.sum()
                target_volume = total_volume * 0.7
                
                # Sort bins by volume to find value area
                sorted_indices = np.argsort(volume_at_price)[::-1]
                cumulative_volume = 0
                value_area_bins = []
                
                for idx in sorted_indices:
                    cumulative_volume += volume_at_price[idx]
                    value_area_bins.append(idx)
                    if cumulative_volume >= target_volume:
                        break
                
                if value_area_bins:
                    vah_price = price_bins[max(value_area_bins) + 1]
                    val_price = price_bins[min(value_area_bins)]
                else:
                    vah_price = poc_price
                    val_price = poc_price
                
                # Current price percentile in volume distribution
                current_price = df.iloc[i]['close']
                current_bin = np.digitize(current_price, price_bins) - 1
                current_bin = max(0, min(current_bin, len(volume_at_price) - 1))
                
                # Calculate percentile based on volume below current price
                volume_below = volume_at_price[:current_bin + 1].sum()
                vol_percentile = volume_below / total_volume if total_volume > 0 else 0.5
                
            else:
                poc_price = df.iloc[i]['close']
                vah_price = poc_price
                val_price = poc_price
                vol_percentile = 0.5
            
            poc_list.append(poc_price)
            vah_list.append(vah_price)
            val_list.append(val_price)
            vol_percentile_list.append(vol_percentile)
        
        df['poc'] = poc_list
        df['vah'] = vah_list
        df['val'] = val_list
        df['volume_profile_percentile'] = vol_percentile_list
        
        # Forward fill any NaN values
        df[['poc', 'vah', 'val', 'volume_profile_percentile']] = df[['poc', 'vah', 'val', 'volume_profile_percentile']].fillna(method='ffill')
        
        logger.info("Calculated volume profile features")
        return df
    
    def calculate_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price action features"""
        df = df.copy()
        
        # Basic price features
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change()
        df['intrabar_range'] = df['high'] - df['low']
        df['intrabar_range_pct'] = df['intrabar_range'] / df['close']
        
        # Volatility measures
        df['atr_5'] = self._calculate_atr(df, 5)
        df['atr_20'] = self._calculate_atr(df, 20)
        df['realized_volatility'] = df['price_change_pct'].rolling(20).std() * np.sqrt(288)  # Annualized
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(self.volume_window).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_change'] = df['volume'].pct_change()
        
        # Price momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            df[f'price_ma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_distance_ma_{period}'] = (df['close'] - df[f'price_ma_{period}']) / df[f'price_ma_{period}']
        
        # Bollinger Bands (20-period, 2 std)
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(bb_period).mean()
        bb_rolling_std = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_rolling_std * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_rolling_std * bb_std)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        logger.info("Calculated price action features")
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift())
        low_close_prev = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def calculate_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate session-based features"""
        df = df.copy()
        
        # Extract time features (assuming UTC timestamps)
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # NY timezone conversion for session detection
        # Convert to NY time (approximately UTC-4 in summer, UTC-5 in winter)
        # For simplicity, we'll use UTC-5 (EST)
        df['hour_ny'] = (df['hour'] - 5) % 24
        
        # Session identification
        df['is_asia_session'] = (
            (df['hour_ny'] >= config.trading.ASIA_SESSION_START) | 
            (df['hour_ny'] <= config.trading.ASIA_SESSION_END)
        )
        df['is_us_session'] = (df['hour_ny'] >= 9) & (df['hour_ny'] <= 16)
        df['is_europe_session'] = (df['hour_ny'] >= 3) & (df['hour_ny'] <= 11)
        
        # Session type encoding
        df['session_type'] = 0  # Default
        df.loc[df['is_asia_session'], 'session_type'] = 1
        df.loc[df['is_europe_session'], 'session_type'] = 2
        df.loc[df['is_us_session'], 'session_type'] = 3
        
        # Market open/close proximity
        df['minutes_from_open'] = np.abs((df['hour_ny'] * 60 + df.index.minute) - (9 * 60 + 30))
        df['minutes_from_close'] = np.abs((df['hour_ny'] * 60 + df.index.minute) - (16 * 60))
        
        logger.info("Calculated session features")
        return df
    
    def calculate_target_labels(self, df: pd.DataFrame, lookahead_periods: int = 10) -> pd.DataFrame:
        """Calculate target labels for 2:1 RR trades"""
        df = df.copy()
        
        # Calculate potential profit/loss for different scenarios
        future_highs = df['high'].rolling(window=lookahead_periods, min_periods=1).max().shift(-lookahead_periods)
        future_lows = df['low'].rolling(window=lookahead_periods, min_periods=1).min().shift(-lookahead_periods)
        future_closes = df['close'].shift(-lookahead_periods)
        
        # For each bar, calculate if a 2:1 RR trade would be successful
        df['can_long_2_to_1'] = 0
        df['can_short_2_to_1'] = 0
        df['optimal_long_rr'] = 0.0
        df['optimal_short_rr'] = 0.0
        
        for i in range(len(df) - lookahead_periods):
            current_price = df.iloc[i]['close']
            
            # Check long trade potential
            max_favorable = future_highs.iloc[i] - current_price
            max_adverse = current_price - future_lows.iloc[i]
            
            if max_adverse > 0:
                potential_rr_long = max_favorable / max_adverse
                df.iloc[i, df.columns.get_loc('optimal_long_rr')] = potential_rr_long
                
                if potential_rr_long >= config.trading.TARGET_RR_RATIO:
                    df.iloc[i, df.columns.get_loc('can_long_2_to_1')] = 1
            
            # Check short trade potential
            max_favorable = current_price - future_lows.iloc[i]
            max_adverse = future_highs.iloc[i] - current_price
            
            if max_adverse > 0:
                potential_rr_short = max_favorable / max_adverse
                df.iloc[i, df.columns.get_loc('optimal_short_rr')] = potential_rr_short
                
                if potential_rr_short >= config.trading.TARGET_RR_RATIO:
                    df.iloc[i, df.columns.get_loc('can_short_2_to_1')] = 1
        
        # Future price movements for quantile predictions
        for horizon in [1, 5, 15, 50]:
            df[f'future_return_{horizon}'] = df['close'].pct_change(horizon).shift(-horizon)
            df[f'future_price_{horizon}'] = df['close'].shift(-horizon)
        
        logger.info("Calculated target labels for 2:1 RR trades")
        return df
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features for ML model input"""
        df = df.copy()
        
        # Features to normalize
        price_features = [col for col in df.columns if any(x in col.lower() for x in ['price', 'vwap', 'poc', 'vah', 'val', 'bb_'])]
        ratio_features = [col for col in df.columns if any(x in col.lower() for x in ['ratio', 'pct', 'change', 'distance', 'momentum'])]
        volume_features = [col for col in df.columns if 'volume' in col.lower() and 'ratio' not in col.lower()]
        
        # Z-score normalization for price features
        for feature in price_features:
            if feature in df.columns:
                rolling_mean = df[feature].rolling(window=100, min_periods=20).mean()
                rolling_std = df[feature].rolling(window=100, min_periods=20).std()
                df[f'{feature}_normalized'] = (df[feature] - rolling_mean) / rolling_std
        
        # Clip extreme values for ratio features
        for feature in ratio_features:
            if feature in df.columns:
                df[f'{feature}_clipped'] = df[feature].clip(
                    lower=df[feature].quantile(0.01), 
                    upper=df[feature].quantile(0.99)
                )
        
        # Log normalization for volume features
        for feature in volume_features:
            if feature in df.columns:
                df[f'{feature}_log'] = np.log1p(df[feature])
        
        logger.info("Normalized features for ML model input")
        return df
    
    def create_complete_feature_set(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create complete feature set with all engineering steps"""
        logger.info("Starting complete feature engineering pipeline")
        
        # Apply all feature engineering steps
        df = self.calculate_vwap_features(df)
        df = self.calculate_volume_profile_features(df)
        df = self.calculate_price_action_features(df)
        df = self.calculate_session_features(df)
        df = self.calculate_target_labels(df)
        df = self.normalize_features(df)
        
        # Remove rows with insufficient data
        df = df.dropna()
        
        logger.info(f"Complete feature engineering finished. Final dataset shape: {df.shape}")
        
        return df
    
    def get_feature_columns(self) -> Dict[str, List[str]]:
        """Get organized list of feature columns for model input"""
        
        feature_groups = {
            'price_features': [
                'open', 'high', 'low', 'close',
                'price_change', 'price_change_pct', 'intrabar_range', 'intrabar_range_pct'
            ],
            'vwap_features': [
                'vwap_hourly', 'vwap_daily', 'vwap_weekly', 'vwap_monthly',
                'vwap_dist_hourly', 'vwap_dist_daily', 'vwap_dist_weekly', 'vwap_dist_monthly',
                'vwap_slope_hourly', 'vwap_slope_daily', 'vwap_slope_weekly', 'vwap_slope_monthly'
            ],
            'volume_profile_features': [
                'poc', 'vah', 'val', 'volume_profile_percentile'
            ],
            'volume_features': [
                'volume', 'volume_ma', 'volume_ratio', 'volume_change'
            ],
            'volatility_features': [
                'atr_5', 'atr_20', 'realized_volatility', 'bb_width', 'bb_position'
            ],
            'momentum_features': [
                'momentum_5', 'momentum_10', 'momentum_20',
                'price_distance_ma_5', 'price_distance_ma_10', 'price_distance_ma_20'
            ],
            'session_features': [
                'hour', 'day_of_week', 'session_type', 'is_asia_session',
                'minutes_from_open', 'minutes_from_close'
            ],
            'target_labels': [
                'can_long_2_to_1', 'can_short_2_to_1', 'optimal_long_rr', 'optimal_short_rr',
                'future_return_1', 'future_return_5', 'future_return_15', 'future_return_50'
            ]
        }
        
        return feature_groups 