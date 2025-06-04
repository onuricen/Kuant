"""
Data loading and preprocessing module for Nasdaq Futures data
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timezone
import glob

from ..config import config

logger = logging.getLogger(__name__)

class NasdaqFuturesDataLoader:
    """Data loader for Nasdaq Futures OHLCV data"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or config.data.DATA_DIR
        self.processed_dir = config.data.PROCESSED_DATA_DIR
        self.condition_data = self._load_condition_data()
        
    def _load_condition_data(self) -> Dict:
        """Load condition data from condition.json"""
        condition_file = self.data_dir / "condition.json"
        if condition_file.exists():
            with open(condition_file, 'r') as f:
                conditions = json.load(f)
            return {item['date']: item for item in conditions}
        return {}
    
    def get_available_dates(self) -> List[str]:
        """Get list of available dates from condition data"""
        available_dates = [
            date for date, info in self.condition_data.items() 
            if info.get('condition') == 'available'
        ]
        return sorted(available_dates)
    
    def load_single_day_data(self, date: str) -> Optional[pd.DataFrame]:
        """Load data for a single day"""
        pattern = f"glbx-mdp3-{date.replace('-', '')}.ohlcv-1m.csv"
        file_path = self.data_dir / pattern
        
        if not file_path.exists():
            logger.warning(f"Data file not found for date {date}: {file_path}")
            return None
        
        try:
            df = pd.read_csv(file_path)
            
            # Filter for main Nasdaq Futures contracts (NQM0, NQU0, NQZ0, etc.)
            # instrument_id = 6641 appears to be the main front month contract
            # We'll also include any NQ contracts that start with NQ and are the front month
            main_contract_filter = (
                (df['instrument_id'] == 6641) |  # Main contract ID
                (df['symbol'].str.startswith('NQ') & 
                 df['symbol'].str.len() == 4 & 
                 ~df['symbol'].str.contains('-'))  # NQ front month contracts, exclude spreads
            )
            
            df = df[main_contract_filter].copy()
            
            if df.empty:
                logger.warning(f"No main contract data found for date {date}")
                return None
            
            # Convert timestamp to datetime with UTC timezone handling
            df['timestamp'] = pd.to_datetime(df['ts_event'], utc=True)
            
            # Remove timezone for easier handling (convert to ET later if needed)
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            
            # Sort by timestamp and remove duplicates (keep first occurrence)
            df = df.sort_values('timestamp')
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
            
            # Set timestamp as index
            df = df.set_index('timestamp')
            
            # Select OHLCV columns and add symbol info for tracking
            df = df[['open', 'high', 'low', 'close', 'volume', 'symbol', 'instrument_id']].copy()
            
            # Data validation and cleaning
            df = self._validate_and_clean_data(df, date)
            
            # Keep only OHLCV for final output (drop symbol and instrument_id after validation)
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data for date {date}: {str(e)}")
            return None
    
    def _validate_and_clean_data(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """Validate and clean the data"""
        initial_rows = len(df)
        
        # Remove rows with zero or negative prices
        price_columns = ['open', 'high', 'low', 'close']
        df = df[(df[price_columns] > 0).all(axis=1)]
        
        # Remove rows with zero volume (optional - some strategies might use these)
        if config.data.MIN_VOLUME_THRESHOLD > 0:
            df = df[df['volume'] >= config.data.MIN_VOLUME_THRESHOLD]
        
        # Remove outliers based on price deviation (only if we have enough data)
        if len(df) >= 20:
            for price_col in price_columns:
                rolling_mean = df[price_col].rolling(20, center=True).mean()
                deviation = np.abs(df[price_col] - rolling_mean) / rolling_mean
                # Only apply outlier filter if we have valid rolling mean
                valid_mask = ~rolling_mean.isna()
                outlier_mask = (deviation <= config.data.MAX_PRICE_DEVIATION) | ~valid_mask
                df = df[outlier_mask]
        
        # Ensure OHLC relationships are correct
        df = df[
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
        ]
        
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} invalid rows from {date} data ({(removed_rows/initial_rows)*100:.1f}%)")
        
        # Log symbol distribution for monitoring
        if 'symbol' in df.columns:
            symbol_counts = df['symbol'].value_counts()
            logger.debug(f"Symbol distribution for {date}: {symbol_counts.to_dict()}")
        
        return df
    
    def load_data_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load data for a date range"""
        available_dates = self.get_available_dates()
        
        # Filter dates within range
        target_dates = [
            date for date in available_dates 
            if start_date <= date <= end_date
        ]
        
        if not target_dates:
            raise ValueError(f"No data available between {start_date} and {end_date}")
        
        logger.info(f"Loading data for {len(target_dates)} dates from {start_date} to {end_date}")
        
        data_frames = []
        for date in target_dates:
            df = self.load_single_day_data(date)
            if df is not None and not df.empty:
                data_frames.append(df)
        
        if not data_frames:
            raise ValueError(f"No valid data found in date range {start_date} to {end_date}")
        
        # Combine all dataframes
        combined_df = pd.concat(data_frames, axis=0)
        combined_df = combined_df.sort_index()
        
        # Remove duplicates if any
        combined_df = combined_df[~combined_df.index.duplicated()]
        
        logger.info(f"Loaded {len(combined_df)} total records")
        
        return combined_df
    
    def load_all_data(self) -> pd.DataFrame:
        """Load all available data"""
        available_dates = self.get_available_dates()
        
        if not available_dates:
            raise ValueError("No data available")
        
        start_date = available_dates[0]
        end_date = available_dates[-1]
        
        return self.load_data_range(start_date, end_date)
    
    def resample_to_5min(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample 1-minute data to 5-minute bars"""
        resampled = df.resample('5T').agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        logger.info(f"Resampled from {len(df)} 1-min bars to {len(resampled)} 5-min bars")
        
        return resampled
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data to parquet format"""
        output_path = self.processed_dir / f"{filename}.parquet"
        df.to_parquet(output_path)
        logger.info(f"Saved processed data to {output_path}")
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """Load processed data from parquet format"""
        input_path = self.processed_dir / f"{filename}.parquet"
        if input_path.exists():
            df = pd.read_parquet(input_path)
            logger.info(f"Loaded processed data from {input_path}")
            return df
        else:
            raise FileNotFoundError(f"Processed data not found: {input_path}")
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics of the data"""
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df.index.min().strftime('%Y-%m-%d %H:%M:%S'),
                'end': df.index.max().strftime('%Y-%m-%d %H:%M:%S')
            },
            'price_statistics': {
                'min_price': float(df['low'].min()),
                'max_price': float(df['high'].max()),
                'mean_close': float(df['close'].mean()),
                'std_close': float(df['close'].std())
            },
            'volume_statistics': {
                'total_volume': int(df['volume'].sum()),
                'mean_volume': float(df['volume'].mean()),
                'std_volume': float(df['volume'].std())
            },
            'missing_data': {
                'total_missing': int(df.isnull().sum().sum()),
                'missing_by_column': df.isnull().sum().to_dict()
            }
        }
        
        return summary

# Convenience function for easy data loading
def load_nasdaq_futures_data(start_date: str = None, end_date: str = None, 
                           resample_5min: bool = True) -> pd.DataFrame:
    """
    Convenience function to load Nasdaq Futures data
    
    Args:
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional) 
        resample_5min: Whether to resample to 5-minute bars
    
    Returns:
        DataFrame with OHLCV data
    """
    loader = NasdaqFuturesDataLoader()
    
    if start_date and end_date:
        df = loader.load_data_range(start_date, end_date)
    else:
        df = loader.load_all_data()
    
    if resample_5min:
        df = loader.resample_to_5min(df)
    
    return df 