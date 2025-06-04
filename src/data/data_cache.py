"""
Data caching system for processed features and datasets
Avoids recomputation when using the same raw data
"""

import pandas as pd
import numpy as np
import hashlib
import pickle
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import logging
from datetime import datetime
import joblib

logger = logging.getLogger(__name__)

class DataCache:
    """Intelligent caching system for processed trading data"""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save cache metadata: {e}")
    
    def _calculate_data_hash(self, df: pd.DataFrame, 
                           start_date: str = None, 
                           end_date: str = None,
                           additional_params: Dict = None) -> str:
        """Calculate a unique hash for the dataset and parameters"""
        
        # Create a signature from the data
        data_signature = {
            'shape': df.shape,
            'columns': sorted(df.columns.tolist()),
            'start_date': start_date,
            'end_date': end_date,
            'first_timestamp': str(df.index[0]) if len(df) > 0 else None,
            'last_timestamp': str(df.index[-1]) if len(df) > 0 else None,
            'data_checksum': hashlib.md5(
                pd.util.hash_pandas_object(df.head(100)).values
            ).hexdigest()  # Sample checksum for speed
        }
        
        # Add additional parameters
        if additional_params:
            data_signature.update(additional_params)
        
        # Create hash
        signature_str = json.dumps(data_signature, sort_keys=True)
        return hashlib.sha256(signature_str.encode()).hexdigest()[:16]
    
    def _get_cache_paths(self, cache_key: str) -> Dict[str, Path]:
        """Get cache file paths for a given key"""
        return {
            'raw_data': self.cache_dir / f"{cache_key}_raw.parquet",
            'processed_data': self.cache_dir / f"{cache_key}_processed.parquet",
            'feature_metadata': self.cache_dir / f"{cache_key}_features.json",
            'scalers': self.cache_dir / f"{cache_key}_scalers.pkl",
            'tft_datasets': self.cache_dir / f"{cache_key}_tft_datasets.pkl"
        }
    
    def check_cache_exists(self, df: pd.DataFrame, 
                          start_date: str = None, 
                          end_date: str = None,
                          feature_params: Dict = None) -> Tuple[bool, str]:
        """Check if processed data exists in cache"""
        
        cache_key = self._calculate_data_hash(df, start_date, end_date, feature_params)
        cache_paths = self._get_cache_paths(cache_key)
        
        # Check if all required files exist
        required_files = ['processed_data', 'feature_metadata']
        exists = all(cache_paths[file].exists() for file in required_files)
        
        if exists:
            # Check if cache is recent (optional freshness check)
            cache_info = self.metadata.get(cache_key, {})
            created_time = cache_info.get('created_at')
            if created_time:
                logger.info(f"Found cached data from {created_time}")
        
        return exists, cache_key
    
    def save_processed_data(self, 
                           raw_data: pd.DataFrame,
                           processed_data: pd.DataFrame,
                           cache_key: str,
                           feature_metadata: Dict = None,
                           scalers: Dict = None,
                           tft_datasets: Tuple = None,
                           start_date: str = None,
                           end_date: str = None) -> bool:
        """Save processed data to cache"""
        
        try:
            cache_paths = self._get_cache_paths(cache_key)
            
            logger.info(f"Saving processed data to cache: {cache_key}")
            
            # Save raw data (compressed)
            raw_data.to_parquet(cache_paths['raw_data'], compression='snappy')
            
            # Save processed data (compressed)
            processed_data.to_parquet(cache_paths['processed_data'], compression='snappy')
            
            # Save feature metadata
            if feature_metadata:
                with open(cache_paths['feature_metadata'], 'w') as f:
                    json.dump(feature_metadata, f, indent=2, default=str)
            
            # Save scalers and encoders
            if scalers:
                joblib.dump(scalers, cache_paths['scalers'])
            
            # Save TFT datasets
            if tft_datasets:
                joblib.dump(tft_datasets, cache_paths['tft_datasets'])
            
            # Update metadata
            self.metadata[cache_key] = {
                'created_at': datetime.now().isoformat(),
                'raw_data_shape': raw_data.shape,
                'processed_data_shape': processed_data.shape,
                'start_date': start_date,
                'end_date': end_date,
                'features_count': len(processed_data.columns),
                'file_sizes': {
                    name: path.stat().st_size if path.exists() else 0
                    for name, path in cache_paths.items()
                }
            }
            
            self._save_metadata()
            
            # Log cache size
            total_size_mb = sum(self.metadata[cache_key]['file_sizes'].values()) / 1024 / 1024
            logger.info(f"✅ Cached data saved: {total_size_mb:.1f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            return False
    
    def load_processed_data(self, cache_key: str) -> Dict[str, Any]:
        """Load processed data from cache"""
        
        try:
            cache_paths = self._get_cache_paths(cache_key)
            
            logger.info(f"Loading processed data from cache: {cache_key}")
            
            result = {}
            
            # Load raw data
            if cache_paths['raw_data'].exists():
                result['raw_data'] = pd.read_parquet(cache_paths['raw_data'])
            
            # Load processed data
            if cache_paths['processed_data'].exists():
                result['processed_data'] = pd.read_parquet(cache_paths['processed_data'])
            
            # Load feature metadata
            if cache_paths['feature_metadata'].exists():
                with open(cache_paths['feature_metadata'], 'r') as f:
                    result['feature_metadata'] = json.load(f)
            
            # Load scalers
            if cache_paths['scalers'].exists():
                result['scalers'] = joblib.load(cache_paths['scalers'])
            
            # Load TFT datasets
            if cache_paths['tft_datasets'].exists():
                result['tft_datasets'] = joblib.load(cache_paths['tft_datasets'])
            
            cache_info = self.metadata.get(cache_key, {})
            
            logger.info(f"✅ Loaded cached data: {result['processed_data'].shape} from {cache_info.get('created_at', 'unknown time')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return {}
    
    def clear_cache(self, cache_key: str = None):
        """Clear cache (specific key or all)"""
        
        if cache_key:
            # Clear specific cache
            cache_paths = self._get_cache_paths(cache_key)
            for path in cache_paths.values():
                if path.exists():
                    path.unlink()
            
            if cache_key in self.metadata:
                del self.metadata[cache_key]
                self._save_metadata()
            
            logger.info(f"Cleared cache for key: {cache_key}")
        else:
            # Clear all cache
            for file in self.cache_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            
            self.metadata = {}
            self._save_metadata()
            
            logger.info("Cleared all cache")
    
    def get_cache_info(self) -> Dict:
        """Get information about cached data"""
        
        total_size = 0
        cache_info = {}
        
        for cache_key, meta in self.metadata.items():
            size_mb = sum(meta.get('file_sizes', {}).values()) / 1024 / 1024
            total_size += size_mb
            
            cache_info[cache_key] = {
                'created_at': meta.get('created_at'),
                'data_shape': meta.get('processed_data_shape'),
                'size_mb': round(size_mb, 2),
                'date_range': f"{meta.get('start_date', 'N/A')} to {meta.get('end_date', 'N/A')}"
            }
        
        return {
            'total_cached_datasets': len(cache_info),
            'total_size_mb': round(total_size, 2),
            'datasets': cache_info
        }

class SmartDataLoader:
    """Data loader with intelligent caching"""
    
    def __init__(self, base_data_loader, cache_dir: Path = None):
        self.base_loader = base_data_loader
        self.cache = DataCache(cache_dir)
        
    def load_and_process_with_cache(self, 
                                   start_date: str = None, 
                                   end_date: str = None,
                                   feature_params: Dict = None,
                                   force_refresh: bool = False) -> Dict[str, Any]:
        """Load and process data with caching"""
        
        logger.info("=== Smart Data Loading with Cache ===")
        
        # Load raw data first
        if start_date and end_date:
            raw_data = self.base_loader.load_data_range(start_date, end_date)
        else:
            raw_data = self.base_loader.load_all_data()
        
        # Check cache
        cache_exists, cache_key = self.cache.check_cache_exists(
            raw_data, start_date, end_date, feature_params
        )
        
        if cache_exists and not force_refresh:
            logger.info("📦 Using cached processed data")
            cached_data = self.cache.load_processed_data(cache_key)
            
            if cached_data and 'processed_data' in cached_data:
                return cached_data
            else:
                logger.warning("Cache load failed, proceeding with fresh processing")
        
        # Process data fresh
        logger.info("🔄 Processing data fresh (no cache found or forced refresh)")
        
        # Resample data
        resampled_data = self.base_loader.resample_to_5min(raw_data)
        
        # Feature engineering
        from src.data.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        
        processed_data = fe.create_complete_feature_set(resampled_data)
        
        # Prepare feature metadata
        feature_metadata = {
            'total_features': len(processed_data.columns),
            'feature_types': {
                'categorical': [col for col in processed_data.columns 
                              if processed_data[col].dtype == 'object' or 
                                 processed_data[col].dtype.name == 'category'],
                'numerical': [col for col in processed_data.columns 
                            if processed_data[col].dtype in ['float64', 'int64']],
                'datetime': [col for col in processed_data.columns 
                           if processed_data[col].dtype == 'datetime64[ns]']
            },
            'processing_time': datetime.now().isoformat(),
            'data_range': {
                'start': str(processed_data.index[0]) if len(processed_data) > 0 else None,
                'end': str(processed_data.index[-1]) if len(processed_data) > 0 else None,
                'total_rows': len(processed_data)
            }
        }
        
        # Save to cache
        success = self.cache.save_processed_data(
            raw_data=raw_data,
            processed_data=processed_data,
            cache_key=cache_key,
            feature_metadata=feature_metadata,
            start_date=start_date,
            end_date=end_date
        )
        
        if success:
            logger.info("💾 Data cached for future use")
        
        return {
            'raw_data': raw_data,
            'processed_data': processed_data,
            'feature_metadata': feature_metadata,
            'cache_key': cache_key
        } 