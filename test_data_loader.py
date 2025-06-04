#!/usr/bin/env python3
"""
Test script to validate the updated data loader with actual Nasdaq Futures data
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.data_loader import NasdaqFuturesDataLoader, load_nasdaq_futures_data
from src.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_loader():
    """Test the data loader with actual data"""
    print("🧪 Testing Nasdaq Futures Data Loader")
    print("=" * 50)
    
    try:
        # Initialize data loader
        loader = NasdaqFuturesDataLoader()
        
        # Test 1: Check available dates
        print("\n📅 Testing available dates...")
        available_dates = loader.get_available_dates()
        print(f"✅ Found {len(available_dates)} available dates")
        print(f"   Date range: {available_dates[0]} to {available_dates[-1]}")
        
        # Test 2: Load single day data
        print("\n📊 Testing single day data loading...")
        test_date = available_dates[100] if len(available_dates) > 100 else available_dates[0]
        single_day_data = loader.load_single_day_data(test_date)
        
        if single_day_data is not None:
            print(f"✅ Successfully loaded data for {test_date}")
            print(f"   Shape: {single_day_data.shape}")
            print(f"   Columns: {single_day_data.columns.tolist()}")
            print(f"   Time range: {single_day_data.index.min()} to {single_day_data.index.max()}")
            print(f"   Sample data:")
            print(single_day_data.head())
        else:
            print(f"❌ Failed to load data for {test_date}")
            return False
        
        # Test 3: Load data range (small sample)
        print("\n📈 Testing data range loading...")
        start_date = available_dates[0]
        end_date = available_dates[min(4, len(available_dates)-1)]  # Load first 5 days
        
        range_data = loader.load_data_range(start_date, end_date)
        print(f"✅ Successfully loaded data range {start_date} to {end_date}")
        print(f"   Shape: {range_data.shape}")
        print(f"   Time range: {range_data.index.min()} to {range_data.index.max()}")
        
        # Test 4: 5-minute resampling
        print("\n⏰ Testing 5-minute resampling...")
        resampled_data = loader.resample_to_5min(range_data)
        print(f"✅ Successfully resampled data")
        print(f"   Original: {len(range_data)} 1-min bars")
        print(f"   Resampled: {len(resampled_data)} 5-min bars")
        print(f"   Sample resampled data:")
        print(resampled_data.head())
        
        # Test 5: Data summary
        print("\n📋 Testing data summary...")
        summary = loader.get_data_summary(resampled_data)
        print(f"✅ Data summary generated:")
        print(f"   Total records: {summary['total_records']}")
        print(f"   Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"   Price range: ${summary['price_statistics']['min_price']:.2f} - ${summary['price_statistics']['max_price']:.2f}")
        print(f"   Mean close: ${summary['price_statistics']['mean_close']:.2f}")
        print(f"   Total volume: {summary['volume_statistics']['total_volume']:,}")
        
        # Test 6: Convenience function
        print("\n🔧 Testing convenience function...")
        convenience_data = load_nasdaq_futures_data(start_date, end_date, resample_5min=True)
        print(f"✅ Convenience function works")
        print(f"   Shape: {convenience_data.shape}")
        
        # Test 7: Data validation checks
        print("\n✅ Testing data validation...")
        
        # Check for proper OHLCV structure
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        if all(col in convenience_data.columns for col in expected_columns):
            print("✅ All OHLCV columns present")
        else:
            print("❌ Missing OHLCV columns")
            return False
        
        # Check for proper OHLC relationships
        ohlc_valid = (
            (convenience_data['high'] >= convenience_data['low']).all() and
            (convenience_data['high'] >= convenience_data['open']).all() and
            (convenience_data['high'] >= convenience_data['close']).all() and
            (convenience_data['low'] <= convenience_data['open']).all() and
            (convenience_data['low'] <= convenience_data['close']).all()
        )
        
        if ohlc_valid:
            print("✅ OHLC relationships are valid")
        else:
            print("❌ Invalid OHLC relationships found")
            return False
        
        # Check for positive prices and volumes
        if (convenience_data[['open', 'high', 'low', 'close']] > 0).all().all():
            print("✅ All prices are positive")
        else:
            print("❌ Found non-positive prices")
            return False
            
        if (convenience_data['volume'] >= 0).all():
            print("✅ All volumes are non-negative")
        else:
            print("❌ Found negative volumes")
            return False
        
        print("\n" + "=" * 50)
        print("🎉 ALL DATA LOADER TESTS PASSED!")
        print("✅ Your data is properly formatted and ready for training")
        print(f"✅ Dataset contains {len(available_dates)} days of data")
        print(f"✅ Date range: {available_dates[0]} to {available_dates[-1]}")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loader()
    if not success:
        sys.exit(1)
    
    print("\n🚀 Ready to proceed with training!") 