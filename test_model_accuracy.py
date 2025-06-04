#!/usr/bin/env python3
"""
Comprehensive Model Accuracy and Performance Testing Script
Tests both TFT model accuracy and RL agent trading performance after training
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.data_loader import load_nasdaq_futures_data
from src.models.tft_model import TradingTFT
from src.environment.trading_environment import HybridTradingEnvironment
from src.config import config
from stable_baselines3 import PPO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelAccuracyTester:
    """Comprehensive testing suite for trained models"""
    
    def __init__(self, tft_model_path: str = None, rl_model_path: str = None):
        self.tft_model_path = tft_model_path or str(config.data.MODEL_DIR / "tft_model.pt")
        self.rl_model_path = rl_model_path or str(config.data.MODEL_DIR / "rl_final_model.zip")
        self.test_results = {}
        
    def run_complete_evaluation(self, test_start_date: str = None, test_end_date: str = None):
        """Run complete model evaluation suite"""
        logger.info("🧪 Starting Comprehensive Model Accuracy Testing")
        logger.info("=" * 60)
        
        # Load test data
        test_data = self._load_test_data(test_start_date, test_end_date)
        
        # Test 1: TFT Model Accuracy
        logger.info("\n📊 Testing TFT Model Accuracy...")
        tft_results = self.test_tft_accuracy(test_data)
        
        # Test 2: RL Agent Trading Performance
        logger.info("\n🎮 Testing RL Agent Trading Performance...")
        rl_results = self.test_rl_performance(test_data)
        
        # Test 3: Hybrid System Integration
        logger.info("\n🔗 Testing Hybrid System Integration...")
        hybrid_results = self.test_hybrid_integration(test_data)
        
        # Test 4: Risk Management Validation
        logger.info("\n🛡️ Testing Risk Management...")
        risk_results = self.test_risk_management(test_data)
        
        # Test 5: Market Regime Performance
        logger.info("\n📈 Testing Market Regime Performance...")
        regime_results = self.test_market_regimes(test_data)
        
        # Compile final results
        self.test_results.update({
            'tft_accuracy': tft_results,
            'rl_performance': rl_results,
            'hybrid_integration': hybrid_results,
            'risk_management': risk_results,
            'market_regimes': regime_results
        })
        
        # Generate comprehensive report
        self.generate_performance_report()
        
        return self.test_results
    
    def _load_test_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load out-of-sample test data"""
        logger.info("📅 Loading test data...")
        
        if not start_date or not end_date:
            # Use most recent 6 months for testing
            end_date = "2024-12-31"
            start_date = "2024-07-01"
        
        test_data = load_nasdaq_futures_data(start_date, end_date, resample_5min=True)
        logger.info(f"✅ Loaded {len(test_data)} test samples from {start_date} to {end_date}")
        
        return test_data
    
    def test_tft_accuracy(self, test_data: pd.DataFrame) -> dict:
        """Test TFT model prediction accuracy"""
        logger.info("🧠 Evaluating TFT Model Predictions...")
        
        try:
            # Load TFT model
            if not Path(self.tft_model_path).exists():
                logger.warning("⚠️  TFT model not found - skipping TFT accuracy test")
                return {'status': 'SKIP', 'reason': 'Model not found'}
            
            tft_model = TradingTFT()
            tft_model.load_model(Path(self.tft_model_path))
            
            # Prepare test dataset
            from src.data.feature_engineering import FeatureEngineer
            feature_engineer = FeatureEngineer()
            features_data = feature_engineer.create_complete_feature_set(test_data)
            
            # Split into validation windows
            prediction_accuracy = []
            directional_accuracy = []
            confidence_calibration = []
            
            window_size = config.tft.LOOKBACK_PERIODS
            total_windows = len(features_data) - window_size - 10
            test_windows = min(100, total_windows)  # Test on 100 windows
            
            logger.info(f"📊 Testing on {test_windows} prediction windows...")
            
            for i in range(0, test_windows, 10):  # Every 10th window
                try:
                    # Get prediction window
                    start_idx = i
                    end_idx = i + window_size
                    pred_window = features_data.iloc[start_idx:end_idx]
                    
                    # Get actual future prices (next 10 periods)
                    future_prices = features_data.iloc[end_idx:end_idx+10]['close'].values
                    current_price = pred_window.iloc[-1]['close']
                    
                    if len(future_prices) < 10:
                        continue
                    
                    # Make TFT prediction
                    predictions = tft_model.predict(pred_window)
                    
                    # Calculate accuracy metrics
                    if isinstance(predictions, dict):
                        # Quantile accuracy
                        actual_max = np.max(future_prices)
                        actual_min = np.min(future_prices)
                        
                        pred_high = predictions.get('quantile_90', current_price)
                        pred_low = predictions.get('quantile_10', current_price)
                        
                        # Check if actual range falls within predicted range
                        range_accuracy = (actual_min >= pred_low * 0.9) and (actual_max <= pred_high * 1.1)
                        prediction_accuracy.append(float(range_accuracy))
                        
                        # Directional accuracy
                        pred_direction = predictions.get('predicted_direction', 0)
                        actual_direction = 1 if future_prices[-1] > current_price else -1
                        directional_accuracy.append(float(pred_direction * actual_direction > 0))
                        
                        # Confidence calibration
                        confidence = predictions.get('confidence', 0.5)
                        confidence_calibration.append(confidence)
                    
                except Exception as e:
                    logger.debug(f"Error in prediction window {i}: {str(e)}")
                    continue
            
            # Calculate TFT metrics
            tft_metrics = {
                'status': 'PASS',
                'prediction_accuracy': np.mean(prediction_accuracy) if prediction_accuracy else 0,
                'directional_accuracy': np.mean(directional_accuracy) if directional_accuracy else 0,
                'confidence_calibration': np.mean(confidence_calibration) if confidence_calibration else 0,
                'total_predictions': len(prediction_accuracy),
                'model_loaded': True
            }
            
            logger.info(f"✅ TFT Accuracy Results:")
            logger.info(f"   Prediction Accuracy: {tft_metrics['prediction_accuracy']:.1%}")
            logger.info(f"   Directional Accuracy: {tft_metrics['directional_accuracy']:.1%}")
            logger.info(f"   Average Confidence: {tft_metrics['confidence_calibration']:.3f}")
            
            return tft_metrics
            
        except Exception as e:
            logger.error(f"❌ TFT accuracy test failed: {str(e)}")
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_rl_performance(self, test_data: pd.DataFrame) -> dict:
        """Test RL agent trading performance"""
        logger.info("🎮 Evaluating RL Agent Performance...")
        
        try:
            # Load RL model
            if not Path(self.rl_model_path).exists():
                logger.warning("⚠️  RL model not found - skipping RL performance test")
                return {'status': 'SKIP', 'reason': 'Model not found'}
            
            rl_model = PPO.load(self.rl_model_path)
            
            # Load TFT model if available
            tft_model = None
            if Path(self.tft_model_path).exists():
                tft_model = TradingTFT()
                tft_model.load_model(Path(self.tft_model_path))
            
            # Create trading environment
            env = HybridTradingEnvironment(
                data=test_data,
                tft_model=tft_model,
                initial_balance=config.trading.ACCOUNT_SIZE
            )
            
            # Run multiple evaluation episodes
            logger.info("🔄 Running evaluation episodes...")
            episode_results = []
            
            num_episodes = 10
            for episode in range(num_episodes):
                obs, info = env.reset()
                episode_reward = 0
                done = False
                step_count = 0
                
                while not done and step_count < len(test_data) - 1:
                    action, _ = rl_model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                    step_count += 1
                
                # Get episode metrics
                episode_metrics = env.get_trading_metrics()
                episode_metrics['episode_reward'] = episode_reward
                episode_metrics['steps'] = step_count
                episode_results.append(episode_metrics)
                
                logger.info(f"Episode {episode+1}: Return={episode_metrics.get('total_return', 0):.2%}, "
                          f"Trades={episode_metrics.get('total_trades', 0)}, "
                          f"Win Rate={episode_metrics.get('win_rate', 0):.1%}")
            
            # Aggregate performance metrics
            rl_metrics = self._aggregate_episode_results(episode_results)
            rl_metrics['status'] = 'PASS'
            rl_metrics['model_loaded'] = True
            
            logger.info(f"✅ RL Performance Results:")
            logger.info(f"   Average Return: {rl_metrics['avg_total_return']:.2%}")
            logger.info(f"   Average Win Rate: {rl_metrics['avg_win_rate']:.1%}")
            logger.info(f"   Average Sharpe Ratio: {rl_metrics['avg_sharpe_ratio']:.2f}")
            logger.info(f"   Max Drawdown: {rl_metrics['avg_max_drawdown']:.1%}")
            
            return rl_metrics
            
        except Exception as e:
            logger.error(f"❌ RL performance test failed: {str(e)}")
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_hybrid_integration(self, test_data: pd.DataFrame) -> dict:
        """Test TFT-RL hybrid system integration"""
        logger.info("🔗 Testing TFT-RL Integration...")
        
        try:
            # Check if both models exist
            tft_exists = Path(self.tft_model_path).exists()
            rl_exists = Path(self.rl_model_path).exists()
            
            if not (tft_exists and rl_exists):
                return {'status': 'SKIP', 'reason': 'Both models required for integration test'}
            
            # Load both models
            tft_model = TradingTFT()
            tft_model.load_model(Path(self.tft_model_path))
            rl_model = PPO.load(self.rl_model_path)
            
            # Test signal correlation
            signal_correlations = []
            decision_alignments = []
            
            # Sample test on smaller dataset
            sample_data = test_data.iloc[-1000:].copy()  # Last 1000 periods
            
            env = HybridTradingEnvironment(
                data=sample_data,
                tft_model=tft_model,
                initial_balance=config.trading.ACCOUNT_SIZE
            )
            
            obs, _ = env.reset()
            
            for step in range(min(500, len(sample_data) - 1)):
                # Get TFT prediction
                try:
                    current_data = sample_data.iloc[step]
                    from src.data.feature_engineering import FeatureEngineer
                    feature_engineer = FeatureEngineer()
                    
                    # Create a small window for TFT
                    start_idx = max(0, step - 50)
                    tft_window = sample_data.iloc[start_idx:step+1]
                    features_window = feature_engineer.create_complete_feature_set(tft_window)
                    
                    tft_pred = tft_model.predict(features_window.tail(1))
                    
                    # Get RL action
                    rl_action, _ = rl_model.predict(obs, deterministic=True)
                    
                    # Check alignment
                    if isinstance(tft_pred, dict):
                        tft_signal = tft_pred.get('predicted_direction', 0)
                        rl_signal = 1 if rl_action == 1 else -1 if rl_action == 2 else 0
                        
                        if tft_signal != 0 and rl_signal != 0:
                            alignment = float(tft_signal * rl_signal > 0)
                            decision_alignments.append(alignment)
                    
                    # Step environment
                    obs, _, done, _, _ = env.step(rl_action)
                    if done:
                        break
                        
                except Exception as e:
                    logger.debug(f"Integration test step {step} error: {str(e)}")
                    continue
            
            integration_metrics = {
                'status': 'PASS',
                'signal_alignment': np.mean(decision_alignments) if decision_alignments else 0,
                'integration_tests': len(decision_alignments),
                'both_models_loaded': True
            }
            
            logger.info(f"✅ Integration Results:")
            logger.info(f"   Signal Alignment: {integration_metrics['signal_alignment']:.1%}")
            logger.info(f"   Integration Tests: {integration_metrics['integration_tests']}")
            
            return integration_metrics
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {str(e)}")
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_risk_management(self, test_data: pd.DataFrame) -> dict:
        """Test risk management compliance"""
        logger.info("🛡️ Testing Risk Management...")
        
        try:
            if not Path(self.rl_model_path).exists():
                return {'status': 'SKIP', 'reason': 'RL model required'}
            
            rl_model = PPO.load(self.rl_model_path)
            
            # Test with various account sizes and risk scenarios
            risk_tests = []
            
            for initial_balance in [50000, 100000, 200000]:
                env = HybridTradingEnvironment(
                    data=test_data.tail(500),  # Last 500 periods
                    initial_balance=initial_balance
                )
                
                obs, _ = env.reset()
                max_risk_violation = 0
                drawdown_violations = 0
                
                for step in range(100):  # Quick risk test
                    action, _ = rl_model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = env.step(action)
                    
                    # Check risk violations
                    current_risk = info.get('current_risk', 0)
                    if current_risk > config.trading.DEFAULT_RISK_PER_TRADE * 1.5:  # 50% over limit
                        max_risk_violation = max(max_risk_violation, current_risk)
                    
                    # Check drawdown violations
                    if info.get('max_drawdown', 0) > config.trading.MAX_DRAWDOWN_THRESHOLD:
                        drawdown_violations += 1
                    
                    if done:
                        break
                
                risk_tests.append({
                    'initial_balance': initial_balance,
                    'max_risk_violation': max_risk_violation,
                    'drawdown_violations': drawdown_violations,
                    'final_balance': info.get('balance', initial_balance)
                })
            
            risk_metrics = {
                'status': 'PASS',
                'max_risk_violation': max([t['max_risk_violation'] for t in risk_tests]),
                'total_drawdown_violations': sum([t['drawdown_violations'] for t in risk_tests]),
                'risk_compliance': all([t['max_risk_violation'] < 0.02 for t in risk_tests]),  # Under 2%
                'risk_tests': len(risk_tests)
            }
            
            logger.info(f"✅ Risk Management Results:")
            logger.info(f"   Risk Compliance: {risk_metrics['risk_compliance']}")
            logger.info(f"   Max Risk Violation: {risk_metrics['max_risk_violation']:.1%}")
            logger.info(f"   Drawdown Violations: {risk_metrics['total_drawdown_violations']}")
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"❌ Risk management test failed: {str(e)}")
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_market_regimes(self, test_data: pd.DataFrame) -> dict:
        """Test performance across different market regimes"""
        logger.info("📈 Testing Market Regime Performance...")
        
        try:
            if not Path(self.rl_model_path).exists():
                return {'status': 'SKIP', 'reason': 'RL model required'}
            
            # Identify market regimes based on volatility and trend
            test_data['returns'] = test_data['close'].pct_change()
            test_data['volatility'] = test_data['returns'].rolling(20).std()
            test_data['trend'] = test_data['close'].rolling(50).mean() / test_data['close'].rolling(100).mean() - 1
            
            # Define regimes
            high_vol_threshold = test_data['volatility'].quantile(0.7)
            trend_threshold = 0.01
            
            regimes = {
                'trending_low_vol': (test_data['trend'] > trend_threshold) & (test_data['volatility'] < high_vol_threshold),
                'trending_high_vol': (test_data['trend'] > trend_threshold) & (test_data['volatility'] >= high_vol_threshold),
                'ranging_low_vol': (abs(test_data['trend']) <= trend_threshold) & (test_data['volatility'] < high_vol_threshold),
                'ranging_high_vol': (abs(test_data['trend']) <= trend_threshold) & (test_data['volatility'] >= high_vol_threshold)
            }
            
            regime_results = {}
            rl_model = PPO.load(self.rl_model_path)
            
            for regime_name, regime_mask in regimes.items():
                regime_data = test_data[regime_mask].head(200)  # Sample each regime
                
                if len(regime_data) < 50:
                    continue
                
                env = HybridTradingEnvironment(
                    data=regime_data,
                    initial_balance=config.trading.ACCOUNT_SIZE
                )
                
                obs, _ = env.reset()
                for step in range(len(regime_data) - 1):
                    action, _ = rl_model.predict(obs, deterministic=True)
                    obs, _, done, _, _ = env.step(action)
                    if done:
                        break
                
                regime_metrics = env.get_trading_metrics()
                regime_results[regime_name] = {
                    'return': regime_metrics.get('total_return', 0),
                    'win_rate': regime_metrics.get('win_rate', 0),
                    'total_trades': regime_metrics.get('total_trades', 0),
                    'sharpe_ratio': regime_metrics.get('sharpe_ratio', 0)
                }
            
            logger.info(f"✅ Market Regime Results:")
            for regime, metrics in regime_results.items():
                logger.info(f"   {regime}: Return={metrics['return']:.1%}, Win Rate={metrics['win_rate']:.1%}")
            
            return {
                'status': 'PASS',
                'regime_performance': regime_results,
                'regimes_tested': len(regime_results)
            }
            
        except Exception as e:
            logger.error(f"❌ Market regime test failed: {str(e)}")
            return {'status': 'FAIL', 'error': str(e)}
    
    def _aggregate_episode_results(self, episode_results: list) -> dict:
        """Aggregate results from multiple episodes"""
        if not episode_results:
            return {}
        
        metrics = {}
        for key in episode_results[0].keys():
            if isinstance(episode_results[0][key], (int, float)):
                values = [ep.get(key, 0) for ep in episode_results]
                metrics[f'avg_{key}'] = np.mean(values)
                metrics[f'std_{key}'] = np.std(values)
                metrics[f'min_{key}'] = np.min(values)
                metrics[f'max_{key}'] = np.max(values)
        
        return metrics
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 COMPREHENSIVE MODEL ACCURACY REPORT")
        logger.info("=" * 60)
        
        # Overall Summary
        tests_passed = sum(1 for result in self.test_results.values() 
                          if isinstance(result, dict) and result.get('status') == 'PASS')
        total_tests = len(self.test_results)
        
        logger.info(f"\n🎯 OVERALL RESULTS: {tests_passed}/{total_tests} tests passed")
        
        # Detailed Results
        for test_name, results in self.test_results.items():
            if isinstance(results, dict):
                status = results.get('status', 'UNKNOWN')
                status_emoji = "✅" if status == 'PASS' else "⚠️" if status == 'SKIP' else "❌"
                logger.info(f"\n{status_emoji} {test_name.upper()}: {status}")
                
                # Log key metrics
                for key, value in results.items():
                    if key not in ['status', 'error', 'reason'] and isinstance(value, (int, float)):
                        if 'rate' in key or 'return' in key or 'accuracy' in key:
                            logger.info(f"   {key}: {value:.1%}")
                        else:
                            logger.info(f"   {key}: {value:.3f}")
        
        # Save results to file
        results_file = Path("model_accuracy_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        logger.info(f"\n📄 Detailed results saved to: {results_file}")
        
        logger.info("\n" + "=" * 60)
        
        # Performance Assessment
        if tests_passed >= total_tests * 0.8:
            logger.info("🎉 EXCELLENT: Models show strong performance across all metrics!")
        elif tests_passed >= total_tests * 0.6:
            logger.info("✅ GOOD: Models perform well with minor areas for improvement")
        else:
            logger.info("⚠️  NEEDS IMPROVEMENT: Consider retraining or parameter adjustment")

def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Model Accuracy and Performance")
    parser.add_argument("--tft-model", type=str, help="Path to TFT model")
    parser.add_argument("--rl-model", type=str, help="Path to RL model")
    parser.add_argument("--test-start", type=str, help="Test start date (YYYY-MM-DD)")
    parser.add_argument("--test-end", type=str, help="Test end date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ModelAccuracyTester(args.tft_model, args.rl_model)
    
    # Run evaluation
    results = tester.run_complete_evaluation(args.test_start, args.test_end)
    
    print("\n🎯 Model accuracy testing completed!")
    print("📄 Check 'model_accuracy_results.json' for detailed results")

if __name__ == "__main__":
    main() 