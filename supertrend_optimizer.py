#!/usr/bin/env python3
"""
Enhanced Supertrend Strategy with Advanced Optimization
Version: 2.0
Current UTC: 2025-05-07 06:22:58
User: arullr001
"""

# Standard library imports
import os
import sys
import json
import logging
import traceback
import warnings
from datetime import datetime
from pathlib import Path
import multiprocessing as mp
from functools import partial
import threading
import gc
import copy

# Data processing and numerical computations
import numpy as np
import pandas as pd
from scipy import stats

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

# Optimization libraries
import optuna
from skopt import gp_minimize, space
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

# GPU acceleration
try:
    import cupy as cp
    import numba
    from numba import cuda
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# System monitoring
import psutil
import platform

# Suppress warnings
warnings.filterwarnings('ignore')

# Global Constants
CURRENT_UTC = "2025-05-07 06:22:58"
CURRENT_USER = "arullr001"
THREADS_PER_BLOCK = 256
MAX_RAM_USAGE_PERCENT = 90
BATCH_SIZE = 1000

class SupertrendConfig:
    """Configuration class for Supertrend Strategy"""
    def __init__(self):
        # Strategy Parameters with updated ranges
        self.param_ranges = {
            'atr_period': {
                'min': 10,
                'max': 100,
                'step': 5
            },
            'factor': {
                'min': 2.0,
                'max': 20.0,
                'step': 0.1
            },
            'buffer_distance': {
                'min': 30,
                'max': 200,
                'step': 1
            },
            'hard_stop_distance': {
                'min': 10,
                'max': 50,
                'step': 1
            },
            'long_target_rr': {
                'min': 0,
                'max': 5.0,
                'step': 0.1
            },
            'short_target_rr': {
                'min': 0,
                'max': 5.0,
                'step': 0.1
            }
        }
        
        # Default parameters
        self.parameters = {
            'atr_period': 10,
            'factor': 3.0,
            'buffer_distance': 50,
            'hard_stop_distance': 20,
            'long_target_rr': 2.0,
            'short_target_rr': 2.0
        }
        
        # Trading settings
        self.position_size = 0.02  # 2% risk per trade
        self.max_drawdown = 0.20   # 20% maximum drawdown
        self.initial_capital = 100000
        
        # Optimization settings
        self.optimization = {
            'enabled': True,
            'method': 'ensemble',
            'trials': 100,
            'parallel_jobs': mp.cpu_count() - 1
        }
        
        # Blackout period settings
        self.blackout = {
            'enabled': False,
            'start_time': '0400',
            'end_time': '1000',
            'timezone': 'UTC'
        }

    def update_parameters(self, new_params):
        """Update strategy parameters with validation"""
        for param, value in new_params.items():
            if param in self.param_ranges:
                # Validate parameter is within allowed range
                min_val = self.param_ranges[param]['min']
                max_val = self.param_ranges[param]['max']
                step = self.param_ranges[param]['step']
                
                # Round to nearest step
                value = round(value / step) * step
                
                # Clamp to range
                value = max(min_val, min(max_val, value))
                
                self.parameters[param] = value

    def to_dict(self):
        """Convert configuration to dictionary"""
        return {
            'parameters': self.parameters,
            'position_size': self.position_size,
            'max_drawdown': self.max_drawdown,
            'initial_capital': self.initial_capital,
            'optimization': self.optimization,
            'blackout': self.blackout,
            'timestamp': CURRENT_UTC,
            'user': CURRENT_USER
        }

    def save_config(self, filepath):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_config(cls, filepath):
        """Load configuration from JSON file"""
        config = cls()
        with open(filepath, 'r') as f:
            data = json.load(f)
            config.parameters.update(data.get('parameters', {}))
            config.position_size = data.get('position_size', 0.02)
            config.max_drawdown = data.get('max_drawdown', 0.20)
            config.initial_capital = data.get('initial_capital', 100000)
            config.optimization.update(data.get('optimization', {}))
            config.blackout.update(data.get('blackout', {}))
        return config
		
		
class DirectoryManager:
    """Directory management and logging setup"""
    def __init__(self, base_dir="Supertrend_Strategy"):
        self.timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.user = "arullr001"
        self.base_dir = f"{base_dir}_{self.timestamp}"
        
        # Directory structure
        self.dirs = {
            'data': os.path.join(self.base_dir, "data"),
            'results': os.path.join(self.base_dir, "results"),
            'logs': os.path.join(self.base_dir, "logs"),
            'charts': os.path.join(self.base_dir, "charts"),
            'optimization': os.path.join(self.base_dir, "optimization")
        }
        
        # Create base directory first
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.dirs['logs'], exist_ok=True)
        
        # Setup logging first
        self.setup_logging()
        
        # Then create other directories
        self.setup_directories()

    def setup_logging(self):
        """Configure logging system"""
        self.logger = logging.getLogger(__name__)
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        try:
            # Create a file handler
            file_handler = logging.FileHandler(
                os.path.join(self.dirs['logs'], 'strategy.log')
            )
            file_handler.setFormatter(logging.Formatter(log_format))
            
            # Create a console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_format))
            
            # Configure logger
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
            self.logger.info("Logging system initialized")
            
        except Exception as e:
            print(f"Failed to setup logging: {str(e)}")
            raise

    def setup_directories(self):
        """Create directory structure"""
        try:
            for dir_name, dir_path in self.dirs.items():
                if dir_name != 'logs':  # logs directory already created
                    os.makedirs(dir_path, exist_ok=True)
                    self.logger.info(f"Created directory: {dir_path}")
                    
        except Exception as e:
            self.logger.error(f"Failed to create directories: {str(e)}")
            raise


class DataHandler:
    """Data loading and preprocessing"""
    def __init__(self, directory_manager):
        self.dir_manager = directory_manager
        self.logger = logging.getLogger(__name__)

    def load_data(self, filepath):
        """Load and validate data"""
        try:
            # Validate file path
            if not os.path.exists(filepath):
                self.logger.error(f"File not found: {filepath}")
                raise FileNotFoundError(f"File not found: {filepath}")

            # Check file extension
            _, ext = os.path.splitext(filepath)
            if ext.lower() not in ['.csv', '.xlsx', '.xls']:
                self.logger.error(f"Unsupported file format: {ext}")
                raise ValueError(f"Unsupported file format. Please use .csv, .xlsx, or .xls files")

            # Load data based on file type
            if ext.lower() == '.csv':
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            
            # Check for required columns with possible variations
            required_columns = {
                'open': ['O', 'o', 'open', 'Open', 'OPEN'],
                'high': ['H', 'h', 'high', 'High', 'HIGH'],
                'low': ['L', 'l', 'low', 'Low', 'LOW'],
                'close': ['C', 'c', 'close', 'Close', 'CLOSE'],
                'date': ['date', 'Date', 'DATE'],
                'time': ['time', 'Time', 'TIME']
            }
            
            # Map columns to standard names
            column_mapping = {}
            for std_col, variations in required_columns.items():
                found = False
                for var in variations:
                    if var in df.columns:
                        column_mapping[var] = std_col
                        found = True
                        break
                if not found:
                    self.logger.error(f"Missing required column: {std_col} (accepted names: {variations})")
                    raise ValueError(f"Missing required column: {std_col}")
            
            # Rename columns to standard names
            df = df.rename(columns=column_mapping)
            
            # Combine date and time columns into timestamp
            try:
                if 'time' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
                else:
                    df['timestamp'] = pd.to_datetime(df['date'])
                
                # Drop original date and time columns
                df = df.drop(['date', 'time'] if 'time' in df.columns else ['date'], axis=1)
                
                # Set timestamp as index
                df.set_index('timestamp', inplace=True)
            except Exception as e:
                self.logger.error(f"Error processing datetime: {str(e)}")
                raise ValueError(f"Error processing datetime: {str(e)}")
            
            # Sort by index
            df.sort_index(inplace=True)
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            # Forward fill missing values
            df.fillna(method='ffill', inplace=True)
            
            # Rename columns to standard lowercase names
            df.columns = [col.lower() for col in df.columns]
            
            # Verify final structure
            required_final_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_final_cols if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns after processing: {missing_cols}")
                raise ValueError(f"Missing required columns after processing: {missing_cols}")
            
            self.logger.info(f"Successfully loaded data with {len(df)} rows")
            self.logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            return None

    def prepare_data(self, df):
        """Prepare data for analysis"""
        try:
            # Calculate basic metrics
            df['hl2'] = (df['high'] + df['low']) / 2
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Calculate daily volatility
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {str(e)}")
            return None

class SupertrendCalculator:
    """Core Supertrend calculations"""
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)



class SupertrendCalculator:
    """Core Supertrend calculations"""
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def calculate_supertrend(self, df):  # Fix the indentation of this method
        """Calculate Supertrend indicators"""
        try:
            # Validate input data
            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_columns):
                self.logger.error(f"Missing required columns. Required: {required_columns}")
                raise ValueError(f"Missing required columns. Required: {required_columns}")

            # Create working copy of dataframe
            df = df.copy()

            # Calculate middle point if not exists
            if 'hl2' not in df.columns:
                df['hl2'] = (df['high'] + df['low']) / 2

            # Calculate True Range and ATR
            try:
                tr = pd.DataFrame()
                tr['hl'] = df['high'] - df['low']
                tr['hc'] = abs(df['high'] - df['close'].shift(1))
                tr['lc'] = abs(df['low'] - df['close'].shift(1))
                tr['tr'] = tr[['hl', 'hc', 'lc']].max(axis=1)
                
                df['atr'] = tr['tr'].rolling(
                    window=self.config.parameters['atr_period']
                ).mean()
                
                # Handle NaN values in ATR
                df['atr'] = df['atr'].fillna(method='bfill')
                
            except Exception as e:
                self.logger.error(f"Error calculating ATR: {str(e)}")
                raise

            # Calculate basic bands
            try:
                df['basic_ub'] = df['hl2'] + (
                    self.config.parameters['factor'] * df['atr']
                )
                df['basic_lb'] = df['hl2'] - (
                    self.config.parameters['factor'] * df['atr']
                )
            except Exception as e:
                self.logger.error(f"Error calculating basic bands: {str(e)}")
                raise

            # Initialize final bands
            df['final_ub'] = 0.0
            df['final_lb'] = 0.0
            
            # Calculate final bands using vectorized operations where possible
            try:
                for i in range(len(df)):
                    if i == 0:
                        df['final_ub'].iat[i] = df['basic_ub'].iat[i]
                        df['final_lb'].iat[i] = df['basic_lb'].iat[i]
                    else:
                        df['final_ub'].iat[i] = (
                            df['basic_ub'].iat[i] 
                            if (df['basic_ub'].iat[i] < df['final_ub'].iat[i-1] or 
                                df['close'].iat[i-1] > df['final_ub'].iat[i-1])
                            else df['final_ub'].iat[i-1]
                        )
                        
                        df['final_lb'].iat[i] = (
                            df['basic_lb'].iat[i] 
                            if (df['basic_lb'].iat[i] > df['final_lb'].iat[i-1] or 
                                df['close'].iat[i-1] < df['final_lb'].iat[i-1])
                            else df['final_lb'].iat[i-1]
                        )
            except Exception as e:
                self.logger.error(f"Error calculating final bands: {str(e)}")
                raise

            # Calculate Supertrend
            try:
                df['supertrend'] = 0.0
                df['trend'] = 0  # 1 for uptrend, -1 for downtrend
                
                for i in range(len(df)):
                    if i == 0:
                        df['supertrend'].iat[i] = df['final_ub'].iat[i]
                        df['trend'].iat[i] = 1
                        continue
                    
                    if df['close'].iat[i-1] <= df['supertrend'].iat[i-1]:
                        df['supertrend'].iat[i] = df['final_ub'].iat[i]
                    else:
                        df['supertrend'].iat[i] = df['final_lb'].iat[i]
                    
                    # Determine trend
                    df['trend'].iat[i] = 1 if df['close'].iat[i] > df['supertrend'].iat[i] else -1
                
            except Exception as e:
                self.logger.error(f"Error calculating Supertrend: {str(e)}")
                raise

            # Calculate buffer zones
            try:
                df['upper_buffer'] = df['supertrend'] + self.config.parameters['buffer_distance']
                df['lower_buffer'] = df['supertrend'] - self.config.parameters['buffer_distance']
            except Exception as e:
                self.logger.error(f"Error calculating buffer zones: {str(e)}")
                raise

            # Validate output
            required_outputs = ['supertrend', 'trend', 'upper_buffer', 'lower_buffer']
            if not all(col in df.columns for col in required_outputs):
                self.logger.error(f"Missing required output columns: {required_outputs}")
                raise ValueError(f"Missing required output columns: {required_outputs}")

            self.logger.info("Supertrend calculation completed successfully")
            return df
                
        except Exception as e:
            self.logger.error(f"Supertrend calculation failed: {str(e)}")
            return None



class SupertrendStrategy:
    """Core Supertrend strategy implementation"""
    def __init__(self, config):
        self.timestamp = "2025-05-08 01:06:48"
        self.user = "arullr001"
        self.config = config
        self.calculator = SupertrendCalculator(config)
        self.logger = logging.getLogger(__name__)



def generate_signals(self, df):
    """Generate trading signals"""
    try:
        # Calculate Supertrend indicators
        df = self.calculator.calculate_supertrend(df)
        if df is None:
            return None
            
        # Initialize signal and tracking columns
        df['long_entry'] = False
        df['short_entry'] = False
        df['long_exit'] = False
        df['short_exit'] = False
        df['position'] = 0
        df['entry_price'] = 0.0  # Initialize entry_price
        df['stop_loss'] = 0.0    # Initialize stop_loss
        
        # Generate signals
        for i in range(1, len(df)):
            # Long entry conditions
            df.loc[df.index[i], 'long_entry'] = (
                df['trend'].iloc[i] == 1 and
                df['trend'].iloc[i-1] == -1 and
                df['close'].iloc[i] <= df['upper_buffer'].iloc[i]
            )
            
            # Short entry conditions
            df.loc[df.index[i], 'short_entry'] = (
                df['trend'].iloc[i] == -1 and
                df['trend'].iloc[i-1] == 1 and
                df['close'].iloc[i] >= df['lower_buffer'].iloc[i]
            )
            
            # Exit conditions
            if df['position'].iloc[i-1] == 1:  # Long position
                # Exit long if:
                # 1. Trend changes (if target_rr is 0)
                # 2. Target reached (if target_rr > 0)
                # 3. Stop loss hit
                if df['entry_price'].iloc[i-1] > 0:  # Check if we have a valid entry price
                    target_price = df['entry_price'].iloc[i-1] + (
                        (df['entry_price'].iloc[i-1] - df['stop_loss'].iloc[i-1]) * 
                        self.config.parameters['long_target_rr']
                    )
                    
                    df.loc[df.index[i], 'long_exit'] = (
                        (self.config.parameters['long_target_rr'] == 0 and 
                         df['trend'].iloc[i] == -1) or
                        (self.config.parameters['long_target_rr'] > 0 and
                         df['high'].iloc[i] >= target_price) or
                        df['low'].iloc[i] <= df['stop_loss'].iloc[i-1]
                    )
            
            elif df['position'].iloc[i-1] == -1:  # Short position
                # Exit short if:
                # 1. Trend changes (if target_rr is 0)
                # 2. Target reached (if target_rr > 0)
                # 3. Stop loss hit
                if df['entry_price'].iloc[i-1] > 0:  # Check if we have a valid entry price
                    target_price = df['entry_price'].iloc[i-1] - (
                        (df['stop_loss'].iloc[i-1] - df['entry_price'].iloc[i-1]) * 
                        self.config.parameters['short_target_rr']
                    )
                    
                    df.loc[df.index[i], 'short_exit'] = (
                        (self.config.parameters['short_target_rr'] == 0 and 
                         df['trend'].iloc[i] == 1) or
                        (self.config.parameters['short_target_rr'] > 0 and
                         df['low'].iloc[i] <= target_price) or
                        df['high'].iloc[i] >= df['stop_loss'].iloc[i-1]
                    )
            
            # Update position and prices
            if df['long_entry'].iloc[i]:
                df.loc[df.index[i], 'position'] = 1
                df.loc[df.index[i], 'entry_price'] = df['close'].iloc[i]
                df.loc[df.index[i], 'stop_loss'] = (
                    df['entry_price'].iloc[i] - 
                    self.config.parameters['hard_stop_distance']
                )
            
            elif df['short_entry'].iloc[i]:
                df.loc[df.index[i], 'position'] = -1
                df.loc[df.index[i], 'entry_price'] = df['close'].iloc[i]
                df.loc[df.index[i], 'stop_loss'] = (
                    df['entry_price'].iloc[i] + 
                    self.config.parameters['hard_stop_distance']
                )
            
            elif df['long_exit'].iloc[i] or df['short_exit'].iloc[i]:
                df.loc[df.index[i], 'position'] = 0
                df.loc[df.index[i], 'entry_price'] = 0
                df.loc[df.index[i], 'stop_loss'] = 0
            
            else:
                df.loc[df.index[i], 'position'] = df['position'].iloc[i-1]
                df.loc[df.index[i], 'entry_price'] = df['entry_price'].iloc[i-1]
                df.loc[df.index[i], 'stop_loss'] = df['stop_loss'].iloc[i-1]
        
        return df
        
    except Exception as e:
        self.logger.error(f"Signal generation failed: {str(e)}")
        return None



    def validate_signals(self, df):
        """Validate generated signals"""
        try:
            validation_results = {
                'total_signals': 0,
                'long_entries': 0,
                'short_entries': 0,
                'long_exits': 0,
                'short_exits': 0,
                'errors': []
            }
            
            # Count signals
            validation_results['long_entries'] = df['long_entry'].sum()
            validation_results['short_entries'] = df['short_entry'].sum()
            validation_results['long_exits'] = df['long_exit'].sum()
            validation_results['short_exits'] = df['short_exit'].sum()
            validation_results['total_signals'] = (
                validation_results['long_entries'] +
                validation_results['short_entries'] +
                validation_results['long_exits'] +
                validation_results['short_exits']
            )
            
            # Validate signal logic
            for i in range(1, len(df)):
                # Check for simultaneous entry signals
                if df['long_entry'].iloc[i] and df['short_entry'].iloc[i]:
                    validation_results['errors'].append(
                        f"Simultaneous entry signals at index {i}"
                    )
                
                # Check for entries while in position
                if df['position'].iloc[i-1] != 0:
                    if df['long_entry'].iloc[i] or df['short_entry'].iloc[i]:
                        validation_results['errors'].append(
                            f"Entry signal while in position at index {i}"
                        )
                
                # Check for exits without position
                if df['position'].iloc[i-1] == 0:
                    if df['long_exit'].iloc[i] or df['short_exit'].iloc[i]:
                        validation_results['errors'].append(
                            f"Exit signal without position at index {i}"
                        )
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Signal validation failed: {str(e)}")
            return None



class BacktestEngine:
    """Advanced backtesting engine with performance analysis"""
    def __init__(self, config, directory_manager):
        self.timestamp = "2025-05-08 01:22:18"
        self.user = "arullr001"
        self.config = config
        self.dir_manager = directory_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize strategy
        self.strategy = SupertrendStrategy(config)
        
        # Performance tracking
        self.trades = []
        self.equity_curve = []
        self.current_position = 0
        self.current_equity = self.config.initial_capital

    def run_backtest(self, df):
        """Execute backtest"""
        try:
            # Generate signals
            df = self.strategy.generate_signals(df)
            
            # Validate signals
            validation_results = self.strategy.validate_signals(df)
            if validation_results['errors']:
                for error in validation_results['errors']:
                    self.logger.warning(error)
            
            # Initialize tracking variables
            position_size = self.config.position_size
            max_equity = self.current_equity
            max_drawdown = 0
            
            # Track each bar
            for i in range(1, len(df)):
                current_bar = df.iloc[i]
                previous_bar = df.iloc[i-1]
                
                # Handle position exits
                if self.current_position != 0:
                    # Calculate current P&L
                    if self.current_position == 1:  # Long position
                        if current_bar['long_exit']:
                            # Calculate exit price based on condition
                            if current_bar['low'] <= self.current_stop:
                                exit_price = max(current_bar['open'], self.current_stop)
                            elif self.config.parameters['long_target_rr'] > 0 and \
                                 current_bar['high'] >= self.current_target:
                                exit_price = self.current_target
                            else:
                                exit_price = current_bar['close']
                            
                            # Record trade
                            self.record_trade(
                                'long',
                                self.entry_price,
                                exit_price,
                                previous_bar.name,
                                current_bar.name,
                                position_size
                            )
                            
                            self.current_position = 0
                    
                    else:  # Short position
                        if current_bar['short_exit']:
                            # Calculate exit price based on condition
                            if current_bar['high'] >= self.current_stop:
                                exit_price = min(current_bar['open'], self.current_stop)
                            elif self.config.parameters['short_target_rr'] > 0 and \
                                 current_bar['low'] <= self.current_target:
                                exit_price = self.current_target
                            else:
                                exit_price = current_bar['close']
                            
                            # Record trade
                            self.record_trade(
                                'short',
                                self.entry_price,
                                exit_price,
                                previous_bar.name,
                                current_bar.name,
                                position_size
                            )
                            
                            self.current_position = 0
                
                # Handle new entries
                if self.current_position == 0:
                    if current_bar['long_entry']:
                        self.current_position = 1
                        self.entry_price = current_bar['close']
                        self.current_stop = self.entry_price - \
                            self.config.parameters['hard_stop_distance']
                        if self.config.parameters['long_target_rr'] > 0:
                            self.current_target = self.entry_price + \
                                (self.entry_price - self.current_stop) * \
                                self.config.parameters['long_target_rr']
                        else:
                            self.current_target = float('inf')
                        
                    elif current_bar['short_entry']:
                        self.current_position = -1
                        self.entry_price = current_bar['close']
                        self.current_stop = self.entry_price + \
                            self.config.parameters['hard_stop_distance']
                        if self.config.parameters['short_target_rr'] > 0:
                            self.current_target = self.entry_price - \
                                (self.current_stop - self.entry_price) * \
                                self.config.parameters['short_target_rr']
                        else:
                            self.current_target = 0
                
                # Update equity and drawdown
                self.current_equity = self.calculate_equity(current_bar['close'])
                max_equity = max(max_equity, self.current_equity)
                current_drawdown = (max_equity - self.current_equity) / max_equity
                max_drawdown = max(max_drawdown, current_drawdown)
                
                # Record equity curve
                self.equity_curve.append({
                    'timestamp': current_bar.name,
                    'equity': self.current_equity,
                    'drawdown': current_drawdown
                })
            
            # Calculate final performance metrics
            performance_metrics = self.calculate_performance_metrics(max_drawdown)
            
            return {
                'trades': self.trades,
                'equity_curve': self.equity_curve,
                'metrics': performance_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Backtest execution failed: {str(e)}")
            return None

    def record_trade(self, direction, entry_price, exit_price, entry_time, 
                    exit_time, position_size):
        """Record trade details"""
        pnl = position_size * (
            (exit_price - entry_price) if direction == 'long'
            else (entry_price - exit_price)
        )
        
        self.trades.append({
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'position_size': position_size,
            'pnl': pnl,
            'pnl_percent': (pnl / self.current_equity) * 100
        })
        
        self.current_equity += pnl

    def calculate_equity(self, current_price):
        """Calculate current equity"""
        if self.current_position == 0:
            return self.current_equity
        
        unrealized_pnl = self.config.position_size * (
            (current_price - self.entry_price) 
            if self.current_position == 1
            else (self.entry_price - current_price)
        )
        
        return self.current_equity + unrealized_pnl

    def calculate_performance_metrics(self, max_drawdown):
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return None
        
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        metrics = {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) if self.trades else 0,
            'total_return': ((self.current_equity - self.config.initial_capital) / 
                           self.config.initial_capital) * 100,
            'max_drawdown': max_drawdown * 100,
            'profit_factor': (sum(t['pnl'] for t in winning_trades) / 
                            abs(sum(t['pnl'] for t in losing_trades))
                            if losing_trades else float('inf')),
            'average_win': np.mean([t['pnl'] for t in winning_trades]) 
                          if winning_trades else 0,
            'average_loss': np.mean([t['pnl'] for t in losing_trades])
                          if losing_trades else 0,
            'largest_win': max([t['pnl'] for t in self.trades]),
            'largest_loss': min([t['pnl'] for t in self.trades]),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'sortino_ratio': self.calculate_sortino_ratio()
        }
        
        return metrics

    def calculate_sharpe_ratio(self, risk_free_rate=0.02):
        """Calculate Sharpe Ratio"""
        if not self.trades:
            return 0
            
        returns = pd.Series([t['pnl_percent'] for t in self.trades])
        excess_returns = returns - risk_free_rate/252
        
        if len(excess_returns) < 2:
            return 0
            
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

    def calculate_sortino_ratio(self, risk_free_rate=0.02):
        """Calculate Sortino Ratio"""
        if not self.trades:
            return 0
            
        returns = pd.Series([t['pnl_percent'] for t in self.trades])
        excess_returns = returns - risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) < 2:
            return 0
            
        return np.sqrt(252) * (excess_returns.mean() / downside_returns.std())


class ParameterOptimizer:
    def __init__(self, config, directory_manager):
        self.timestamp = "2025-05-08 03:09:32"  # Update timestamp
        self.user = "arullr001"
        self.config = config
        self.dir_manager = directory_manager
        self.logger = logging.getLogger(__name__)
        
        # Define parameter spaces
        self.param_spaces = {
            'atr_period': {
                'min': 10,
                'max': 100,
                'step': 5
            },
            'factor': {
                'min': 2.0,
                'max': 20.0,
                'step': 0.1
            },
            'buffer_distance': {
                'min': 30,
                'max': 200,
                'step': 1
            },
            'hard_stop_distance': {
                'min': 10,
                'max': 50,
                'step': 1
            },
            'long_target_rr': {
                'min': 0,
                'max': 5.0,
                'step': 0.1
            },
            'short_target_rr': {
                'min': 0,
                'max': 5.0,
                'step': 0.1
            }
        }

    def optimize(self, data, method='ensemble', n_trials=100):
        """Run optimization with specified method"""
        try:
            self.logger.info(f"Starting parameter optimization using {method} method")
            
            optimization_results = {
                'timestamp': self.timestamp,
                'user': self.user,
                'method': method,
                'results': {}
            }
            
            if method in ['ensemble', 'bayesian']:
                bayes_results = self.run_bayesian_optimization(data, n_trials)
                optimization_results['results']['bayesian'] = bayes_results
                
            if method in ['ensemble', 'optuna']:
                optuna_results = self.optuna_optimization(data, n_trials)
                optimization_results['results']['optuna'] = optuna_results
                
            if method == 'ensemble':
                ensemble_results = self.ensemble_optimization(
                    optimization_results['results']
                )
                optimization_results['results']['ensemble'] = ensemble_results
            
            # Save optimization results
            self.save_optimization_results(optimization_results)
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            return None

    def run_bayesian_optimization(self, data, n_trials):  # Add this method here
        """Bayesian optimization implementation"""
        try:
            def objective(atr_period, factor, buffer_distance, 
                         hard_stop_distance, long_target_rr, short_target_rr):
                try:
                    # Round parameters to nearest step
                    params = {
                        'atr_period': round(atr_period / 5) * 5,
                        'factor': round(factor * 10) / 10,
                        'buffer_distance': round(buffer_distance),
                        'hard_stop_distance': round(hard_stop_distance),
                        'long_target_rr': round(long_target_rr * 10) / 10,
                        'short_target_rr': round(short_target_rr * 10) / 10
                    }
                    
                    # Update config and run backtest
                    test_config = copy.deepcopy(self.config)
                    test_config.update_parameters(params)
                    
                    backtest_engine = BacktestEngine(
                        test_config,
                        self.dir_manager
                    )
                    
                    results = backtest_engine.run_backtest(data)
                    if results is None or results['metrics'] is None:
                        return float('-inf')
                        
                    return self.calculate_objective_score(results['metrics'])
                    
                except Exception as e:
                    self.logger.error(f"Bayesian optimization iteration failed: {str(e)}")
                    return float('-inf')

            optimizer = BayesianOptimization(
                f=objective,
                pbounds={
                    'atr_period': (
                        self.param_spaces['atr_period']['min'],
                        self.param_spaces['atr_period']['max']
                    ),
                    'factor': (
                        self.param_spaces['factor']['min'],
                        self.param_spaces['factor']['max']
                    ),
                    'buffer_distance': (
                        self.param_spaces['buffer_distance']['min'],
                        self.param_spaces['buffer_distance']['max']
                    ),
                    'hard_stop_distance': (
                        self.param_spaces['hard_stop_distance']['min'],
                        self.param_spaces['hard_stop_distance']['max']
                    ),
                    'long_target_rr': (
                        self.param_spaces['long_target_rr']['min'],
                        self.param_spaces['long_target_rr']['max']
                    ),
                    'short_target_rr': (
                        self.param_spaces['short_target_rr']['min'],
                        self.param_spaces['short_target_rr']['max']
                    )
                },
                random_state=42
            )

            # Set up utility function (acquisition function)
            from bayes_opt import UtilityFunction
            utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)
            
            # Initialize with random points
            init_points = n_trials // 4
            n_iter = n_trials - init_points
            
            self.logger.info(f"Starting Bayesian optimization with {init_points} initial points and {n_iter} iterations")
            
            # Initialize results array to store valid scores
            valid_results = []
            
            # Use tqdm for progress tracking
            from tqdm import tqdm
            with tqdm(total=n_iter, desc="Bayesian Optimization Progress") as pbar:
                for i in range(n_iter):
                    try:
                        result = optimizer.maximize(
                            init_points=0 if i > 0 else init_points,
                            n_iter=1,
                            acquisition_function=utility
                        )
                        
                        # Only store finite values
                        if np.isfinite(result.max['target']):
                            valid_results.append(result)
                        
                        pbar.update(1)
                        
                        # Log progress every 100 iterations
                        if (i + 1) % 100 == 0:
                            self.logger.info(f"Iteration {i + 1}: Best score = {optimizer.max['target']:.4f}")
                        
                    except Exception as e:
                        self.logger.warning(f"Iteration {i} failed: {str(e)}")
                        continue
            
            if not valid_results:
                raise ValueError("No valid results found during optimization")
                
            # Find best result among valid results
            best_result = max(valid_results, key=lambda x: x.max['target'])
            
            self.logger.info("Bayesian optimization completed successfully")
            
            return {
                'best_params': best_result.max['params'],
                'best_score': best_result.max['target'],
                'all_results': optimizer.space.params
            }
            
        except Exception as e:
            self.logger.error(f"Bayesian optimization failed: {str(e)}")
            return None

    

def main():
    """Main execution flow"""
    try:
        print(f"\nSupertrend Strategy Optimizer v2.0")
        print(f"Timestamp: {CURRENT_UTC}")
        print(f"User: {CURRENT_USER}\n")
        
        # Initialize components
        dir_manager = DirectoryManager()
        config = SupertrendConfig()  # This was missing
        
        # Load data
        data_handler = DataHandler(dir_manager)
        
        print("\nData Loading:")
        print("-" * 50)
        print("Please provide the full path to your OHLC data file")
        print("Supported formats: .csv, .xlsx, .xls")
        print("Example: C:/Users/username/data/BTCUSDT_1h.csv")
        print("\nRequired columns: open/O, high/H, low/L, close/C")
        print("Optional: date, time, volume")
        
        while True:
            data_path = input("\nEnter the path to your OHLC data file: ").strip()
            
            # Remove quotes if present
            data_path = data_path.strip('"\'')
            
            # Check if user wants to exit
            if data_path.lower() in ['exit', 'quit', 'q']:
                print("\nExiting program...")
                return False
            
            # Try to load the data
            df = data_handler.load_data(data_path)
            if df is not None:
                break
            
            print("\nWould you like to:")
            print("1. Try again")
            print("2. Exit")
            choice = input("Enter your choice (1 or 2): ").strip()
            
            if choice != '1':
                print("\nExiting program...")
                return False
        
        print(f"\nSuccessfully loaded {len(df)} bars of data")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        # Display data sample
        print("\nFirst few rows of data:")
        print(df.head())
        
        # Ask for confirmation
        print("\nIs this the correct data?")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("\nExiting program...")
            return False

        # Optimization setup
        print("\nOptimization Setup:")
        print("-" * 50)
        n_trials = int(input("Enter number of optimization trials (default 100): ") or "100")
        
        print("\nOptimization Methods:")
        print("1. Ensemble (combines results from all methods)")
        print("2. Bayesian")
        print("3. Optuna")
        print("4. Run All Methods")
        
        method_choice = int(input("\nSelect optimization method (1-4): "))
        
        if method_choice == 4:
            optimization_methods = ['bayesian', 'optuna', 'ensemble']
            print("\nRunning all optimization methods sequentially...")
        else:
            optimization_methods = [['ensemble', 'bayesian', 'optuna'][method_choice - 1]]
        
        # Initialize optimizer
        optimizer = ParameterOptimizer(config, dir_manager)
        all_results = {}
        
        # Run selected optimization methods
        for method in optimization_methods:
            print(f"\nRunning {method.capitalize()} Optimization:")
            print("-" * 50)
            print(f"Starting {method} optimization with {n_trials} trials...")
            
            optimization_results = optimizer.optimize(
                df,
                method=method,
                n_trials=n_trials
            )
            
            if optimization_results is None:
                print(f"\n{method.capitalize()} optimization failed!")
                continue
                
            all_results[method] = optimization_results
            
            # Display results for this method
            best_params = (optimization_results['results']['ensemble']['best_params'] 
                         if method == 'ensemble' 
                         else optimization_results['results'][method]['best_params'])
            
            best_score = (optimization_results['results']['ensemble']['best_score']
                         if method == 'ensemble'
                         else optimization_results['results'][method]['best_score'])
            
            print(f"\n{method.capitalize()} Optimization Results:")
            print("-" * 50)
            print(f"Best Score: {best_score:.2f}")
            print("\nBest Parameters:")
            for param, value in best_params.items():
                print(f"{param}: {value}")
        
        # Rest of the function remains the same...
        # If all methods were run, show comparison
        if len(all_results) > 1:
            print("\nOptimization Methods Comparison:")
            print("-" * 50)
            comparison_data = []
            
            for method, results in all_results.items():
                if method == 'ensemble':
                    score = results['results']['ensemble']['best_score']
                else:
                    score = results['results'][method]['best_score']
                comparison_data.append((method, score))
            
            # Sort by score
            comparison_data.sort(key=lambda x: x[1], reverse=True)
            
            print("\nRanking by Performance:")
            for i, (method, score) in enumerate(comparison_data, 1):
                print(f"{i}. {method.capitalize()}: {score:.2f}")
        
        # Ask which method's parameters to use for final backtest
        if len(all_results) > 1:
            print("\nSelect parameters to use for final backtest:")
            for i, method in enumerate(all_results.keys(), 1):
                print(f"{i}. {method.capitalize()}")
            
            final_choice = int(input("\nEnter your choice (1-{0}): ".format(len(all_results))))
            final_method = list(all_results.keys())[final_choice - 1]
            
            best_params = (all_results[final_method]['results']['ensemble']['best_params']
                         if final_method == 'ensemble'
                         else all_results[final_method]['results'][final_method]['best_params'])
        else:
            best_params = (optimization_results['results']['ensemble']['best_params']
                         if method == 'ensemble'
                         else optimization_results['results'][method]['best_params'])
        
        # Run final backtest
        print("\nRunning Final Backtest:")
        print("-" * 50)
        
        config.update_parameters(best_params)
        backtest_engine = BacktestEngine(config, dir_manager)
        final_results = backtest_engine.run_backtest(df)
        
        if final_results is None:
            raise ValueError("Final backtest failed")
        
        # Display final results
        print("\nFinal Performance Metrics:")
        print("-" * 50)
        metrics = final_results['metrics']
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Total Trades: {metrics['total_trades']}")
        
        # Save results
        save_path = os.path.join(dir_manager.base_dir, 'final_results')
        print(f"\nResults saved in: {save_path}")
        
        return True
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Set the timestamp and user globally
    CURRENT_UTC = "2025-05-08 01:48:16"
    CURRENT_USER = "arullr001"
    
    try:
        success = main()
        
        if success:
            print("\nStrategy optimization completed successfully.")
        else:
            print("\nStrategy optimization failed. Check the error logs for details.")
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        traceback.print_exc()
