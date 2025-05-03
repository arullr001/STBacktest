import os
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
import json
import glob
import time
import traceback
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import numba
import psutil
import warnings
from concurrent.futures import ProcessPoolExecutor
import cupy as cp
import numba.cuda
from numba import cuda
import concurrent.futures
import logging
from pathlib import Path
import math

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define constants
THREADS_PER_BLOCK = 256
MAX_RAM_USAGE_PERCENT = 90
BATCH_SIZE = 1000
CURRENT_UTC = "2025-05-03 20:01:35"
CURRENT_USER = "arullr001"
TARGET_MAX = 15.00  # Maximum target (1500 points)

# Directory Structure Constants
LOG_DIR = "logs"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
CACHE_DIR = "cache"


# Updated timestamp
CURRENT_UTC = "2025-05-03 20:02:01"
CURRENT_USER = "arullr001"

def find_ohlc_files(data_dir):
    """Find all OHLC data files in the given directory"""
    files = []
    supported_extensions = ['.csv', '.xlsx', '.xls']
    for ext in supported_extensions:
        files.extend(glob.glob(os.path.join(data_dir, f'*{ext}')))
    return files

def select_files_to_process(data_files):
    """Let user select which files to process"""
    if not data_files:
        print("No data files found")
        return []

    print(f"Found {len(data_files)} data files for testing")
    print("\nAvailable OHLC files:")
    for i, file_path in enumerate(data_files):
        print(f"[{i+1}] {os.path.basename(file_path)}")

    print("\nEnter file number(s) to process (comma-separated, or 'all' for all files):")
    selection = input("> ").strip().lower()

    selected_files = []
    if selection == 'all':
        selected_files = data_files
    else:
        try:
            indices = [int(idx.strip()) - 1 for idx in selection.split(',') if idx.strip()]
            valid_indices = [idx for idx in indices if 0 <= idx < len(data_files)]
            if not valid_indices:
                print("No valid file selections. Exiting.")
                return []
            selected_files = [data_files[idx] for idx in valid_indices]
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")
            return []

    print(f"\nProcessing {len(selected_files)} file(s)")
    return selected_files

def load_ohlc_data(file_path):
    """Enhanced load and prepare OHLC data from file"""
    print(f"Attempting to load data from {file_path}...")

    try:
        if file_path.endswith('.csv'):
            sample = pd.read_csv(file_path, nrows=5)
            print(f"File columns: {list(sample.columns)}")

            has_date_col = any('date' in col.lower() for col in sample.columns)
            has_time_col = any('time' in col.lower() and 'datetime' not in col.lower() for col in sample.columns)

            df = pd.read_csv(file_path)

            column_mapping = {}
            date_col = None
            time_col = None

            for col in df.columns:
                col_lower = col.lower()
                if col in ['O', 'o'] or 'open' in col_lower:
                    column_mapping[col] = 'open'
                elif col in ['H', 'h'] or 'high' in col_lower:
                    column_mapping[col] = 'high'
                elif col in ['L', 'l'] or 'low' in col_lower:
                    column_mapping[col] = 'low'
                elif col in ['C', 'c'] or 'close' in col_lower:
                    column_mapping[col] = 'close'
                elif 'date' in col_lower and 'datetime' not in col_lower:
                    date_col = col
                elif 'time' in col_lower and 'datetime' not in col_lower:
                    time_col = col
                elif any(x in col_lower for x in ['datetime']):
                    column_mapping[col] = 'datetime'

            print(f"Column mapping: {column_mapping}")
            print(f"Date column: {date_col}, Time column: {time_col}")

            df = df.rename(columns=column_mapping)

            if date_col and time_col:
                print(f"Combining '{date_col}' and '{time_col}' into datetime")
                try:
                    df['datetime'] = pd.to_datetime(df[date_col] + ' ' + df[time_col])
                except Exception as e:
                    print(f"Error combining date and time: {e}")
                    try:
                        df['datetime'] = pd.to_datetime(df[date_col]) + pd.to_timedelta(df[time_col])
                    except Exception as e2:
                        print(f"Second attempt failed: {e2}")
                        print("Sample date values:", df[date_col].head())
                        print("Sample time values:", df[time_col].head())
            elif date_col:
                print(f"Using '{date_col}' as datetime")
                df['datetime'] = pd.to_datetime(df[date_col])

            required_cols = ['open', 'high', 'low', 'close', 'datetime']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"Missing required columns: {missing_cols}")
                print("Available columns:", list(df.columns))
                raise ValueError(f"Could not find all required columns. Missing: {missing_cols}")

        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        numeric_cols = ['open', 'high', 'low', 'close']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        df.dropna(subset=numeric_cols, inplace=True)

        print(f"Successfully loaded data with shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")

        return df

    except Exception as e:
        print(f"Error during data loading: {str(e)}")
        raise

def get_float_input(prompt, min_value=None, max_value=None, allow_zero=False):
    """Get validated float input from user"""
    while True:
        try:
            value = input(prompt).strip()
            if value == "" and allow_zero:
                return 0.0
            value = float(value)

            if min_value is not None and value < min_value:
                print(f"Value must be at least {min_value}")
                continue

            if max_value is not None and value > max_value:
                print(f"Value must be at most {max_value}")
                continue

            return value
        except ValueError:
            print("Please enter a valid number")

def get_int_input(prompt, min_value=None, max_value=None):
    """Get validated integer input from user"""
    while True:
        try:
            value = int(input(prompt).strip())

            if min_value is not None and value < min_value:
                print(f"Value must be at least {min_value}")
                continue

            if max_value is not None and value > max_value:
                print(f"Value must be at most {max_value}")
                continue

            return value
        except ValueError:
            print("Please enter a valid integer")
            
            
# Updated timestamp
CURRENT_UTC = "2025-05-03 20:03:04"
CURRENT_USER = "arullr001"


# Updated timestamp
CURRENT_UTC = "2025-05-03 20:13:00"
CURRENT_USER = "arullr001"

def get_parameter_inputs():
    """Automatically determine parameter ranges based on price range and target"""
    print("\n" + "=" * 50)
    print(" PARAMETER CONFIGURATION ".center(50, "="))
    print("=" * 50)

    # Fixed Price Range values (50-300 points)
    print("\nUsing standard price ranges: 50 to 300 points (0.50 to 3.00)")
    print("Price range increment: 10 points (0.10)")
    price_ranges = [round(x, 2) for x in np.arange(0.50, 3.01, 0.10)]

    # Automatically calculate period ranges based on price ranges
    # Smaller price ranges need shorter periods, larger ranges need longer periods
    periods = []
    for pr in price_ranges:
        min_period = max(5, int(pr * 10))  # Minimum period of 5
        max_period = min(50, int(pr * 20))  # Maximum period of 50
        step = max(1, int((max_period - min_period) / 5))  # Dynamic step size
        periods.extend(range(min_period, max_period + 1, step))
    periods = sorted(list(set(periods)))  # Remove duplicates and sort

    # Automatically calculate multiplier ranges based on price ranges
    # Smaller price ranges need lower multipliers, larger ranges need higher multipliers
    multipliers = []
    for pr in price_ranges:
        min_mult = max(0.5, pr)  # Minimum multiplier of 0.5
        max_mult = min(5.0, pr * 2)  # Maximum multiplier of 5.0
        step = (max_mult - min_mult) / 10  # Dynamic step size
        multipliers.extend(np.arange(min_mult, max_mult + step, step))
    multipliers = sorted(list(set([round(x, 2) for x in multipliers])))  # Remove duplicates and sort

    # Generate target ranges for each price range
    target_combinations = []
    for pr in price_ranges:
        min_target = pr * 3  # Minimum target is 3x price range
        target_step = pr * 0.1  # Increment is 10% of price range
        targets = [round(x, 2) for x in np.arange(min_target, TARGET_MAX + target_step, target_step)]
        # Create combinations for both long and short targets
        for long_target in targets:
            for short_target in targets:
                target_combinations.append((pr, long_target, short_target))

    # Calculate total combinations
    total_combinations = len(periods) * len(multipliers) * len(target_combinations)

    # Summary
    print("\n" + "=" * 50)
    print(" PARAMETER SUMMARY ".center(50, "="))
    print("=" * 50)
    print(f"Period range: {min(periods)} to {max(periods)} - {len(periods)} values")
    print(f"Multiplier range: {min(multipliers):.2f} to {max(multipliers):.2f} - {len(multipliers)} values")
    print(f"Price ranges: 50 to 300 points (step 10 points) - {len(price_ranges)} values")
    print(f"Target configurations: {len(target_combinations)} combinations")
    print(f"\nTotal parameter combinations to test: {total_combinations}")

    # Memory estimation
    mem = psutil.virtual_memory()
    estimated_memory_mb = total_combinations * 0.5
    print(f"Estimated memory required: ~{estimated_memory_mb:.1f} MB")
    print(f"Available memory: {mem.available / (1024**2):.1f} MB")

    if total_combinations > 100000:
        print("\nNOTE: Large number of combinations. Processing may take some time.")

    print("\nProceeding with automated parameter optimization...")

    return {
        'periods': periods,
        'multipliers': multipliers,
        'price_ranges': price_ranges,
        'target_combinations': target_combinations,
        'total_combinations': total_combinations
    }


# Update the timestamp constants
CURRENT_UTC = "2025-05-03 20:24:36"
CURRENT_USER = "arullr001"

class DirectoryManager:
    """Manages directory structure for the application"""
    def __init__(self):
        # Get the name of the currently executing Python file
        current_file = os.path.basename(__file__)  # Gets 'ST-Deven.py'
        file_name = os.path.splitext(current_file)[0]  # Removes '.py' extension
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Simplified directory name: just filename_timestamp
        self.base_dir = f"{file_name}_{self.timestamp}"
        self.csv_dumps_dir = os.path.join(self.base_dir, "csv_dumps")
        self.error_logs_dir = os.path.join(self.base_dir, "error_logs")
        self.final_results_dir = os.path.join(self.base_dir, "final_results")
        self.plots_dir = os.path.join(self.base_dir, "plots")
        self.cache_dir = os.path.join(self.base_dir, "cache")
        self.create_directory_structure()
        self.setup_logging()

    def create_directory_structure(self):
        """Creates the required directory structure"""
        directories = [
            self.base_dir,
            self.csv_dumps_dir,
            self.error_logs_dir,
            self.final_results_dir,
            self.plots_dir,
            self.cache_dir
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def setup_logging(self):
        """Sets up logging configuration"""
        # Processing errors logger
        processing_logger = logging.getLogger('processing_errors')
        processing_logger.setLevel(logging.ERROR)
        fh_processing = logging.FileHandler(
            os.path.join(self.error_logs_dir, 'processing_errors.log')
        )
        fh_processing.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        processing_logger.addHandler(fh_processing)

        # System errors logger
        system_logger = logging.getLogger('system_errors')
        system_logger.setLevel(logging.ERROR)
        fh_system = logging.FileHandler(
            os.path.join(self.error_logs_dir, 'system_errors.log')
        )
        fh_system.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        system_logger.addHandler(fh_system)


class MemoryManager:
    """Manages memory usage and monitoring"""
    def __init__(self, max_ram_percent=MAX_RAM_USAGE_PERCENT):
        self.max_ram_percent = max_ram_percent
        self.total_ram = psutil.virtual_memory().total
        self.max_ram_bytes = (self.total_ram * self.max_ram_percent) / 100

    def get_current_ram_usage(self):
        """Returns current RAM usage in bytes"""
        return psutil.Process().memory_info().rss

    def get_available_ram(self):
        """Returns available RAM in bytes"""
        return self.max_ram_bytes - self.get_current_ram_usage()

    def is_ram_available(self, required_bytes):
        """Checks if there's enough RAM available"""
        return self.get_available_ram() >= required_bytes

    def estimate_combination_memory(self, df_size, n_combinations):
        """Estimates memory required for processing combinations"""
        return df_size * 2 * n_combinations

    def calculate_optimal_batch_size(self, df_size, total_combinations):
        """Calculates optimal batch size based on available memory"""
        available_ram = self.get_available_ram()
        estimated_mem_per_combo = df_size * 2
        optimal_batch_size = int(available_ram / (estimated_mem_per_combo * 1.2))
        return min(optimal_batch_size, BATCH_SIZE, total_combinations)
        
        
        
# Updated timestamp
CURRENT_UTC = "2025-05-03 20:04:33"
CURRENT_USER = "arullr001"

@cuda.jit
def calculate_supertrend_cuda_kernel(high, low, close, period, multiplier, price_range_1, up, dn, trend, trailing_up_30, trailing_dn_30):
    """CUDA kernel for SuperTrend calculation"""
    i = cuda.grid(1)
    if i >= len(close):
        return

    hl2 = (high[i] + low[i]) / 2

    if i == 0:
        tr = high[0] - low[0]
    else:
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )

    atr_sum = 0.0
    count = 0
    for j in range(max(0, i - period + 1), i + 1):
        if j == 0:
            atr_sum += high[0] - low[0]
        else:
            atr_sum += max(
                high[j] - low[j],
                abs(high[j] - close[j-1]),
                abs(low[j] - close[j-1])
            )
        count += 1
    atr = atr_sum / count

    up_basic = hl2 - (multiplier * atr)
    dn_basic = hl2 + (multiplier * atr)

    if i == 0:
        up[i] = up_basic
        dn[i] = dn_basic
        trend[i] = 0
    else:
        if close[i-1] > up[i-1]:
            up[i] = max(up_basic, up[i-1])
        else:
            up[i] = up_basic

        if close[i-1] < dn[i-1]:
            dn[i] = min(dn_basic, dn[i-1])
        else:
            dn[i] = dn_basic

        if close[i] > dn[i-1]:
            trend[i] = 1
        elif close[i] < up[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]

    trailing_up_30[i] = up[i] + up[i] * (price_range_1 / 100)
    trailing_dn_30[i] = dn[i] - dn[i] * (price_range_1 / 100)

def calculate_supertrend_cuda(df, period, multiplier, price_range_1):
    """Calculate SuperTrend using CUDA"""
    n = len(df)
    
    # Prepare arrays
    high = cuda.to_device(df['high'].values.astype(np.float64))
    low = cuda.to_device(df['low'].values.astype(np.float64))
    close = cuda.to_device(df['close'].values.astype(np.float64))
    
    up = cuda.device_array(n, dtype=np.float64)
    dn = cuda.device_array(n, dtype=np.float64)
    trend = cuda.device_array(n, dtype=np.int32)
    trailing_up_30 = cuda.device_array(n, dtype=np.float64)
    trailing_dn_30 = cuda.device_array(n, dtype=np.float64)
    
    # Calculate grid and block dimensions
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (n + (threadsperblock - 1)) // threadsperblock
    
    # Launch kernel
    calculate_supertrend_cuda_kernel[blockspergrid, threadsperblock](
        high, low, close, period, multiplier, price_range_1,
        up, dn, trend, trailing_up_30, trailing_dn_30
    )
    
    # Copy results back to host
    trend_host = trend.copy_to_host()
    trailing_up_30_host = trailing_up_30.copy_to_host()
    trailing_dn_30_host = trailing_dn_30.copy_to_host()
    
    return trend_host, trailing_up_30_host, trailing_dn_30_host

def calculate_supertrend(df, period, multiplier, price_range_1):
    """Wrapper function that chooses between CPU or GPU implementation"""
    try:
        if cuda.is_available():
            try:
                device = cuda.get_current_device()
                try:
                    free_mem = device.mem_info()[0]
                except AttributeError:
                    try:
                        free_mem = device.memory_info().free
                    except AttributeError:
                        try:
                            ctx = cuda.current_context()
                            free_mem = ctx.get_memory_info().free
                        except AttributeError:
                            print("Unable to determine GPU memory, using CPU", end='\r')
                            return calculate_supertrend_cuda(df, period, multiplier, price_range_1)

                data_size = len(df) * 8 * 5  # Approximate size in bytes
                
                if free_mem > data_size * 3:
                    print("Using GPU acceleration", end='\r')
                    return calculate_supertrend_cuda(df, period, multiplier, price_range_1)
                else:
                    print("Insufficient GPU memory, using CPU", end='\r')
                    return calculate_supertrend_cuda(df, period, multiplier, price_range_1)
            
            except Exception as e:
                print(f"GPU initialization failed, using CPU: {str(e)}", end='\r')
                return calculate_supertrend_cuda(df, period, multiplier, price_range_1)
        
        return calculate_supertrend_cuda(df, period, multiplier, price_range_1)
    
    except Exception as e:
        print(f"Error in SuperTrend calculation: {str(e)}", end='\r')
        return calculate_supertrend_cuda(df, period, multiplier, price_range_1)

def cleanup_gpu_memory():
    """Clean up GPU memory to prevent leaks"""
    try:
        if cuda.is_available():
            cuda.current_context().deallocations.clear()
            print("GPU memory cleaned", end='\r')
    except Exception as e:
        print(f"GPU memory cleanup error: {e}", end='\r')
        
        
# Updated timestamp
CURRENT_UTC = "2025-05-03 20:05:37"
CURRENT_USER = "arullr001"

def process_parameter_combination(df, params):
    """Process a single parameter combination"""
    period, multiplier, price_range_1, long_target, short_target = params
    
    try:
        trend, trailing_up_30, trailing_dn_30 = calculate_supertrend(
            df, period, multiplier, price_range_1
        )
        
        trades = []
        position = 0
        entry_price = 0
        entry_time = None
        
        for i in range(1, len(df)):
            current_time = df.index[i]
            current_close = df['close'].iloc[i]
            
            if position == 0:  # No position
                if trend[i] == 1 and trend[i-1] == -1:  # Buy signal
                    position = 1
                    entry_price = current_close
                    entry_time = current_time
                elif trend[i] == -1 and trend[i-1] == 1:  # Sell signal
                    position = -1
                    entry_price = current_close
                    entry_time = current_time
            
            elif position == 1:  # Long position
                target_price = entry_price * (1 + long_target/100)
                stop_loss = trailing_dn_30[i]
                
                if current_close >= target_price or current_close <= stop_loss:
                    exit_price = current_close
                    profit = ((exit_price - entry_price) / entry_price) * 100
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': 'LONG',
                        'profit_pct': profit,
                        'target': long_target,
                        'price_range': price_range_1,
                        'exit_reason': 'target' if current_close >= target_price else 'stop_loss'
                    })
                    
                    position = 0
            
            elif position == -1:  # Short position
                target_price = entry_price * (1 - short_target/100)
                stop_loss = trailing_up_30[i]
                
                if current_close <= target_price or current_close >= stop_loss:
                    exit_price = current_close
                    profit = ((entry_price - exit_price) / entry_price) * 100
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': 'SHORT',
                        'profit_pct': profit,
                        'target': short_target,
                        'price_range': price_range_1,
                        'exit_reason': 'target' if current_close <= target_price else 'stop_loss'
                    })
                    
                    position = 0
        
        if trades:
            trades_df = pd.DataFrame(trades)
            metrics = calculate_metrics(trades_df)
            
            # Enhanced metrics
            metrics.update({
                'total_long_trades': len(trades_df[trades_df['position'] == 'LONG']),
                'total_short_trades': len(trades_df[trades_df['position'] == 'SHORT']),
                'avg_trade_duration': (pd.to_datetime(trades_df['exit_time']) - 
                                     pd.to_datetime(trades_df['entry_time'])).mean(),
                'target_hits': len(trades_df[trades_df['exit_reason'] == 'target']),
                'stop_hits': len(trades_df[trades_df['exit_reason'] == 'stop_loss'])
            })
            
            return {
                'period': period,
                'multiplier': multiplier,
                'price_range_1': price_range_1,
                'long_target': long_target,
                'short_target': short_target,
                'trades': trades_df,
                **metrics
            }
        return None
    
    except Exception as e:
        logging.getLogger('processing_errors').error(
            f"Error processing combination: {params}\n{str(e)}\n{traceback.format_exc()}"
        )
        return None

def calculate_metrics(trades_df):
    """Calculate enhanced trading metrics from trades"""
    if trades_df.empty:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'max_drawdown': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'consecutive_wins': 0,
            'consecutive_losses': 0
        }
    
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['profit_pct'] > 0])
    win_rate = (winning_trades / total_trades) * 100
    
    avg_profit = trades_df['profit_pct'].mean()
    
    # Enhanced metrics calculation
    winning_trades_df = trades_df[trades_df['profit_pct'] > 0]
    losing_trades_df = trades_df[trades_df['profit_pct'] < 0]
    
    avg_win = winning_trades_df['profit_pct'].mean() if not winning_trades_df.empty else 0
    avg_loss = losing_trades_df['profit_pct'].mean() if not losing_trades_df.empty else 0
    largest_win = winning_trades_df['profit_pct'].max() if not winning_trades_df.empty else 0
    largest_loss = losing_trades_df['profit_pct'].min() if not losing_trades_df.empty else 0
    
    # Calculate consecutive wins/losses
    trades_df['win'] = trades_df['profit_pct'] > 0
    consecutive = trades_df['win'].groupby((trades_df['win'] != trades_df['win'].shift()).cumsum()).cumcount() + 1
    consecutive_wins = consecutive[trades_df['win']].max() if not trades_df[trades_df['win']].empty else 0
    consecutive_losses = consecutive[~trades_df['win']].max() if not trades_df[~trades_df['win']].empty else 0
    
    # Calculate drawdown
    cumulative_returns = (1 + trades_df['profit_pct']/100).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max * 100
    max_drawdown = abs(drawdowns.min())
    
    # Profit factor
    gross_profit = trades_df[trades_df['profit_pct'] > 0]['profit_pct'].sum()
    gross_loss = abs(trades_df[trades_df['profit_pct'] < 0]['profit_pct'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    # Sharpe ratio (simplified, assuming risk-free rate = 0)
    returns = trades_df['profit_pct']
    sharpe_ratio = (returns.mean() / returns.std()) * math.sqrt(252) if returns.std() != 0 else 0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'consecutive_wins': consecutive_wins,
        'consecutive_losses': consecutive_losses
    }
    
    
    
# Updated timestamp
CURRENT_UTC = "2025-05-03 20:06:37"
CURRENT_USER = "arullr001"

def process_batch(df, parameter_combinations, batch_start, batch_size):
    """Process a batch of parameter combinations"""
    batch_end = min(batch_start + batch_size, len(parameter_combinations))
    batch_combinations = parameter_combinations[batch_start:batch_end]
    
    results = []
    for params in batch_combinations:
        result = process_parameter_combination(df, params)
        if result:
            results.append(result)
    
    return results

class ResultsManager:
    """Manages the generation and storage of results and analysis"""
    def __init__(self, directory_manager):
        self.dir_manager = directory_manager
        self.processing_logger = logging.getLogger('processing_errors')
        self.system_logger = logging.getLogger('system_errors')

    def create_performance_summary(self, final_results_df):
        """Creates and saves a performance summary of the results"""
        try:
            if final_results_df.empty:
                return

            summary = {
                'total_combinations_tested': len(final_results_df),
                'best_performance': {
                    'total_profit': float(final_results_df['avg_profit'].max()),
                    'parameters': final_results_df.iloc[final_results_df['avg_profit'].idxmax()].to_dict(),
                },
                'average_performance': {
                    'profit': float(final_results_df['avg_profit'].mean()),
                    'trade_count': float(final_results_df['total_trades'].mean()),
                    'win_rate': float(final_results_df['win_rate'].mean()),
                },
                'performance_distribution': {
                    'min_profit': float(final_results_df['avg_profit'].min()),
                    'max_profit': float(final_results_df['avg_profit'].max()),
                    'median_profit': float(final_results_df['avg_profit'].median()),
                    'profit_std': float(final_results_df['avg_profit'].std()),
                    'win_rate_range': {
                        'min': float(final_results_df['win_rate'].min()),
                        'max': float(final_results_df['win_rate'].max()),
                        'avg': float(final_results_df['win_rate'].mean())
                    }
                },
                'risk_metrics': {
                    'avg_drawdown': float(final_results_df['max_drawdown'].mean()),
                    'worst_drawdown': float(final_results_df['max_drawdown'].max()),
                    'best_sharpe': float(final_results_df['sharpe_ratio'].max()),
                    'avg_profit_factor': float(final_results_df['profit_factor'].mean())
                },
                'generated_at': CURRENT_UTC,
                'generated_by': CURRENT_USER
            }

            # Save summary to JSON
            summary_file = os.path.join(self.dir_manager.final_results_dir, 'performance_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4)

            # Create visualizations
            self.create_performance_visualizations(final_results_df)

        except Exception as e:
            self.processing_logger.error(f"Error creating performance summary: {str(e)}")

    def create_performance_visualizations(self, df):
        """Creates visualization plots for the results"""
        try:
            plt.style.use('seaborn')
            figures_dir = os.path.join(self.dir_manager.plots_dir, 'analysis')
            os.makedirs(figures_dir, exist_ok=True)

            # 1. Profit Distribution
            plt.figure(figsize=(12, 6))
            plt.hist(df['avg_profit'], bins=50, edgecolor='black')
            plt.title('Distribution of Average Profit per Trade')
            plt.xlabel('Average Profit (%)')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(figures_dir, 'profit_distribution.png'))
            plt.close()

            # 2. Win Rate vs Profit Factor
            plt.figure(figsize=(10, 6))
            plt.scatter(df['win_rate'], df['profit_factor'], alpha=0.5)
            plt.title('Win Rate vs Profit Factor')
            plt.xlabel('Win Rate (%)')
            plt.ylabel('Profit Factor')
            plt.savefig(os.path.join(figures_dir, 'winrate_vs_profitfactor.png'))
            plt.close()

            # 3. Parameter Impact Analysis
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            axes[0,0].scatter(df['period'], df['avg_profit'], alpha=0.5)
            axes[0,0].set_title('Period vs Average Profit')
            axes[0,0].set_xlabel('Period')
            axes[0,0].set_ylabel('Average Profit (%)')

            axes[0,1].scatter(df['multiplier'], df['avg_profit'], alpha=0.5)
            axes[0,1].set_title('Multiplier vs Average Profit')
            axes[0,1].set_xlabel('Multiplier')
            axes[0,1].set_ylabel('Average Profit (%)')

            axes[1,0].scatter(df['price_range_1'], df['avg_profit'], alpha=0.5)
            axes[1,0].set_title('Price Range vs Average Profit')
            axes[1,0].set_xlabel('Price Range')
            axes[1,0].set_ylabel('Average Profit (%)')

            axes[1,1].scatter(df['max_drawdown'], df['sharpe_ratio'], alpha=0.5)
            axes[1,1].set_title('Max Drawdown vs Sharpe Ratio')
            axes[1,1].set_xlabel('Max Drawdown (%)')
            axes[1,1].set_ylabel('Sharpe Ratio')

            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'parameter_analysis.png'))
            plt.close()

        except Exception as e:
            self.processing_logger.error(f"Error creating visualizations: {str(e)}")

    def save_detailed_results(self, final_results_df, original_data):
        """Saves detailed analysis of the results"""
        try:
            # Save full results CSV
            results_file = os.path.join(self.dir_manager.final_results_dir, 'complete_results.csv')
            final_results_df.to_csv(results_file, index=False)

            # Save top 10 combinations with detailed metrics
            top_10 = final_results_df.nlargest(10, 'avg_profit')
            top_10_file = os.path.join(self.dir_manager.final_results_dir, 'top_10_combinations.csv')
            top_10.to_csv(top_10_file, index=False)

            # Create detailed analysis report
            report_file = os.path.join(self.dir_manager.final_results_dir, 'analysis_report.txt')
            with open(report_file, 'w') as f:
                f.write("SuperTrend Strategy Optimization Report\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Generated at: {CURRENT_UTC}\n")
                f.write(f"Generated by: {CURRENT_USER}\n\n")
                
                f.write("Top 10 Parameter Combinations:\n")
                f.write("-" * 30 + "\n")
                for idx, row in top_10.iterrows():
                    f.write(f"\nRank {idx + 1}:\n")
                    f.write(f"Period: {row['period']}\n")
                    f.write(f"Multiplier: {row['multiplier']:.2f}\n")
                    f.write(f"Price Range: {row['price_range_1']:.2f}\n")
                    f.write(f"Average Profit: {row['avg_profit']:.2f}%\n")
                    f.write(f"Win Rate: {row['win_rate']:.2f}%\n")
                    f.write(f"Profit Factor: {row['profit_factor']:.2f}\n")
                    f.write(f"Sharpe Ratio: {row['sharpe_ratio']:.2f}\n")
                    f.write(f"Max Drawdown: {row['max_drawdown']:.2f}%\n")
                    f.write("-" * 20 + "\n")

        except Exception as e:
            self.processing_logger.error(f"Error saving detailed results: {str(e)}")
            
            
            
# Updated timestamp
CURRENT_UTC = "2025-05-03 20:07:37"
CURRENT_USER = "arullr001"

class BatchProcessor:
    """Handles batched processing of parameter combinations"""
    def __init__(self, directory_manager, memory_manager):
        self.dir_manager = directory_manager
        self.mem_manager = memory_manager
        self.current_batch = 0
        self.processing_logger = logging.getLogger('processing_errors')
        self.system_logger = logging.getLogger('system_errors')

    def process_batch(self, df, param_combinations, batch_start, batch_size, max_workers=4):
        """Processes a batch of parameter combinations"""
        batch_end = min(batch_start + batch_size, len(param_combinations))
        batch_combinations = param_combinations[batch_start:batch_end]
        
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for params in batch_combinations:
                futures.append(
                    executor.submit(process_parameter_combination, df.copy(), params)
                )

            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), 
                             desc=f"Processing batch {self.current_batch + 1}"):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    self.processing_logger.error(
                        f"Error in batch {self.current_batch + 1}: {str(e)}"
                    )

        return results

    def save_batch_results(self, results, batch_num):
        """Saves batch results to CSV with metadata"""
        try:
            batch_file = os.path.join(
                self.dir_manager.csv_dumps_dir, 
                f'batch_{batch_num}_results.csv'
            )
            metadata_file = os.path.join(
                self.dir_manager.csv_dumps_dir, 
                f'batch_{batch_num}_metadata.json'
            )

            # Convert results to DataFrame
            results_data = []
            for r in results:
                if r and 'trades' in r:
                    row = {
                        'period': r['period'],
                        'multiplier': r['multiplier'],
                        'price_range_1': r['price_range_1'],
                        'long_target': r['long_target'],
                        'short_target': r['short_target'],
                        'total_trades': r['total_trades'],
                        'win_rate': r['win_rate'],
                        'avg_profit': r['avg_profit'],
                        'max_drawdown': r['max_drawdown'],
                        'profit_factor': r['profit_factor'],
                        'sharpe_ratio': r['sharpe_ratio'],
                        'avg_win': r['avg_win'],
                        'avg_loss': r['avg_loss'],
                        'consecutive_wins': r['consecutive_wins'],
                        'consecutive_losses': r['consecutive_losses']
                    }
                    results_data.append(row)

            if results_data:
                df = pd.DataFrame(results_data)
                df.to_csv(batch_file, index=False)

                metadata = {
                    'batch_number': batch_num,
                    'processed_at': CURRENT_UTC,
                    'processed_by': CURRENT_USER,
                    'combinations_processed': len(results_data),
                    'memory_usage_mb': self.mem_manager.get_current_ram_usage() / (1024 * 1024),
                    'batch_success': True,
                    'performance_summary': {
                        'avg_win_rate': float(df['win_rate'].mean()),
                        'best_profit': float(df['avg_profit'].max()),
                        'worst_drawdown': float(df['max_drawdown'].max()),
                        'avg_profit_factor': float(df['profit_factor'].mean())
                    }
                }

                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=4)

                return True
            return False

        except Exception as e:
            self.processing_logger.error(f"Error saving batch {batch_num}: {str(e)}")
            return False

    def merge_batch_results(self):
        """Merges all batch results into final results"""
        try:
            all_results = []
            batch_files = glob.glob(os.path.join(
                self.dir_manager.csv_dumps_dir, 
                'batch_*_results.csv'
            ))
            
            for file in batch_files:
                try:
                    df = pd.read_csv(file)
                    all_results.append(df)
                except Exception as e:
                    self.processing_logger.error(f"Error reading batch file {file}: {str(e)}")

            if all_results:
                final_df = pd.concat(all_results, ignore_index=True)
                final_df.sort_values('avg_profit', ascending=False, inplace=True)
                
                # Save final results
                final_results_file = os.path.join(
                    self.dir_manager.final_results_dir, 
                    'complete_results.csv'
                )
                final_df.to_csv(final_results_file, index=False)
                
                # Save metadata
                final_metadata = {
                    'processed_at': CURRENT_UTC,
                    'processed_by': CURRENT_USER,
                    'total_combinations': len(final_df),
                    'best_profit': float(final_df['avg_profit'].max()),
                    'avg_profit': float(final_df['avg_profit'].mean()),
                    'total_batches': len(batch_files),
                    'performance_summary': {
                        'best_win_rate': float(final_df['win_rate'].max()),
                        'best_profit_factor': float(final_df['profit_factor'].max()),
                        'best_sharpe': float(final_df['sharpe_ratio'].max()),
                        'lowest_drawdown': float(final_df['max_drawdown'].min())
                    }
                }
                
                with open(os.path.join(
                    self.dir_manager.final_results_dir, 
                    'final_metadata.json'
                ), 'w') as f:
                    json.dump(final_metadata, f, indent=4)
                
                return final_df
                
            return pd.DataFrame()

        except Exception as e:
            self.system_logger.error(f"Error merging batch results: {str(e)}")
            return pd.DataFrame()
            
            
# Updated timestamp
CURRENT_UTC = "2025-05-03 20:08:50"
CURRENT_USER = "arullr001"

def main():
    """Main execution function"""
    try:
        start_time = time.time()
        print("\nSuperTrend Strategy Backtester")
        print("=" * 50)
        print(f"Started at (UTC): {CURRENT_UTC}")
        print(f"User: {CURRENT_USER}")

        # Initialize managers
        dir_manager = DirectoryManager()
        mem_manager = MemoryManager()
        batch_processor = BatchProcessor(dir_manager, mem_manager)
        results_manager = ResultsManager(dir_manager)

        print(f"\nCreated working directory: {dir_manager.base_dir}")
        print("Directory structure:")
        print(f"├── csv_dumps/")
        print(f"├── error_logs/")
        print(f"├── final_results/")
        print(f"├── plots/")
        print(f"└── cache/")

        # Find and select data files
        data_dir = input("\nEnter the directory containing OHLC data files: ").strip()
        data_files = find_ohlc_files(data_dir)

        if not data_files:
            print("No data files found in the specified directory. Exiting.")
            return

        selected_files = select_files_to_process(data_files)
        if not selected_files:
            print("No files selected for processing. Exiting.")
            return

        # Get parameter inputs
        params = get_parameter_inputs()

        # Process each selected file
        for file_path in selected_files:
            print(f"\nProcessing file: {os.path.basename(file_path)}")
            try:
                # Load OHLC data
                df = load_ohlc_data(file_path)
                
                # Save input data metadata
                input_metadata = {
                    'filename': os.path.basename(file_path),
                    'rows': len(df),
                    'date_range': {
                        'start': str(df.index.min()),
                        'end': str(df.index.max())
                    },
                    'processed_at': CURRENT_UTC,
                    'processed_by': CURRENT_USER
                }
                
                with open(os.path.join(dir_manager.final_results_dir, 'input_metadata.json'), 'w') as f:
                    json.dump(input_metadata, f, indent=4)

                # Generate parameter combinations
                param_combinations = list(product(
                    params['periods'],
                    params['multipliers'],
                    params['price_ranges'],
                    [params['long_target']],
                    [params['short_target']]
                ))

                total_combinations = len(param_combinations)
                print(f"\nTotal parameter combinations to test: {total_combinations}")

                # Process in batches
                batch_size = mem_manager.calculate_optimal_batch_size(
                    df.memory_usage().sum(),
                    total_combinations
                )

                print("\nProcessing combinations in batches...")
                for batch_start in range(0, total_combinations, batch_size):
                    try:
                        batch_results = batch_processor.process_batch(
                            df,
                            param_combinations,
                            batch_start,
                            batch_size
                        )
                        
                        if batch_results:
                            batch_processor.save_batch_results(
                                batch_results,
                                batch_processor.current_batch
                            )
                            batch_processor.current_batch += 1
                            
                        cleanup_gpu_memory()
                        
                    except Exception as e:
                        logging.getLogger('processing_errors').error(
                            f"Error processing batch starting at {batch_start}: {str(e)}"
                        )

                # Merge all batch results
                print("\nMerging batch results...")
                final_results_df = batch_processor.merge_batch_results()

                # Generate performance summary and visualizations
                print("\nGenerating performance summary and visualizations...")
                results_manager.create_performance_summary(final_results_df)
                results_manager.save_detailed_results(final_results_df, df)

                end_time = time.time()
                duration = end_time - start_time
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = int(duration % 60)

                print(f"\nProcessing completed in {hours:02d}:{minutes:02d}:{seconds:02d}")
                print(f"Results saved in: {dir_manager.base_dir}")
                
                if not final_results_df.empty:
                    print("\nTop 5 Parameter Combinations:")
                    print("=" * 80)
                    top_5 = final_results_df.head(5)
                    for idx, row in top_5.iterrows():
                        print(f"\nRank {idx + 1}:")
                        print(f"Period: {row['period']}")
                        print(f"Multiplier: {row['multiplier']:.2f}")
                        print(f"Price Range: {row['price_range_1']:.2f}")
                        print(f"Average Profit: {row['avg_profit']:.2f}%")
                        print(f"Win Rate: {row['win_rate']:.2f}%")
                        print(f"Profit Factor: {row['profit_factor']:.2f}")
                        print(f"Sharpe Ratio: {row['sharpe_ratio']:.2f}")
                        print("-" * 50)

            except Exception as e:
                logging.getLogger('processing_errors').error(
                    f"Error processing file {file_path}: {str(e)}\n{traceback.format_exc()}"
                )
                print(f"Error processing file {file_path}. Check error logs for details.")

        print(f"\nAll processing completed at: {CURRENT_UTC}")

    except Exception as e:
        logging.getLogger('system_errors').error(
            f"System error: {str(e)}\n{traceback.format_exc()}"
        )
        print("A system error occurred. Check error logs for details.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.getLogger('system_errors').error(
            f"Fatal error: {str(e)}\n{traceback.format_exc()}"
        )
    finally:
        cleanup_gpu_memory()