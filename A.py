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
CURRENT_UTC = "2025-05-27 01:00:33"  # Updated timestamp
CURRENT_USER = "arullr001"           # Updated user

# Utility Functions
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

def get_parameter_inputs():
    """Get all parameter inputs from user with validation for Pine Script strategy"""
    while True:
        print("\n" + "=" * 60)
        print(" PINE SCRIPT SUPERTREND STRATEGY CONFIGURATION ".center(60, "="))
        print("=" * 60)

        # ATR Period range inputs
        print("\nEnter ATR Period range (Integer values):")
        atr_start = get_int_input("  Start value (min 2, max 200): ", 2, 200)
        atr_end = get_int_input(f"  End value (min {atr_start}, max 200): ", atr_start, 200)
        atr_step = get_int_input("  Step/increment (min 1): ", 1)

        # Factor range inputs
        print("\nEnter Factor range (Float values):")
        factor_start = get_float_input("  Start value (min 1.0, max 100.0): ", 1.0, 100.0)
        factor_end = get_float_input(f"  End value (min {factor_start}, max 100.0): ", factor_start, 100.0)
        factor_step = get_float_input("  Step/increment (min 0.01): ", 0.01)

        # Buffer Distance range inputs
        print("\nEnter Buffer Distance range (Float values):")
        buffer_start = get_float_input("  Start value (min 1, max 500): ", 1, 500)
        buffer_end = get_float_input(f"  End value (min {buffer_start}, max 500): ", buffer_start, 500)
        buffer_step = get_float_input("  Step/increment (min 1): ", 1)

        # Hard Stop Distance range inputs
        print("\nEnter Hard Stop Distance range (Float values):")
        hardstop_start = get_float_input("  Start value (min 1, max 100): ", 1, 100)
        hardstop_end = get_float_input(f"  End value (min {hardstop_start}, max 100): ", hardstop_start, 100)
        hardstop_step = get_float_input("  Step/increment (min 1): ", 1)

        # Target-based exit configuration
        print("\nTarget-based Exit Configuration:")
        enable_target = input("  Enable target-based exits? (y/n): ").lower().strip() == 'y'
        
        if enable_target:
            print("  Target-based exits ENABLED - trades will exit at R:R targets")
            print("\nEnter Long R:R Ratio range (Integer values):")
            long_rr_start = get_int_input("    Start value (min 2, max 20): ", 2, 20)
            long_rr_end = get_int_input(f"    End value (min {long_rr_start}, max 20): ", long_rr_start, 20)
            long_rr_step = get_int_input("    Step/increment (min 1): ", 1)

            print("\nEnter Short R:R Ratio range (Integer values):")
            short_rr_start = get_int_input("    Start value (min 2, max 20): ", 2, 20)
            short_rr_end = get_int_input(f"    End value (min {short_rr_start}, max 20): ", short_rr_start, 20)
            short_rr_step = get_int_input("    Step/increment (min 1): ", 1)
        else:
            print("  Target-based exits DISABLED - trades will exit on trend changes")
            long_rr_start = long_rr_end = long_rr_step = 0
            short_rr_start = short_rr_end = short_rr_step = 0

        # Generate the ranges
        atr_periods = list(range(atr_start, atr_end + 1, atr_step))
        factors = [round(x, 2) for x in np.arange(factor_start, factor_end + (factor_step / 2), factor_step)]
        buffer_distances = [round(x, 1) for x in np.arange(buffer_start, buffer_end + (buffer_step / 2), buffer_step)]
        hardstop_distances = [round(x, 1) for x in np.arange(hardstop_start, hardstop_end + (hardstop_step / 2), hardstop_step)]
        
        if enable_target:
            long_rr_ratios = list(range(long_rr_start, long_rr_end + 1, long_rr_step))
            short_rr_ratios = list(range(short_rr_start, short_rr_end + 1, short_rr_step))
        else:
            long_rr_ratios = [0]
            short_rr_ratios = [0]

        # Calculate total combinations
        total_combinations = (len(atr_periods) * len(factors) * len(buffer_distances) * 
                            len(hardstop_distances) * len(long_rr_ratios) * len(short_rr_ratios))

        # Summary
        print("\n" + "=" * 60)
        print(" PARAMETER SUMMARY ".center(60, "="))
        print("=" * 60)
        print(f"ATR Period range: {atr_start} to {atr_end} (step {atr_step}) - {len(atr_periods)} values")
        print(f"Factor range: {factor_start} to {factor_end} (step {factor_step}) - {len(factors)} values")
        print(f"Buffer Distance range: {buffer_start} to {buffer_end} (step {buffer_step}) - {len(buffer_distances)} values")
        print(f"Hard Stop Distance range: {hardstop_start} to {hardstop_end} (step {hardstop_step}) - {len(hardstop_distances)} values")
        
        if enable_target:
            print(f"Long R:R ratio range: {long_rr_start} to {long_rr_end} (step {long_rr_step}) - {len(long_rr_ratios)} values")
            print(f"Short R:R ratio range: {short_rr_start} to {short_rr_end} (step {short_rr_step}) - {len(short_rr_ratios)} values")
        else:
            print("R:R ratios: Not applicable (trend-based exits)")
            
        print(f"\nTotal parameter combinations to test: {total_combinations:,}")

        # Memory estimation
        mem = psutil.virtual_memory()
        estimated_memory_mb = total_combinations * 0.8  # Increased for additional arrays
        print(f"Estimated memory required: ~{estimated_memory_mb:.1f} MB")
        print(f"Available memory: {mem.available / (1024**2):.1f} MB")

        if total_combinations > 100000:
            print("\nWARNING: Very high number of combinations may cause long processing time")
        if total_combinations > 1000000:
            print("CRITICAL: Extremely high combinations count! Consider reducing parameter ranges.")

        confirm = input("\nProceed with these parameters? (y/n): ").lower().strip()
        if confirm == 'y':
            return {
                'atr_periods': atr_periods,
                'factors': factors,
                'buffer_distances': buffer_distances,
                'hardstop_distances': hardstop_distances,
                'enable_target': enable_target,
                'long_rr_ratios': long_rr_ratios,
                'short_rr_ratios': short_rr_ratios,
                'total_combinations': total_combinations
            }
        print("\nLet's reconfigure the parameters...")
        
        
        
class DirectoryManager:
    """Manages directory structure for the application"""
    def __init__(self):
        # Get the name of the currently executing Python file
        current_file = os.path.basename(__file__)  # Gets 'ST.py'
        file_name = os.path.splitext(current_file)[0]  # Removes '.py' extension
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Simplified directory name: just filename_timestamp
        self.base_dir = f"{file_name}_{self.timestamp}"
        self.csv_dumps_dir = os.path.join(self.base_dir, "csv_dumps")
        self.error_logs_dir = os.path.join(self.base_dir, "error_logs")
        self.final_results_dir = os.path.join(self.base_dir, "final_results")
        self.create_directory_structure()
        self.setup_logging()

    def create_directory_structure(self):
        """Creates the necessary directory structure for the application"""
        try:
            # Create main directories
            os.makedirs(self.base_dir, exist_ok=True)
            os.makedirs(self.csv_dumps_dir, exist_ok=True)
            os.makedirs(self.error_logs_dir, exist_ok=True)
            os.makedirs(self.final_results_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory structure: {str(e)}")
            raise

    def setup_logging(self):
        """Sets up logging configuration"""
        try:
            # Configure logging for processing errors
            processing_logger = logging.getLogger('processing_errors')
            processing_logger.setLevel(logging.ERROR)
            processing_handler = logging.FileHandler(
                os.path.join(self.error_logs_dir, 'processing_errors.log')
            )
            processing_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            processing_logger.addHandler(processing_handler)

            # Configure logging for system errors
            system_logger = logging.getLogger('system_errors')
            system_logger.setLevel(logging.ERROR)
            system_handler = logging.FileHandler(
                os.path.join(self.error_logs_dir, 'system_errors.log')
            )
            system_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            system_logger.addHandler(system_handler)

        except Exception as e:
            print(f"Error setting up logging: {str(e)}")
            raise

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
        return df_size * 2.5 * n_combinations  # Increased for additional arrays

    def calculate_optimal_batch_size(self, df_size, total_combinations):
        """Calculates optimal batch size based on available memory"""
        available_ram = self.get_available_ram()
        estimated_mem_per_combo = df_size * 2.5  # Increased for buffers and hard stops
        optimal_batch_size = int(available_ram / (estimated_mem_per_combo * 1.2))
        return min(optimal_batch_size, BATCH_SIZE, total_combinations)

def load_ohlc_data(file_path):
    """Load and prepare OHLC data from file"""
    print(f"Attempting to load data from {file_path}...")

    try:
        if file_path.endswith('.csv'):
            # Read a small sample first to examine the columns
            sample = pd.read_csv(file_path, nrows=5)
            print(f"File columns: {list(sample.columns)}")

            # Check for separate date and time columns
            has_date_col = any('date' in col.lower() for col in sample.columns)
            has_time_col = any('time' in col.lower() and 'datetime' not in col.lower() for col in sample.columns)

            # Read the full file
            df = pd.read_csv(file_path)

            # Map columns to standard names
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

            # Apply column mapping for OHLC
            df = df.rename(columns=column_mapping)

            # Handle datetime creation
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

            # Check for required columns
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

        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Set datetime as index
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        df.dropna(subset=numeric_cols, inplace=True)

        print(f"Successfully loaded data with shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")

        return df

    except Exception as e:
        print(f"Error during data loading: {str(e)}")
        raise
        

# Updated timestamp constant
CURRENT_UTC = "2025-05-27 01:03:19"
CURRENT_USER = "arullr001"

@cuda.jit
def calculate_supertrend_pine_cuda_kernel(high, low, close, atr_period, factor, buffer_distance, hardstop_distance, 
                                         up, dn, trend, up_trend_buffer, down_trend_buffer, up_trend_hardstop, down_trend_hardstop):
    """
    CUDA kernel for Pine Script SuperTrend calculation with buffers and hard stops
    """
    i = cuda.grid(1)
    if i >= len(close):
        return

    # Initialize thread-local variables
    hl2 = (high[i] + low[i]) / 2

    # Calculate TR for this point
    if i == 0:
        tr = high[0] - low[0]
    else:
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )

    # Calculate ATR for this point
    atr_sum = 0.0
    count = 0
    for j in range(max(0, i - atr_period + 1), i + 1):
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

    # Calculate basic SuperTrend components
    up_basic = hl2 - (factor * atr)
    dn_basic = hl2 + (factor * atr)

    # Calculate SuperTrend values
    if i == 0:
        up[i] = up_basic
        dn[i] = dn_basic
        trend[i] = 1  # Start with uptrend
    else:
        # Calculate up
        if close[i-1] > up[i-1]:
            up[i] = max(up_basic, up[i-1])
        else:
            up[i] = up_basic

        # Calculate down
        if close[i-1] < dn[i-1]:
            dn[i] = min(dn_basic, dn[i-1])
        else:
            dn[i] = dn_basic

        # Determine trend
        if close[i] > dn[i-1]:
            trend[i] = 1  # Uptrend
        elif close[i] < up[i-1]:
            trend[i] = -1  # Downtrend
        else:
            trend[i] = trend[i-1]  # Maintain previous trend

    # Calculate buffer lines based on trend
    if trend[i] == 1:  # Uptrend
        up_trend_buffer[i] = up[i] + buffer_distance
        down_trend_buffer[i] = 0  # Not applicable in uptrend
        up_trend_hardstop[i] = up[i] - hardstop_distance
        down_trend_hardstop[i] = 0  # Not applicable in uptrend
    else:  # Downtrend
        up_trend_buffer[i] = 0  # Not applicable in downtrend
        down_trend_buffer[i] = dn[i] - buffer_distance
        up_trend_hardstop[i] = 0  # Not applicable in downtrend
        down_trend_hardstop[i] = dn[i] + hardstop_distance

def calculate_supertrend_pine_gpu(df, atr_period, factor, buffer_distance, hardstop_distance):
    """
    GPU-accelerated Pine Script SuperTrend calculation using CUDA
    """
    df = df.copy()

    # Extract arrays for GPU
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    n = len(close)

    # Convert to contiguous arrays for CUDA
    high_gpu = cuda.to_device(high.astype(np.float64))
    low_gpu = cuda.to_device(low.astype(np.float64))
    close_gpu = cuda.to_device(close.astype(np.float64))

    # Create output arrays on GPU
    up_gpu = cuda.device_array(n, dtype=np.float64)
    dn_gpu = cuda.device_array(n, dtype=np.float64)
    trend_gpu = cuda.device_array(n, dtype=np.float64)
    up_trend_buffer_gpu = cuda.device_array(n, dtype=np.float64)
    down_trend_buffer_gpu = cuda.device_array(n, dtype=np.float64)
    up_trend_hardstop_gpu = cuda.device_array(n, dtype=np.float64)
    down_trend_hardstop_gpu = cuda.device_array(n, dtype=np.float64)

    # Configure CUDA kernel
    blocks_per_grid = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    # Launch CUDA kernel
    calculate_supertrend_pine_cuda_kernel[blocks_per_grid, THREADS_PER_BLOCK](
        high_gpu, low_gpu, close_gpu, atr_period, factor, buffer_distance, hardstop_distance,
        up_gpu, dn_gpu, trend_gpu, up_trend_buffer_gpu, down_trend_buffer_gpu, 
        up_trend_hardstop_gpu, down_trend_hardstop_gpu
    )

    # Copy results back from GPU
    up = up_gpu.copy_to_host()
    dn = dn_gpu.copy_to_host()
    trend = trend_gpu.copy_to_host()
    up_trend_buffer = up_trend_buffer_gpu.copy_to_host()
    down_trend_buffer = down_trend_buffer_gpu.copy_to_host()
    up_trend_hardstop = up_trend_hardstop_gpu.copy_to_host()
    down_trend_hardstop = down_trend_hardstop_gpu.copy_to_host()

    # Add results to DataFrame
    df['up'] = up
    df['dn'] = dn
    df['trend'] = trend
    df['supertrend'] = np.where(trend == 1, up, dn)
    df['up_trend_buffer'] = up_trend_buffer
    df['down_trend_buffer'] = down_trend_buffer
    df['up_trend_hardstop'] = up_trend_hardstop
    df['down_trend_hardstop'] = down_trend_hardstop

    # Generate Pine Script signals - vectorized operations
    df['buy_signal'] = (df['trend'] == 1) & (df['close'] > df['supertrend']) & (df['close'] <= df['up_trend_buffer'])
    df['sell_signal'] = (df['trend'] == -1) & (df['close'] < df['supertrend']) & (df['close'] >= df['down_trend_buffer'])

    # Clean up GPU memory
    try:
        del (high_gpu, low_gpu, close_gpu, up_gpu, dn_gpu, trend_gpu, 
             up_trend_buffer_gpu, down_trend_buffer_gpu, up_trend_hardstop_gpu, down_trend_hardstop_gpu)
        cuda.current_context().deallocations.clear()
        import gc
        gc.collect()
    except Exception as e:
        print(f"Warning: Could not clear GPU memory: {e}", end='\r')

    return df

@numba.jit(nopython=True)
def calculate_supertrend_pine_numba(high, low, close, atr_period, factor, buffer_distance, hardstop_distance):
    """
    Optimized Numba implementation of Pine Script SuperTrend calculation for CPU
    """
    n = len(close)
    up = np.zeros(n, dtype=np.float64)
    dn = np.zeros(n, dtype=np.float64)
    trend = np.zeros(n, dtype=np.int64)
    up_trend_buffer = np.zeros(n, dtype=np.float64)
    down_trend_buffer = np.zeros(n, dtype=np.float64)
    up_trend_hardstop = np.zeros(n, dtype=np.float64)
    down_trend_hardstop = np.zeros(n, dtype=np.float64)

    for i in range(n):
        hl2 = (high[i] + low[i]) / 2

        # Calculate TR and ATR
        if i == 0:
            tr = high[0] - low[0]
            atr = tr
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            # Simple moving average for ATR
            if i < atr_period:
                atr_sum = 0.0
                for j in range(i + 1):
                    if j == 0:
                        atr_sum += high[0] - low[0]
                    else:
                        atr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr = atr_sum / (i + 1)
            else:
                atr_sum = 0.0
                for j in range(i - atr_period + 1, i + 1):
                    if j == 0:
                        atr_sum += high[0] - low[0]
                    else:
                        atr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr = atr_sum / atr_period

        # Determine basic up and down levels
        up_basic = hl2 - (factor * atr)
        dn_basic = hl2 + (factor * atr)

        if i == 0:
            up[i] = up_basic
            dn[i] = dn_basic
            trend[i] = 1  # Start with uptrend
        else:
            # Update up and down levels
            up[i] = max(up_basic, up[i-1]) if close[i-1] > up[i-1] else up_basic
            dn[i] = min(dn_basic, dn[i-1]) if close[i-1] < dn[i-1] else dn_basic

            # Update trend
            if close[i] > dn[i-1]:
                trend[i] = 1  # Uptrend
            elif close[i] < up[i-1]:
                trend[i] = -1  # Downtrend
            else:
                trend[i] = trend[i-1]  # Maintain previous trend

        # Calculate buffer and hard stop lines based on current trend
        if trend[i] == 1:  # Uptrend
            up_trend_buffer[i] = up[i] + buffer_distance
            down_trend_buffer[i] = 0  # Not applicable
            up_trend_hardstop[i] = up[i] - hardstop_distance
            down_trend_hardstop[i] = 0  # Not applicable
        else:  # Downtrend
            up_trend_buffer[i] = 0  # Not applicable
            down_trend_buffer[i] = dn[i] - buffer_distance
            up_trend_hardstop[i] = 0  # Not applicable
            down_trend_hardstop[i] = dn[i] + hardstop_distance

    return up, dn, trend, up_trend_buffer, down_trend_buffer, up_trend_hardstop, down_trend_hardstop

def calculate_supertrend_pine_cpu(df, atr_period, factor, buffer_distance, hardstop_distance):
    """CPU version of Pine Script SuperTrend calculation"""
    df = df.copy()

    # Extract numpy arrays for Numba function
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    # Calculate using Numba function
    up, dn, trend, up_trend_buffer, down_trend_buffer, up_trend_hardstop, down_trend_hardstop = calculate_supertrend_pine_numba(
        high, low, close, atr_period, factor, buffer_distance, hardstop_distance
    )

    # Add results to DataFrame
    df['up'] = up
    df['dn'] = dn
    df['trend'] = trend
    df['supertrend'] = np.where(trend == 1, up, dn)
    df['up_trend_buffer'] = up_trend_buffer
    df['down_trend_buffer'] = down_trend_buffer
    df['up_trend_hardstop'] = up_trend_hardstop
    df['down_trend_hardstop'] = down_trend_hardstop

    # Generate Pine Script signals - vectorized operations
    df['buy_signal'] = (df['trend'] == 1) & (df['close'] > df['supertrend']) & (df['close'] <= df['up_trend_buffer'])
    df['sell_signal'] = (df['trend'] == -1) & (df['close'] < df['supertrend']) & (df['close'] >= df['down_trend_buffer'])

    return df

def calculate_supertrend_pine(df, atr_period, factor, buffer_distance, hardstop_distance):
    """Wrapper function that chooses between CPU or GPU implementation for Pine Script strategy"""
    try:
        if cuda.is_available():
            try:
                # Get GPU device
                device = cuda.get_current_device()
                
                # Try to get memory info using different methods
                try:
                    free_mem = device.mem_info()[0]  # First try this method
                except AttributeError:
                    try:
                        free_mem = device.memory_info().free  # Then try this
                    except AttributeError:
                        try:
                            ctx = cuda.current_context()
                            free_mem = ctx.get_memory_info().free
                        except AttributeError:
                            # If we can't get memory info, assume we don't have enough
                            print("Unable to determine GPU memory, falling back to CPU", end='\r')
                            return calculate_supertrend_pine_cpu(df, atr_period, factor, buffer_distance, hardstop_distance)

                # Calculate required memory (increased for additional arrays)
                data_size = len(df) * 8 * 7  # 7 arrays instead of 5
                
                # Check if we have enough GPU memory (with buffer)
                if free_mem > data_size * 3:
                    print("Using GPU acceleration for Pine Script SuperTrend calculation", end='\r')
                    return calculate_supertrend_pine_gpu(df, atr_period, factor, buffer_distance, hardstop_distance)
                else:
                    print("Not enough GPU memory, falling back to CPU", end='\r')
                    return calculate_supertrend_pine_cpu(df, atr_period, factor, buffer_distance, hardstop_distance)
            
            except Exception as e:
                print(f"GPU initialization failed, falling back to CPU: {str(e)}", end='\r')
                return calculate_supertrend_pine_cpu(df, atr_period, factor, buffer_distance, hardstop_distance)
        
        return calculate_supertrend_pine_cpu(df, atr_period, factor, buffer_distance, hardstop_distance)
    
    except Exception as e:
        print(f"Error in Pine Script SuperTrend calculation, falling back to CPU: {str(e)}", end='\r')
        return calculate_supertrend_pine_cpu(df, atr_period, factor, buffer_distance, hardstop_distance)

def cleanup_gpu_memory():
    """Clean up GPU memory to prevent leaks"""
    try:
        if cuda.is_available():
            import gc
            gc.collect()
            cuda.current_context().deallocations.clear()
            print("GPU memory cleaned", end='\r')
    except Exception as e:
        print(f"GPU memory cleanup error: {e}", end='\r')
        
        


# Updated timestamp constant
CURRENT_UTC = "2025-05-27 01:08:35"
CURRENT_USER = "arullr001"

def backtest_supertrend_pine(df, atr_period, factor, buffer_distance, hardstop_distance, enable_target=True, long_rr=2, short_rr=2):
    """Enhanced backtesting of the Pine Script SuperTrend strategy"""
    
    # Calculate Pine Script SuperTrend and signals
    st_df = calculate_supertrend_pine(df, atr_period, factor, buffer_distance, hardstop_distance)

    # Convert to numpy arrays for faster operations
    close = st_df['close'].values
    high = st_df['high'].values
    low = st_df['low'].values
    supertrend = st_df['supertrend'].values
    trend = st_df['trend'].values
    buy_signals = st_df['buy_signal'].values
    sell_signals = st_df['sell_signal'].values
    up_hardstop = st_df['up_trend_hardstop'].values
    down_hardstop = st_df['down_trend_hardstop'].values
    dates = st_df.index.to_numpy()

    # Initialize trade tracking variables
    trades = []
    trade_number = 0
    in_long = False
    in_short = False
    entry_price = 0
    entry_time = None
    entry_supertrend = 0
    target_price = 0

    # Loop through data for backtesting - optimized version
    for i in range(1, len(st_df)):
        current_price = close[i]
        current_time = dates[i]
        current_high = high[i]
        current_low = low[i]
        current_trend = trend[i]
        current_supertrend = supertrend[i]

        # Handle existing long position
        if in_long:
            exit_condition = False
            exit_price = current_price
            exit_reason = ""

            # Check hard stop first (immediate exit)
            if current_low <= up_hardstop[i] and up_hardstop[i] > 0:
                exit_condition = True
                exit_price = up_hardstop[i]  # Assume hit at hard stop level
                exit_reason = "Hard Stop"
            
            # Check target hit (if enabled)
            elif enable_target and current_high >= target_price:
                exit_condition = True
                exit_price = target_price
                exit_reason = "Target"
            
            # Check trend change (SuperTrend color change)
            elif current_trend == -1:  # Trend changed to downtrend
                exit_condition = True
                exit_price = current_supertrend  # Exit at SuperTrend level
                exit_reason = "Trend Change"

            if exit_condition:
                points = exit_price - entry_price
                trades.append({
                    'trade_number': trade_number,
                    'direction': 'long',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': current_time,
                    'exit_price': exit_price,
                    'points': points,
                    'exit_reason': exit_reason
                })
                in_long = False

        # Handle existing short position
        if in_short:
            exit_condition = False
            exit_price = current_price
            exit_reason = ""

            # Check hard stop first (immediate exit)
            if current_high >= down_hardstop[i] and down_hardstop[i] > 0:
                exit_condition = True
                exit_price = down_hardstop[i]  # Assume hit at hard stop level
                exit_reason = "Hard Stop"
            
            # Check target hit (if enabled)
            elif enable_target and current_low <= target_price:
                exit_condition = True
                exit_price = target_price
                exit_reason = "Target"
            
            # Check trend change (SuperTrend color change)
            elif current_trend == 1:  # Trend changed to uptrend
                exit_condition = True
                exit_price = current_supertrend  # Exit at SuperTrend level
                exit_reason = "Trend Change"

            if exit_condition:
                points = entry_price - exit_price
                trades.append({
                    'trade_number': trade_number,
                    'direction': 'short',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': current_time,
                    'exit_price': exit_price,
                    'points': points,
                    'exit_reason': exit_reason
                })
                in_short = False

        # Check for new entries (only if not in position)
        if not in_long and not in_short:
            if buy_signals[i]:
                trade_number += 1
                in_long = True
                entry_price = current_price
                entry_time = current_time
                entry_supertrend = current_supertrend
                
                # Calculate target price using R:R ratio if enabled
                if enable_target:
                    risk = entry_price - entry_supertrend  # Risk is distance to SuperTrend
                    target_price = entry_price + (risk * long_rr)
                else:
                    target_price = 0

            elif sell_signals[i]:
                trade_number += 1
                in_short = True
                entry_price = current_price
                entry_time = current_time
                entry_supertrend = current_supertrend
                
                # Calculate target price using R:R ratio if enabled
                if enable_target:
                    risk = entry_supertrend - entry_price  # Risk is distance to SuperTrend
                    target_price = entry_price - (risk * short_rr)
                else:
                    target_price = 0

    # Close any open position at the end of testing
    if in_long:
        points = close[-1] - entry_price
        trades.append({
            'trade_number': trade_number,
            'direction': 'long',
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': dates[-1],
            'exit_price': close[-1],
            'points': points,
            'exit_reason': "End of Data"
        })

    if in_short:
        points = entry_price - close[-1]
        trades.append({
            'trade_number': trade_number,
            'direction': 'short',
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': dates[-1],
            'exit_price': close[-1],
            'points': points,
            'exit_reason': "End of Data"
        })

    # Calculate performance metrics
    total_profit = sum(trade['points'] for trade in trades) if trades else 0
    win_trades = [trade for trade in trades if trade['points'] > 0]
    win_rate = len(win_trades) / len(trades) if trades else 0

    return {
        'parameters': {
            'atr_period': atr_period,
            'factor': factor,
            'buffer_distance': buffer_distance,
            'hardstop_distance': hardstop_distance,
            'enable_target': enable_target,
            'long_rr': long_rr,
            'short_rr': short_rr
        },
        'total_profit': total_profit,
        'trade_count': len(trades),
        'win_rate': win_rate,
        'trades': trades
    }

def process_param_combo_pine(args):
    """Process a single parameter combination for Pine Script strategy"""
    try:
        df, atr_period, factor, buffer_distance, hardstop_distance, enable_target, long_rr, short_rr = args
        result = backtest_supertrend_pine(
            df, 
            atr_period=atr_period,
            factor=factor,
            buffer_distance=buffer_distance,
            hardstop_distance=hardstop_distance,
            enable_target=enable_target,
            long_rr=long_rr,
            short_rr=short_rr
        )
        return result
    except Exception as e:
        logging.getLogger('processing_errors').error(
            f"Error processing combination (atr={atr_period}, factor={factor}, "
            f"buffer={buffer_distance}, hardstop={hardstop_distance}, "
            f"target={enable_target}, long_rr={long_rr}, short_rr={short_rr}): {str(e)}"
        )
        return {
            'parameters': {
                'atr_period': atr_period,
                'factor': factor,
                'buffer_distance': buffer_distance,
                'hardstop_distance': hardstop_distance,
                'enable_target': enable_target,
                'long_rr': long_rr,
                'short_rr': short_rr
            },
            'total_profit': float('-inf'),
            'trade_count': 0,
            'win_rate': 0,
            'trades': []
        }

def cleanup_memory():
    """Performs memory cleanup operations"""
    try:
        # Clear Python memory
        import gc
        gc.collect()

        # Clear GPU memory if available
        if cuda.is_available():
            cleanup_gpu_memory()

    except Exception as e:
        logging.getLogger('system_errors').error(f"Memory cleanup error: {str(e)}")

class EnhancedBatchProcessor:
    """Enhanced batch processor with progress tracking and early termination"""
    def __init__(self, directory_manager, memory_manager):
        self.dir_manager = directory_manager
        self.mem_manager = memory_manager
        self.current_batch = 0
        self.processing_logger = logging.getLogger('processing_errors')
        self.system_logger = logging.getLogger('system_errors')
        self.current_utc = CURRENT_UTC
        self.user = CURRENT_USER
        self.best_profit = float('-inf')
        self.current_best_params = None
        self.start_time = time.time()
        self.processed_combinations = 0

    def save_batch_results(self, results, batch_num):
        """Saves batch results to CSV with enhanced metadata"""
        batch_file = os.path.join(self.dir_manager.csv_dumps_dir, f'batch_{batch_num}.csv')
        metadata_file = os.path.join(self.dir_manager.csv_dumps_dir, f'batch_{batch_num}_metadata.json')
        
        try:
            # Convert results to DataFrame for easier CSV handling
            results_data = []
            for r in results:
                if r['trades']:  # Only include valid results
                    row = {
                        'atr_period': r['parameters']['atr_period'],
                        'factor': r['parameters']['factor'],
                        'buffer_distance': r['parameters']['buffer_distance'],
                        'hardstop_distance': r['parameters']['hardstop_distance'],
                        'enable_target': r['parameters']['enable_target'],
                        'long_rr': r['parameters']['long_rr'],
                        'short_rr': r['parameters']['short_rr'],
                        'total_profit': r['total_profit'],
                        'trade_count': r['trade_count'],
                        'win_rate': r['win_rate']
                    }
                    results_data.append(row)
                    
                    # Update best result tracking
                    if r['total_profit'] > self.best_profit:
                        self.best_profit = r['total_profit']
                        self.current_best_params = r['parameters'].copy()

            if results_data:
                df = pd.DataFrame(results_data)
                df.to_csv(batch_file, index=False)

                # Save enhanced metadata
                elapsed_time = time.time() - self.start_time
                self.processed_combinations += len(results_data)
                
                metadata = {
                    'batch_number': batch_num,
                    'processed_at': self.current_utc,
                    'processed_by': self.user,
                    'combinations_count': len(results_data),
                    'memory_usage_mb': self.mem_manager.get_current_ram_usage() / (1024 * 1024),
                    'batch_success': True,
                    'elapsed_time_seconds': elapsed_time,
                    'processed_combinations_total': self.processed_combinations,
                    'current_best_profit': self.best_profit,
                    'current_best_params': self.current_best_params
                }
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=4)
                
                return True
            return False

        except Exception as e:
            self.processing_logger.error(f"Error saving batch {batch_num}: {str(e)}")
            return False

    def process_batch_with_progress(self, df, param_combinations, batch_start, batch_size, max_workers=4, total_combinations=0):
        """Enhanced batch processing with progress tracking and ETA"""
        batch_end = min(batch_start + batch_size, len(param_combinations))
        batch_combinations = param_combinations[batch_start:batch_end]
        
        # Display progress information
        elapsed_time = time.time() - self.start_time
        progress_percent = (self.processed_combinations / total_combinations * 100) if total_combinations > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"Processing Batch {self.current_batch + 1}")
        print(f"Combinations: {batch_start + 1} to {batch_end} of {total_combinations}")
        print(f"Progress: {progress_percent:.1f}% | Elapsed: {elapsed_time/3600:.1f}h")
        if self.current_best_params:
            print(f"Current Best Profit: {self.best_profit:.2f}")
            print(f"Best Params: ATR={self.current_best_params['atr_period']}, "
                  f"Factor={self.current_best_params['factor']:.2f}, "
                  f"Buffer={self.current_best_params['buffer_distance']:.1f}")
        print(f"{'='*60}")
        
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                executor.submit(
                    process_param_combo_pine,
                    (df.copy(), atr_period, factor, buffer_distance, hardstop_distance, 
                     enable_target, long_rr, short_rr)
                )
                for atr_period, factor, buffer_distance, hardstop_distance, enable_target, long_rr, short_rr in batch_combinations
            ]

            for future in tqdm(concurrent.futures.as_completed(tasks), 
                             total=len(tasks), 
                             desc=f"Batch {self.current_batch + 1} Progress",
                             ncols=80):
                try:
                    result = future.result()
                    if result['total_profit'] != float('-inf'):
                        results.append(result)
                except Exception as e:
                    self.processing_logger.error(f"Error in batch {self.current_batch + 1}: {str(e)}")

        return results

    def merge_batch_results(self):
        """Merges all batch results into final results with enhanced analysis"""
        all_results = []
        batch_files = glob.glob(os.path.join(self.dir_manager.csv_dumps_dir, 'batch_*.csv'))
        
        try:
            for file in batch_files:
                try:
                    df = pd.read_csv(file)
                    all_results.append(df)
                except Exception as e:
                    self.processing_logger.error(f"Error reading batch file {file}: {str(e)}")

            if all_results:
                final_df = pd.concat(all_results, ignore_index=True)
                final_df.sort_values('total_profit', ascending=False, inplace=True)
                
                # Save final results
                final_results_file = os.path.join(
                    self.dir_manager.final_results_dir, 
                    'best_combinations_pine_script.csv'
                )
                final_df.to_csv(final_results_file, index=False)
                
                # Enhanced final metadata
                total_time = time.time() - self.start_time
                final_metadata = {
                    'strategy_type': 'Pine Script SuperTrend with Buffers',
                    'processed_at': self.current_utc,
                    'processed_by': self.user,
                    'total_combinations': len(final_df),
                    'processing_time_hours': total_time / 3600,
                    'top_profit': float(final_df['total_profit'].max()),
                    'avg_profit': float(final_df['total_profit'].mean()),
                    'median_profit': float(final_df['total_profit'].median()),
                    'std_profit': float(final_df['total_profit'].std()),
                    'total_batches': len(batch_files),
                    'parameter_ranges': {
                        'atr_period': f"{final_df['atr_period'].min()}-{final_df['atr_period'].max()}",
                        'factor': f"{final_df['factor'].min():.2f}-{final_df['factor'].max():.2f}",
                        'buffer_distance': f"{final_df['buffer_distance'].min():.1f}-{final_df['buffer_distance'].max():.1f}",
                        'hardstop_distance': f"{final_df['hardstop_distance'].min():.1f}-{final_df['hardstop_distance'].max():.1f}"
                    }
                }
                
                with open(os.path.join(
                    self.dir_manager.final_results_dir, 
                    'final_metadata_pine_script.json'
                ), 'w') as f:
                    json.dump(final_metadata, f, indent=4)
                
                return final_df
                
            return pd.DataFrame()

        except Exception as e:
            self.system_logger.error(f"Error merging batch results: {str(e)}")
            return pd.DataFrame()
            
            
# Updated timestamp constant
CURRENT_UTC = "2025-05-27 01:15:51"
CURRENT_USER = "arullr001"

class ResultsAnalyzer:
    """Enhanced results analyzer for Pine Script SuperTrend strategy"""
    
    def __init__(self, directory_manager):
        self.dir_manager = directory_manager
        
    def generate_top_combinations_report(self, final_df, top_n=50):
        """Generate detailed report of top performing combinations"""
        if final_df.empty:
            print("No results to analyze")
            return
            
        try:
            # Get top combinations
            top_combinations = final_df.head(top_n).copy()
            
            # Enhanced analysis
            print(f"\n{'='*80}")
            print(f" TOP {min(top_n, len(final_df))} PINE SCRIPT SUPERTREND COMBINATIONS ".center(80, "="))
            print(f"{'='*80}")
            
            # Overall statistics
            print(f"\nOVERALL STATISTICS:")
            print(f"Total combinations tested: {len(final_df):,}")
            print(f"Profitable combinations: {len(final_df[final_df['total_profit'] > 0]):,}")
            print(f"Profitability rate: {len(final_df[final_df['total_profit'] > 0])/len(final_df)*100:.1f}%")
            print(f"Best profit: {final_df['total_profit'].max():.2f}")
            print(f"Worst loss: {final_df['total_profit'].min():.2f}")
            print(f"Average profit: {final_df['total_profit'].mean():.2f}")
            print(f"Median profit: {final_df['total_profit'].median():.2f}")
            
            # Parameter analysis
            print(f"\nPARAMETER ANALYSIS (Top {min(top_n, len(final_df))} combinations):")
            print(f"ATR Period range: {top_combinations['atr_period'].min()} - {top_combinations['atr_period'].max()}")
            print(f"Factor range: {top_combinations['factor'].min():.2f} - {top_combinations['factor'].max():.2f}")
            print(f"Buffer Distance range: {top_combinations['buffer_distance'].min():.1f} - {top_combinations['buffer_distance'].max():.1f}")
            print(f"Hard Stop Distance range: {top_combinations['hardstop_distance'].min():.1f} - {top_combinations['hardstop_distance'].max():.1f}")
            
            # Best parameters frequency analysis
            print(f"\nMOST FREQUENT PARAMETERS IN TOP {min(top_n, len(final_df))}:")
            print(f"Most common ATR Period: {top_combinations['atr_period'].mode().iloc[0] if not top_combinations['atr_period'].mode().empty else 'N/A'}")
            print(f"Most common Factor: {top_combinations['factor'].mode().iloc[0] if not top_combinations['factor'].mode().empty else 'N/A'}")
            print(f"Most common Buffer Distance: {top_combinations['buffer_distance'].mode().iloc[0] if not top_combinations['buffer_distance'].mode().empty else 'N/A'}")
            print(f"Most common Hard Stop Distance: {top_combinations['hardstop_distance'].mode().iloc[0] if not top_combinations['hardstop_distance'].mode().empty else 'N/A'}")
            
            # Top 10 detailed results
            print(f"\nTOP 10 COMBINATIONS DETAILS:")
            print(f"{'Rank':<4} {'ATR':<4} {'Factor':<7} {'Buffer':<7} {'HardStop':<9} {'Target':<7} {'LongRR':<7} {'ShortRR':<8} {'Profit':<8} {'Trades':<7} {'WinRate':<7}")
            print("-" * 80)
            
            for idx, row in top_combinations.head(10).iterrows():
                rank = idx + 1 if 'rank' not in top_combinations.columns else row.get('rank', idx + 1)
                print(f"{rank:<4} {int(row['atr_period']):<4} {row['factor']:<7.2f} {row['buffer_distance']:<7.1f} "
                      f"{row['hardstop_distance']:<9.1f} {str(row['enable_target']):<7} {int(row['long_rr']):<7} "
                      f"{int(row['short_rr']):<8} {row['total_profit']:<8.2f} {int(row['trade_count']):<7} {row['win_rate']:<7.1%}")
            
            # Save detailed top combinations
            top_file = os.path.join(self.dir_manager.final_results_dir, f'top_{top_n}_combinations_detailed.csv')
            top_combinations.to_csv(top_file, index=False)
            print(f"\nDetailed top {top_n} combinations saved to: {top_file}")
            
        except Exception as e:
            print(f"Error generating top combinations report: {str(e)}")
    
    def generate_parameter_sensitivity_analysis(self, final_df):
        """Generate parameter sensitivity analysis"""
        try:
            print(f"\n{'='*60}")
            print(" PARAMETER SENSITIVITY ANALYSIS ".center(60, "="))
            print(f"{'='*60}")
            
            # Correlation analysis
            numeric_cols = ['atr_period', 'factor', 'buffer_distance', 'hardstop_distance', 'long_rr', 'short_rr', 'total_profit']
            correlation_matrix = final_df[numeric_cols].corr()['total_profit'].sort_values(ascending=False)
            
            print("\nCORRELATION WITH PROFIT:")
            for param, corr in correlation_matrix.items():
                if param != 'total_profit':
                    print(f"{param:<20}: {corr:>8.3f}")
            
            # Parameter ranges for best performers (top 10%)
            top_10_percent = final_df.head(int(len(final_df) * 0.1))
            
            print(f"\nOPTIMAL PARAMETER RANGES (Top 10% performers):")
            print(f"ATR Period: {top_10_percent['atr_period'].min()} - {top_10_percent['atr_period'].max()}")
            print(f"Factor: {top_10_percent['factor'].min():.2f} - {top_10_percent['factor'].max():.2f}")
            print(f"Buffer Distance: {top_10_percent['buffer_distance'].min():.1f} - {top_10_percent['buffer_distance'].max():.1f}")
            print(f"Hard Stop Distance: {top_10_percent['hardstop_distance'].min():.1f} - {top_10_percent['hardstop_distance'].max():.1f}")
            
        except Exception as e:
            print(f"Error in parameter sensitivity analysis: {str(e)}")

class OptimizationEngine:
    """Main optimization engine for Pine Script SuperTrend strategy"""
    
    def __init__(self):
        self.dir_manager = DirectoryManager()
        self.mem_manager = MemoryManager()
        self.batch_processor = EnhancedBatchProcessor(self.dir_manager, self.mem_manager)
        self.results_analyzer = ResultsAnalyzer(self.dir_manager)
        
    def run_optimization(self, df, parameters):
        """Run the complete optimization process"""
        try:
            print(f"\n{'='*80}")
            print(" STARTING PINE SCRIPT SUPERTREND OPTIMIZATION ".center(80, "="))
            print(f"{'='*80}")
            print(f"Timestamp: {CURRENT_UTC}")
            print(f"User: {CURRENT_USER}")
            print(f"Data shape: {df.shape}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            
            # Generate parameter combinations
            param_combinations = list(product(
                parameters['atr_periods'],
                parameters['factors'],
                parameters['buffer_distances'],
                parameters['hardstop_distances'],
                [parameters['enable_target']],
                parameters['long_rr_ratios'],
                parameters['short_rr_ratios']
            ))
            
            total_combinations = len(param_combinations)
            print(f"Total combinations to process: {total_combinations:,}")
            
            # Calculate optimal batch size and worker count
            optimal_batch_size = self.mem_manager.calculate_optimal_batch_size(
                df.memory_usage(deep=True).sum(), total_combinations
            )
            max_workers = min(mp.cpu_count(), 8)  # Limit to prevent memory issues
            
            print(f"Optimal batch size: {optimal_batch_size}")
            print(f"Using {max_workers} workers")
            
            # Process in batches
            batch_start = 0
            self.batch_processor.start_time = time.time()
            
            while batch_start < total_combinations:
                try:
                    batch_results = self.batch_processor.process_batch_with_progress(
                        df, param_combinations, batch_start, optimal_batch_size, 
                        max_workers, total_combinations
                    )
                    
                    if batch_results:
                        save_success = self.batch_processor.save_batch_results(
                            batch_results, self.batch_processor.current_batch
                        )
                        if save_success:
                            print(f"Batch {self.batch_processor.current_batch + 1} completed successfully")
                        else:
                            print(f"Warning: Batch {self.batch_processor.current_batch + 1} save failed")
                    
                    batch_start += optimal_batch_size
                    self.batch_processor.current_batch += 1
                    
                    # Memory cleanup between batches
                    cleanup_memory()
                    
                    # Memory check
                    if not self.mem_manager.is_ram_available(optimal_batch_size * 1024 * 1024):
                        print("Warning: Low memory detected, reducing batch size")
                        optimal_batch_size = max(100, optimal_batch_size // 2)
                    
                except KeyboardInterrupt:
                    print("\nOptimization interrupted by user")
                    break
                except Exception as e:
                    print(f"Error in batch {self.batch_processor.current_batch + 1}: {str(e)}")
                    self.batch_processor.current_batch += 1
                    batch_start += optimal_batch_size
                    continue
            
            # Merge and analyze results
            print(f"\n{'='*60}")
            print(" MERGING AND ANALYZING RESULTS ".center(60, "="))
            print(f"{'='*60}")
            
            final_df = self.batch_processor.merge_batch_results()
            
            if not final_df.empty:
                # Generate comprehensive reports
                self.results_analyzer.generate_top_combinations_report(final_df, 50)
                self.results_analyzer.generate_parameter_sensitivity_analysis(final_df)
                
                # Final summary
                total_time = time.time() - self.batch_processor.start_time
                print(f"\n{'='*80}")
                print(" OPTIMIZATION COMPLETE ".center(80, "="))
                print(f"{'='*80}")
                print(f"Total processing time: {total_time/3600:.2f} hours")
                print(f"Total combinations processed: {len(final_df):,}")
                print(f"Results saved in: {self.dir_manager.base_dir}")
                print(f"Best combination profit: {final_df['total_profit'].max():.2f}")
                
                best_row = final_df.iloc[0]
                print(f"\nBEST COMBINATION:")
                print(f"ATR Period: {int(best_row['atr_period'])}")
                print(f"Factor: {best_row['factor']:.2f}")
                print(f"Buffer Distance: {best_row['buffer_distance']:.1f}")
                print(f"Hard Stop Distance: {best_row['hardstop_distance']:.1f}")
                print(f"Target-based exits: {best_row['enable_target']}")
                if best_row['enable_target']:
                    print(f"Long R:R ratio: {int(best_row['long_rr'])}")
                    print(f"Short R:R ratio: {int(best_row['short_rr'])}")
                print(f"Total Profit: {best_row['total_profit']:.2f}")
                print(f"Trade Count: {int(best_row['trade_count'])}")
                print(f"Win Rate: {best_row['win_rate']:.1%}")
                
                return final_df
            else:
                print("No valid results generated")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Critical error in optimization engine: {str(e)}")
            logging.getLogger('system_errors').error(f"Optimization engine error: {str(e)}")
            return pd.DataFrame()

def main():
    """Enhanced main function with improved user experience"""
    try:
        print(f"{'='*80}")
        print(" PINE SCRIPT SUPERTREND STRATEGY OPTIMIZER ".center(80, "="))
        print(f"{'='*80}")
        print(f"Version: Enhanced with Buffers and Hard Stops")
        print(f"Timestamp: {CURRENT_UTC}")
        print(f"User: {CURRENT_USER}")
        print(f"{'='*80}")
        
        # Check system requirements
        print("\nSYSTEM REQUIREMENTS CHECK:")
        print(f"CPU cores: {mp.cpu_count()}")
        print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        
        # Check GPU availability
        if cuda.is_available():
            try:
                device = cuda.get_current_device()
                print(f"GPU: Available ({device.name.decode() if hasattr(device.name, 'decode') else 'CUDA Device'})")
            except:
                print("GPU: Available (CUDA)")
        else:
            print("GPU: Not available (will use CPU)")
        
        # Get data directory
        while True:
            data_dir = input("\nEnter the directory path containing OHLC data files: ").strip()
            if os.path.exists(data_dir):
                break
            print("Directory not found. Please enter a valid path.")
        
        # Find and select data files
        data_files = find_ohlc_files(data_dir)
        selected_files = select_files_to_process(data_files)
        
        if not selected_files:
            print("No files selected. Exiting.")
            return
        
        # Get parameter inputs
        parameters = get_parameter_inputs()
        
        # Process each selected file
        optimization_engine = OptimizationEngine()
        
        for file_path in selected_files:
            try:
                print(f"\n{'='*80}")
                print(f" PROCESSING FILE: {os.path.basename(file_path)} ".center(80, "="))
                print(f"{'='*80}")
                
                # Load data
                df = load_ohlc_data(file_path)
                
                if df.empty:
                    print(f"No data loaded from {file_path}, skipping...")
                    continue
                
                # Run optimization
                results_df = optimization_engine.run_optimization(df, parameters)
                
                if not results_df.empty:
                    print(f"\nOptimization completed for {os.path.basename(file_path)}")
                else:
                    print(f"\nNo valid results for {os.path.basename(file_path)}")
                
                # Memory cleanup between files
                cleanup_memory()
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                logging.getLogger('system_errors').error(f"File processing error for {file_path}: {str(e)}")
                continue
        
        print(f"\n{'='*80}")
        print(" ALL OPTIMIZATIONS COMPLETE ".center(80, "="))
        print(f"{'='*80}")
        print(f"Results directory: {optimization_engine.dir_manager.base_dir}")
        print("Check the 'final_results' folder for comprehensive reports.")
        print("Check the 'csv_dumps' folder for batch-wise results.")
        print("Check the 'error_logs' folder for any processing errors.")
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Critical error in main: {str(e)}")
        logging.getLogger('system_errors').error(f"Main function error: {str(e)}")
    finally:
        # Final cleanup
        cleanup_memory()
        print("\nProgram finished.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()
    finally:
        # Ensure GPU memory cleanup on exit
        try:
            cleanup_gpu_memory()
        except:
            pass
            
            
