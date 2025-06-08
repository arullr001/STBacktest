# SuperTrend Dynamic ATR Buffer Strategy
# Version: 1.0
# Date: 2025-06-07
# Author: GitHub Copilot for arullr001

import os
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime, timedelta
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
CURRENT_UTC = "2025-06-07 23:17:24"
CURRENT_USER = "arullr001"

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
    """Get all parameter inputs from user with validation for SuperTrend Dynamic Buffer strategy"""
    while True:
        print("\n" + "=" * 60)
        print(" SUPERTREND DYNAMIC ATR BUFFER STRATEGY CONFIGURATION ".center(60, "="))
        print("=" * 60)

        # ATR Period range inputs
        print("\nEnter ATR Period range (Integer values):")
        atr_start = get_int_input("  Start value (min 5, max 100): ", 5, 100)
        atr_end = get_int_input(f"  End value (min {atr_start}, max 100): ", atr_start, 100)
        atr_step = get_int_input("  Step/increment (min 1): ", 1)

        # Factor range inputs
        print("\nEnter Factor range (Float values):")
        factor_start = get_float_input("  Start value (min 2.0, max 30.0): ", 2.0, 30.0)
        factor_end = get_float_input(f"  End value (min {factor_start}, max 30.0): ", factor_start, 30.0)
        factor_step = get_float_input("  Step/increment (min 0.01): ", 0.01)

        # ATR Buffer Multiplier range inputs
        print("\nEnter ATR Buffer Multiplier range (Float values):")
        buffer_start = get_float_input("  Start value (min 0.01, max 5.0): ", 0.01, 5.0)
        buffer_end = get_float_input(f"  End value (min {buffer_start}, max 5.0): ", buffer_start, 5.0)
        buffer_step = get_float_input("  Step/increment (min 0.01): ", 0.01)

        # Hard Stop Distance range inputs
        print("\nEnter Hard Stop Distance range (Float values):")
        hardstop_start = get_float_input("  Start value (min 10, max 400): ", 10, 400)
        hardstop_end = get_float_input(f"  End value (min {hardstop_start}, max 400): ", hardstop_start, 400)
        hardstop_step = get_float_input("  Step/increment (min 1): ", 1)

        # Note: Time-based exit is always enabled as per the strategy
        print("\nTime-based Exit Configuration:")
        print("  Time-based exits are ENABLED - trades will exit after 1 day at 5:29 PM IST (11:59 AM UTC)")

        # Generate the ranges
        atr_periods = list(range(atr_start, atr_end + 1, atr_step))
        factors = [round(x, 2) for x in np.arange(factor_start, factor_end + (factor_step / 2), factor_step)]
        buffer_multipliers = [round(x, 2) for x in np.arange(buffer_start, buffer_end + (buffer_step / 2), buffer_step)]
        hardstop_distances = [round(x, 1) for x in np.arange(hardstop_start, hardstop_end + (hardstop_step / 2), hardstop_step)]

        # Calculate total combinations
        total_combinations = (len(atr_periods) * len(factors) * len(buffer_multipliers) * len(hardstop_distances))

        # Summary
        print("\n" + "=" * 60)
        print(" PARAMETER SUMMARY ".center(60, "="))
        print("=" * 60)
        print(f"ATR Period range: {atr_start} to {atr_end} (step {atr_step}) - {len(atr_periods)} values")
        print(f"Factor range: {factor_start} to {factor_end} (step {factor_step}) - {len(factors)} values")
        print(f"ATR Buffer Multiplier range: {buffer_start} to {buffer_end} (step {buffer_step}) - {len(buffer_multipliers)} values")
        print(f"Hard Stop Distance range: {hardstop_start} to {hardstop_end} (step {hardstop_step}) - {len(hardstop_distances)} values")
        print(f"Time-based exit: Enabled (after 1 day at 5:29 PM IST)")
        
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
                'buffer_multipliers': buffer_multipliers,
                'hardstop_distances': hardstop_distances,
                'enable_time_exit': True,  # Always enabled as per the strategy
                'total_combinations': total_combinations
            }
        print("\nLet's reconfigure the parameters...")
		
		
# Part 2: Directory Manager, Memory Manager, and Data Loading

class DirectoryManager:
    """Manages directory structure for the application"""
    def __init__(self):
        # Get the name of the currently executing Python file
        current_file = os.path.basename(__file__)  # Gets the file name
        file_name = os.path.splitext(current_file)[0]  # Removes extension
        
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
        estimated_mem_per_combo = df_size * 2.5  # Increased for dynamic buffer
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

        # Add time components for time-based exit analysis
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        
        # Flag for exit time (5:29 PM IST = 11:59 AM UTC)
        df['is_exit_time'] = (df['hour'] == 11) & (df['minute'] == 59)

        print(f"Successfully loaded data with shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")

        return df

    except Exception as e:
        print(f"Error during data loading: {str(e)}")
        raise

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
		
		
		
# Part 3: GPU-Accelerated SuperTrend Implementation

# CUDA kernel for ATR calculation
@cuda.jit
def calculate_atr_kernel(high, low, close, atr_period, result):
    """CUDA kernel for ATR calculation"""
    i = cuda.grid(1)
    if i < high.size:
        # Skip until we have enough data for the period
        if i < atr_period:
            result[i] = 0.0
            return

        # First TR value is high-low of first bar
        if i == atr_period:
            # Calculate first TR for first ATR
            tr_sum = 0.0
            for j in range(1, atr_period + 1):
                h_l = high[j] - low[j]
                h_pc = abs(high[j] - close[j - 1])
                l_pc = abs(low[j] - close[j - 1])
                tr = max(h_l, h_pc, l_pc)
                tr_sum += tr
            result[i] = tr_sum / atr_period
            return

        # For subsequent values, use EMA-style calculation
        if i > atr_period:
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr = max(h_l, h_pc, l_pc)
            result[i] = ((result[i - 1] * (atr_period - 1)) + tr) / atr_period

# CUDA kernel for dynamic SuperTrend calculation with buffer zones
@cuda.jit
def calculate_supertrend_kernel(high, low, close, atr, factor, atr_buffer_multiplier, hardstop_distance, direction, supertrend, up_trend_buffer, down_trend_buffer, up_trend_hardstop, down_trend_hardstop):
    """CUDA kernel for SuperTrend calculation with dynamic buffer zones based on ATR"""
    i = cuda.grid(1)
    
    if i < high.size:
        # Initialize with default values for first few bars
        if i <= 1:
            direction[i] = 0.0
            supertrend[i] = close[i]
            up_trend_buffer[i] = 0.0
            down_trend_buffer[i] = 0.0
            up_trend_hardstop[i] = 0.0
            down_trend_hardstop[i] = 0.0
            return

        # ATR multiplied by factor
        factor_atr = atr[i] * factor
        
        # Calculate upper and lower bands
        upper_band = (high[i] + low[i]) / 2.0 + factor_atr
        lower_band = (high[i] + low[i]) / 2.0 - factor_atr
        
        # Calculate dynamic buffer using ATR
        dynamic_buffer = atr[i] * atr_buffer_multiplier

        # Calculate SuperTrend value based on previous direction
        if close[i-1] > supertrend[i-1]:
            # Previous bar was an uptrend
            curr_supertrend = max(lower_band, supertrend[i-1])
            curr_direction = -1.0  # Up trend (uses -1 for up, +1 for down)
        elif close[i-1] < supertrend[i-1]:
            # Previous bar was a downtrend
            curr_supertrend = min(upper_band, supertrend[i-1])
            curr_direction = 1.0  # Down trend
        else:
            # No change
            curr_supertrend = supertrend[i-1]
            curr_direction = direction[i-1]
        
        # Check for trend reversals based on current close
        if curr_direction == -1.0 and close[i] < curr_supertrend:
            curr_direction = 1.0  # Flipped to down trend
        elif curr_direction == 1.0 and close[i] > curr_supertrend:
            curr_direction = -1.0  # Flipped to up trend
        
        # Store results
        direction[i] = curr_direction
        supertrend[i] = curr_supertrend
        
        # Calculate buffer lines based on trend direction
        if curr_direction < 0:  # Uptrend
            up_trend_buffer[i] = curr_supertrend + dynamic_buffer
            down_trend_buffer[i] = 0.0  # Not applicable in uptrend
            up_trend_hardstop[i] = curr_supertrend - hardstop_distance
            down_trend_hardstop[i] = 0.0  # Not applicable in uptrend
        else:  # Downtrend
            up_trend_buffer[i] = 0.0  # Not applicable in downtrend
            down_trend_buffer[i] = curr_supertrend - dynamic_buffer
            up_trend_hardstop[i] = 0.0  # Not applicable in downtrend
            down_trend_hardstop[i] = curr_supertrend + hardstop_distance

class GPUAcceleratedSuperTrend:
    """Implements SuperTrend calculation using GPU acceleration"""
    
    def __init__(self, threadsperblock=THREADS_PER_BLOCK):
        self.threadsperblock = threadsperblock
        
    def calculate_atr(self, df, atr_period):
        """Calculate ATR using GPU acceleration"""
        try:
            # Check for cuda availability
            if not cuda.is_available():
                raise RuntimeError("CUDA is not available")
            
            high = np.array(df['high'].values, dtype=np.float64)
            low = np.array(df['low'].values, dtype=np.float64)
            close = np.array(df['close'].values, dtype=np.float64)
            
            # Prepare data for GPU
            high_gpu = cuda.to_device(high)
            low_gpu = cuda.to_device(low)
            close_gpu = cuda.to_device(close)
            atr_gpu = cuda.device_array_like(high)
            
            # Launch kernel
            blockspergrid = (high.size + (self.threadsperblock - 1)) // self.threadsperblock
            calculate_atr_kernel[blockspergrid, self.threadsperblock](high_gpu, low_gpu, close_gpu, atr_period, atr_gpu)
            
            # Get results back
            atr = atr_gpu.copy_to_host()
            
            # Clean up GPU memory
            del high_gpu, low_gpu, close_gpu, atr_gpu
            cuda.current_context().deallocations.clear()
            
            return atr
            
        except Exception as e:
            print(f"Error in ATR calculation: {str(e)}")
            traceback.print_exc()
            return None
    
    def calculate_supertrend(self, df, atr_period, factor, atr_buffer_multiplier, hardstop_distance):
        """Calculate SuperTrend with dynamic buffer zones using GPU acceleration"""
        try:
            # Check for cuda availability
            if not cuda.is_available():
                raise RuntimeError("CUDA is not available")
            
            # Calculate ATR first
            atr_values = self.calculate_atr(df, atr_period)
            if atr_values is None:
                return None
            
            high = np.array(df['high'].values, dtype=np.float64)
            low = np.array(df['low'].values, dtype=np.float64)
            close = np.array(df['close'].values, dtype=np.float64)
            
            # Create output arrays
            direction = np.zeros_like(high)
            supertrend = np.zeros_like(high)
            up_trend_buffer = np.zeros_like(high)
            down_trend_buffer = np.zeros_like(high)
            up_trend_hardstop = np.zeros_like(high)
            down_trend_hardstop = np.zeros_like(high)
            
            # Prepare data for GPU
            high_gpu = cuda.to_device(high)
            low_gpu = cuda.to_device(low)
            close_gpu = cuda.to_device(close)
            atr_gpu = cuda.to_device(atr_values)
            direction_gpu = cuda.to_device(direction)
            supertrend_gpu = cuda.to_device(supertrend)
            up_trend_buffer_gpu = cuda.to_device(up_trend_buffer)
            down_trend_buffer_gpu = cuda.to_device(down_trend_buffer)
            up_trend_hardstop_gpu = cuda.to_device(up_trend_hardstop)
            down_trend_hardstop_gpu = cuda.to_device(down_trend_hardstop)
            
            # Launch kernel
            blockspergrid = (high.size + (self.threadsperblock - 1)) // self.threadsperblock
            calculate_supertrend_kernel[blockspergrid, self.threadsperblock](
                high_gpu, low_gpu, close_gpu, atr_gpu, factor, atr_buffer_multiplier, hardstop_distance,
                direction_gpu, supertrend_gpu, up_trend_buffer_gpu, down_trend_buffer_gpu, 
                up_trend_hardstop_gpu, down_trend_hardstop_gpu
            )
            
            # Get results back
            direction_values = direction_gpu.copy_to_host()
            supertrend_values = supertrend_gpu.copy_to_host()
            up_trend_buffer_values = up_trend_buffer_gpu.copy_to_host()
            down_trend_buffer_values = down_trend_buffer_gpu.copy_to_host()
            up_trend_hardstop_values = up_trend_hardstop_gpu.copy_to_host()
            down_trend_hardstop_values = down_trend_hardstop_gpu.copy_to_host()
            
            # Clean up GPU memory
            del high_gpu, low_gpu, close_gpu, atr_gpu
            del direction_gpu, supertrend_gpu, up_trend_buffer_gpu, down_trend_buffer_gpu
            del up_trend_hardstop_gpu, down_trend_hardstop_gpu
            cuda.current_context().deallocations.clear()
            
            return {
                'atr': atr_values,
                'direction': direction_values,
                'supertrend': supertrend_values,
                'up_trend_buffer': up_trend_buffer_values,
                'down_trend_buffer': down_trend_buffer_values,
                'up_trend_hardstop': up_trend_hardstop_values,
                'down_trend_hardstop': down_trend_hardstop_values
            }
            
        except Exception as e:
            print(f"Error in SuperTrend calculation: {str(e)}")
            traceback.print_exc()
            return None

# CPU fallback implementation for systems without CUDA
class CPUSuperTrend:
    """Fallback implementation of SuperTrend calculation using CPU"""
    
    @staticmethod
    def calculate_atr(df, atr_period):
        """Calculate ATR using pandas"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = np.zeros_like(close)
        atr[atr_period] = np.mean(tr[:atr_period])
        for i in range(atr_period + 1, len(close)):
            atr[i] = ((atr[i-1] * (atr_period - 1)) + tr[i-1]) / atr_period
        return atr
    
    @staticmethod
    def calculate_supertrend(df, atr_period, factor, atr_buffer_multiplier, hardstop_distance):
        """Calculate SuperTrend with buffer zones using CPU"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        atr_values = CPUSuperTrend.calculate_atr(df, atr_period)
        
        direction = np.zeros_like(high)
        supertrend = np.zeros_like(high)
        supertrend[0] = close[0]
        up_trend_buffer = np.zeros_like(high)
        down_trend_buffer = np.zeros_like(high)
        up_trend_hardstop = np.zeros_like(high)
        down_trend_hardstop = np.zeros_like(high)
        
        for i in range(1, len(high)):
            # Basic bands
            factor_atr = atr_values[i] * factor
            upper_band = (high[i] + low[i]) / 2 + factor_atr
            lower_band = (high[i] + low[i]) / 2 - factor_atr
            
            # Dynamic buffer using ATR
            dynamic_buffer = atr_values[i] * atr_buffer_multiplier
            
            # Default to previous trend
            if close[i-1] > supertrend[i-1]:
                supertrend[i] = max(lower_band, supertrend[i-1])
                direction[i] = -1  # Up trend
            else:
                supertrend[i] = min(upper_band, supertrend[i-1])
                direction[i] = 1  # Down trend
            
            # Check for trend reversals
            if direction[i] == -1 and close[i] < supertrend[i]:
                direction[i] = 1
            elif direction[i] == 1 and close[i] > supertrend[i]:
                direction[i] = -1
            
            # Calculate buffer and hardstop lines
            if direction[i] == -1:  # Up trend
                up_trend_buffer[i] = supertrend[i] + dynamic_buffer
                up_trend_hardstop[i] = supertrend[i] - hardstop_distance
            else:  # Down trend
                down_trend_buffer[i] = supertrend[i] - dynamic_buffer
                down_trend_hardstop[i] = supertrend[i] + hardstop_distance
        
        return {
            'atr': atr_values,
            'direction': direction,
            'supertrend': supertrend,
            'up_trend_buffer': up_trend_buffer,
            'down_trend_buffer': down_trend_buffer,
            'up_trend_hardstop': up_trend_hardstop,
            'down_trend_hardstop': down_trend_hardstop
        }

# Factory to get appropriate implementation based on system capabilities
def get_supertrend_implementation():
    """Returns appropriate SuperTrend implementation based on system capabilities"""
    try:
        if cuda.is_available():
            print("CUDA is available. Using GPU acceleration for SuperTrend calculations.")
            return GPUAcceleratedSuperTrend()
        else:
            print("CUDA is not available. Using CPU fallback for SuperTrend calculations.")
            return CPUSuperTrend()
    except:
        print("Error checking CUDA availability. Using CPU fallback for SuperTrend calculations.")
        return CPUSuperTrend()
		
		
# Part 4: Backtesting Engine and Strategy Implementation

class Position:
    """Represents a trading position"""
    def __init__(self, entry_price, entry_time, direction, entry_index):
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.direction = direction  # 1 for long, -1 for short
        self.entry_index = entry_index
        self.exit_price = None
        self.exit_time = None
        self.exit_index = None
        self.exit_reason = None
        self.pnl = None
        self.pnl_percent = None
        
        # Extract time components for time-based exit checking
        self.entry_day = entry_time.day
        self.entry_month = entry_time.month
        self.entry_year = entry_time.year
        self.entry_hour = entry_time.hour
        self.entry_minute = entry_time.minute
        
    def close_position(self, exit_price, exit_time, exit_reason, exit_index):
        """Close the position and calculate P&L"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_index = exit_index
        self.exit_reason = exit_reason
        
        # Calculate P&L
        if self.direction == 1:  # Long
            self.pnl = self.exit_price - self.entry_price
        else:  # Short
            self.pnl = self.entry_price - self.exit_price
            
        self.pnl_percent = (self.pnl / self.entry_price) * 100
        return self.pnl
    
    def get_duration(self):
        """Calculate position duration in hours"""
        if self.exit_time is None:
            return 0
        duration = self.exit_time - self.entry_time
        return duration.total_seconds() / 3600  # Convert to hours

    def is_time_exit_due(self, current_time, current_day_diff):
        """Check if time-based exit criteria is met (5:29 PM IST / 11:59 AM UTC)"""
        # If trade is at least 1 day old and it's 11:59 AM UTC (5:29 PM IST)
        return current_day_diff >= 1 and current_time.hour == 11 and current_time.minute == 59

class BacktestResult:
    """Stores and analyzes backtest results"""
    def __init__(self, parameters, trades, df, initial_capital=10000.0):
        self.parameters = parameters
        self.trades = trades
        self.df = df
        self.initial_capital = initial_capital
        self.metrics = self.calculate_metrics()
        self.equity_curve = self.calculate_equity_curve()
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'net_profit': 0,
                'net_profit_percent': 0,
                'max_drawdown': 0,
                'max_drawdown_percent': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'avg_trade_duration': 0,
                'expectancy': 0,
                'total_trades': 0
            }
        
        # Extract PnL data
        pnls = [trade.pnl for trade in self.trades]
        pnl_percentages = [trade.pnl_percent for trade in self.trades]
        durations = [trade.get_duration() for trade in self.trades]
        
        # Calculate basic metrics
        net_pnl = sum(pnls)
        net_pnl_percent = sum(pnl_percentages)
        total_trades = len(self.trades)
        winning_trades = sum(1 for pnl in pnls if pnl > 0)
        losing_trades = sum(1 for pnl in pnls if pnl < 0)
        
        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate average trade duration
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Calculate Sharpe ratio (annualized)
        returns = np.array(pnl_percentages)
        sharpe_ratio = 0
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            if std_return > 0:
                # Assuming 252 trading days per year and daily returns
                sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
        
        # Calculate expectancy
        avg_win = np.mean([pnl for pnl in pnls if pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([pnl for pnl in pnls if pnl < 0]) if losing_trades > 0 else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss)) if total_trades > 0 else 0
        
        # Calculate max drawdown
        equity = self.calculate_equity_curve()
        peak = self.initial_capital
        max_dd = 0
        max_dd_percent = 0
        
        for value in equity:
            if value > peak:
                peak = value
            dd = peak - value
            dd_percent = (dd / peak) * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
            max_dd_percent = max(max_dd_percent, dd_percent)
                
        return {
            'net_profit': net_pnl,
            'net_profit_percent': net_pnl_percent,
            'max_drawdown': max_dd,
            'max_drawdown_percent': max_dd_percent,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate * 100,  # Convert to percentage
            'avg_trade_duration': avg_duration,
            'expectancy': expectancy,
            'total_trades': total_trades
        }
    
    def calculate_equity_curve(self):
        """Calculate equity curve over time"""
        equity = [self.initial_capital]
        current_equity = self.initial_capital
        
        for trade in self.trades:
            current_equity += trade.pnl
            equity.append(current_equity)
            
        return equity
    
    def is_valid_result(self, min_profit_percent=10, max_drawdown_percent=30):
        """Check if result meets minimum criteria"""
        return (self.metrics['net_profit_percent'] >= min_profit_percent and 
                self.metrics['max_drawdown_percent'] <= max_drawdown_percent and
                self.metrics['total_trades'] > 0)

    def get_parameter_str(self):
        """Format parameters as a readable string"""
        return (f"ATR Length: {self.parameters['atr_period']}, "
                f"Factor: {self.parameters['factor']}, "
                f"ATR Buffer Mult: {self.parameters['atr_buffer_multiplier']}, "
                f"Hard Stop: {self.parameters['hardstop_distance']}")

    def get_metrics_str(self):
        """Format metrics as a readable string"""
        m = self.metrics
        return (f"Net Profit: {m['net_profit']:.2f} ({m['net_profit_percent']:.2f}%), "
                f"Max DD: {m['max_drawdown']:.2f} ({m['max_drawdown_percent']:.2f}%), "
                f"PF: {m['profit_factor']:.2f}, "
                f"Sharpe: {m['sharpe_ratio']:.2f}, "
                f"Win Rate: {m['win_rate']:.1f}%, "
                f"Trades: {m['total_trades']}")

class SuperTrendATRBufferStrategy:
    """Implements the SuperTrend strategy with ATR-based dynamic buffer zones"""
    
    def __init__(self, df, atr_period, factor, atr_buffer_multiplier, hardstop_distance, enable_time_exit=True):
        self.df = df.copy()
        self.atr_period = atr_period
        self.factor = factor
        self.atr_buffer_multiplier = atr_buffer_multiplier
        self.hardstop_distance = hardstop_distance
        self.enable_time_exit = enable_time_exit
        self.positions = []
        self.current_position = None
        self.supertrend_impl = get_supertrend_implementation()
        
    def calculate_indicators(self):
        """Calculate SuperTrend and related indicators"""
        result = self.supertrend_impl.calculate_supertrend(
            self.df, 
            self.atr_period, 
            self.factor, 
            self.atr_buffer_multiplier, 
            self.hardstop_distance
        )
        
        if result is None:
            raise ValueError("Failed to calculate SuperTrend indicators")
        
        # Add indicators to dataframe
        self.df['atr'] = result['atr']
        self.df['direction'] = result['direction']
        self.df['supertrend'] = result['supertrend']
        self.df['up_trend_buffer'] = result['up_trend_buffer']
        self.df['down_trend_buffer'] = result['down_trend_buffer']
        self.df['up_trend_hardstop'] = result['up_trend_hardstop']
        self.df['down_trend_hardstop'] = result['down_trend_hardstop']
        
        # Add entry and exit signals for later analysis
        self.df['long_entry'] = False
        self.df['short_entry'] = False
        self.df['long_exit'] = False
        self.df['short_exit'] = False
        
    def run_backtest(self):
        """Run the backtest with the given parameters"""
        try:
            # Calculate indicators
            self.calculate_indicators()
            
            # Iterate through data
            for i in range(1, len(self.df)):
                # Get current and previous candle data
                prev_idx = self.df.index[i-1]
                curr_idx = self.df.index[i]
                
                prev_close = self.df.loc[prev_idx, 'close']
                curr_open = self.df.loc[curr_idx, 'open']
                curr_high = self.df.loc[curr_idx, 'high']
                curr_low = self.df.loc[curr_idx, 'low']
                curr_close = self.df.loc[curr_idx, 'close']
                
                prev_direction = self.df.loc[prev_idx, 'direction']
                curr_direction = self.df.loc[curr_idx, 'direction']
                
                curr_supertrend = self.df.loc[curr_idx, 'supertrend']
                curr_up_buffer = self.df.loc[curr_idx, 'up_trend_buffer']
                curr_down_buffer = self.df.loc[curr_idx, 'down_trend_buffer']
                curr_up_hardstop = self.df.loc[curr_idx, 'up_trend_hardstop']
                curr_down_hardstop = self.df.loc[curr_idx, 'down_trend_hardstop']
                
                # Current time components
                curr_time = curr_idx
                curr_day = self.df.loc[curr_idx, 'day']
                curr_month = self.df.loc[curr_idx, 'month']
                curr_year = self.df.loc[curr_idx, 'year']
                curr_hour = self.df.loc[curr_idx, 'hour']
                curr_minute = self.df.loc[curr_idx, 'minute']
                is_exit_time = self.df.loc[curr_idx, 'is_exit_time']
                
                # Process exits first
                if self.current_position is not None:
                    position = self.current_position
                    
                    # Calculate days difference for time-based exit
                    day_diff = 0
                    if (curr_year == position.entry_year and 
                        curr_month == position.entry_month):
                        day_diff = curr_day - position.entry_day
                    else:
                        day_diff = 1  # If month/year changes, assume next day
                    
                    # Check for trend change exit
                    trend_changed = (prev_direction != curr_direction)
                    
                    # Check for hard stop exit
                    hard_stop_hit = False
                    if position.direction == 1:  # Long position
                        hard_stop_hit = curr_low <= curr_up_hardstop and curr_up_hardstop > 0
                    else:  # Short position
                        hard_stop_hit = curr_high >= curr_down_hardstop and curr_down_hardstop > 0
                    
                    # Check for time-based exit
                    time_exit_due = self.enable_time_exit and day_diff >= 1 and is_exit_time
                    
                    # Handle exits
                    if position.direction == 1:  # Long position
                        if trend_changed:
                            self.df.loc[curr_idx, 'long_exit'] = True
                            position.close_position(curr_close, curr_time, "Trend Change", i)
                            self.current_position = None
                        elif hard_stop_hit:
                            self.df.loc[curr_idx, 'long_exit'] = True
                            position.close_position(curr_up_hardstop, curr_time, "Hard Stop", i)
                            self.current_position = None
                        elif time_exit_due:
                            self.df.loc[curr_idx, 'long_exit'] = True
                            position.close_position(curr_close, curr_time, "Time Exit", i)
                            self.current_position = None
                    
                    else:  # Short position
                        if trend_changed:
                            self.df.loc[curr_idx, 'short_exit'] = True
                            position.close_position(curr_close, curr_time, "Trend Change", i)
                            self.current_position = None
                        elif hard_stop_hit:
                            self.df.loc[curr_idx, 'short_exit'] = True
                            position.close_position(curr_down_hardstop, curr_time, "Hard Stop", i)
                            self.current_position = None
                        elif time_exit_due:
                            self.df.loc[curr_idx, 'short_exit'] = True
                            position.close_position(curr_close, curr_time, "Time Exit", i)
                            self.current_position = None
                
                # Then process entries
                if self.current_position is None:
                    # Long entry condition - price between supertrend and buffer zone, in uptrend
                    long_condition = (
                        curr_direction < 0 and  # Direction check (uptrend)
                        curr_close <= curr_up_buffer and  # Price below buffer 
                        curr_close >= curr_supertrend and  # Price above supertrend
                        curr_up_buffer > 0  # Valid buffer value
                    )
                    
                    # Short entry condition - price between supertrend and buffer zone, in downtrend
                    short_condition = (
                        curr_direction > 0 and  # Direction check (downtrend) 
                        curr_close >= curr_down_buffer and  # Price above buffer
                        curr_close <= curr_supertrend and  # Price below supertrend
                        curr_down_buffer > 0  # Valid buffer value
                    )
                    
                    if long_condition:
                        self.df.loc[curr_idx, 'long_entry'] = True
                        self.current_position = Position(curr_close, curr_time, 1, i)  # Long
                    
                    elif short_condition:
                        self.df.loc[curr_idx, 'short_entry'] = True
                        self.current_position = Position(curr_close, curr_time, -1, i)  # Short
            
            # Close any open position at the end of the test
            if self.current_position is not None:
                last_idx = self.df.index[-1]
                last_close = self.df.loc[last_idx, 'close']
                self.current_position.close_position(last_close, last_idx, "End of Test", len(self.df) - 1)
            
            # Collect all completed trades
            self.positions = [pos for pos in self.positions if pos.exit_price is not None]
            if self.current_position is not None and self.current_position.exit_price is not None:
                self.positions.append(self.current_position)
            
            # Create and return result object
            return BacktestResult(
                parameters={
                    'atr_period': self.atr_period,
                    'factor': self.factor,
                    'atr_buffer_multiplier': self.atr_buffer_multiplier,
                    'hardstop_distance': self.hardstop_distance,
                    'enable_time_exit': self.enable_time_exit
                },
                trades=self.positions,
                df=self.df
            )
            
        except Exception as e:
            print(f"Error during backtest: {str(e)}")
            traceback.print_exc()
            return None
			
			
# Part 5: Optimization Engine and Results Visualization

class StrategyOptimizer:
    """Handles parameter optimization for the SuperTrend dynamic buffer strategy"""
    
    def __init__(self, df, parameters, directory_manager, memory_manager):
        self.df = df
        self.parameters = parameters
        self.directory_manager = directory_manager
        self.memory_manager = memory_manager
        self.results = []
        self.elite_results = []
        self.min_profit_percent = 10  # Minimum profit percentage to consider a result "good"
        self.max_drawdown_percent = 30  # Maximum allowable drawdown percentage
    
    def generate_parameter_combinations(self):
        """Generate all combinations of parameters to test"""
        param_grid = list(product(
            self.parameters['atr_periods'],
            self.parameters['factors'],
            self.parameters['buffer_multipliers'],
            self.parameters['hardstop_distances']
        ))
        
        total_combinations = len(param_grid)
        print(f"Generated {total_combinations:,} parameter combinations to test")
        return param_grid
    
    def optimize_single(self, params):
        """Run a single backtest with given parameters"""
        atr_period, factor, buffer_multiplier, hardstop_distance = params
        
        # Add real-time progress reporting
        print(f"Testing: ATR={atr_period}, Factor={factor:.2f}, Buffer={buffer_multiplier:.2f}, Stop={hardstop_distance:.1f}", end='\r')
        
        strategy = SuperTrendATRBufferStrategy(
            df=self.df,
            atr_period=atr_period,
            factor=factor,
            atr_buffer_multiplier=buffer_multiplier,
            hardstop_distance=hardstop_distance,
            enable_time_exit=self.parameters['enable_time_exit']
        )
        
        result = strategy.run_backtest()
        return result
    
    def optimize_batch(self, param_grid, batch_size=None):
        """Run optimization in batches to manage memory"""
        if batch_size is None:
            # Calculate optimal batch size based on memory constraints
            df_size = self.df.memory_usage(deep=True).sum()
            batch_size = self.memory_manager.calculate_optimal_batch_size(
                df_size, len(param_grid))
            
        # Make sure batch size is at least 1
        batch_size = max(batch_size, 1)
        
        num_batches = math.ceil(len(param_grid) / batch_size)
        all_valid_results = []
        
        print(f"Processing {len(param_grid):,} parameter combinations in {num_batches:,} batches")
        print(f"Batch size: {batch_size:,} combinations per batch")
        
        start_time = time.time()
        combinations_processed = 0
        
        with tqdm(total=len(param_grid), desc="Optimization Progress") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(param_grid))
                batch_params = param_grid[start_idx:end_idx]
                
                batch_start_time = time.time()
                print(f"\nProcessing batch {batch_idx+1}/{num_batches} ({len(batch_params)} combinations)")
                
                # Process batch with explicit progress updates
                valid_results_in_batch = 0
                
                # Use smaller chunks for better progress visibility
                chunk_size = 10  # Process in smaller chunks for more frequent updates
                for chunk_start in range(0, len(batch_params), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(batch_params))
                    chunk_params = batch_params[chunk_start:chunk_end]
                    
                    # Process chunk sequentially with progress updates
                    for param in chunk_params:
                        result = self.optimize_single(param)
                        combinations_processed += 1
                        
                        if result is not None and result.is_valid_result(
                            self.min_profit_percent, self.max_drawdown_percent):
                            all_valid_results.append(result)
                            valid_results_in_batch += 1
                    
                    # Update progress after each chunk
                    pbar.update(len(chunk_params))
                    elapsed = time.time() - start_time
                    combos_per_sec = combinations_processed / elapsed if elapsed > 0 else 0
                    est_remaining = (len(param_grid) - combinations_processed) / combos_per_sec if combos_per_sec > 0 else float('inf')
                    
                    print(f"Progress: {combinations_processed}/{len(param_grid)} combinations "
                        f"({(combinations_processed/len(param_grid)*100):.1f}%) | "
                        f"Valid results: {len(all_valid_results)} | "
                        f"Speed: {combos_per_sec:.2f} combos/sec | "
                        f"Est. time remaining: {est_remaining/60:.1f} minutes")
                
                # Batch summary
                batch_time = time.time() - batch_start_time
                print(f"Batch {batch_idx+1} completed in {batch_time:.1f}s with {valid_results_in_batch} valid results")
                
                # Cleanup between batches
                cleanup_gpu_memory()
                
        total_time = time.time() - start_time
        print(f"\nOptimization completed in {total_time/60:.1f} minutes")
        print(f"Total valid results: {len(all_valid_results)} out of {len(param_grid)} combinations")
        
        self.results = all_valid_results
        return all_valid_results
    
    def optimize_batch_parallel(self, param_grid, batch_size=None):
        """Run optimization in batches using parallel processing (original method)"""
        if batch_size is None:
            # Calculate optimal batch size based on memory constraints
            df_size = self.df.memory_usage(deep=True).sum()
            batch_size = self.memory_manager.calculate_optimal_batch_size(
                df_size, len(param_grid))
            
        # Make sure batch size is at least 1
        batch_size = max(batch_size, 1)
        
        num_batches = math.ceil(len(param_grid) / batch_size)
        all_valid_results = []
        
        print(f"Processing {len(param_grid):,} parameter combinations in {num_batches:,} batches")
        print(f"Batch size: {batch_size:,} combinations per batch")
        
        with tqdm(total=len(param_grid), desc="Optimization Progress") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(param_grid))
                batch_params = param_grid[start_idx:end_idx]
                
                # Process batch in parallel
                with ProcessPoolExecutor() as executor:
                    batch_results = list(executor.map(self.optimize_single, batch_params))
                
                # Filter valid results
                valid_results = [r for r in batch_results if r is not None and r.is_valid_result(
                    self.min_profit_percent, self.max_drawdown_percent)]
                all_valid_results.extend(valid_results)
                
                # Update progress
                pbar.update(len(batch_params))
                pbar.set_postfix(valid=len(all_valid_results))
                
                # Cleanup between batches
                cleanup_gpu_memory()
                
        self.results = all_valid_results
        return all_valid_results

def main():
    """Main entry point for the application"""
    print("\n" + "=" * 60)
    print(" SUPERTREND DYNAMIC ATR BUFFER STRATEGY OPTIMIZER ".center(60, "="))
    print("=" * 60)
    print(f"Version: 1.0")
    print(f"Date: 2025-06-08")
    print(f"User: {CURRENT_USER}")
    print(f"Current UTC Time: {CURRENT_UTC}")
    print("=" * 60 + "\n")
    
    try:
        # Initialize managers
        dir_manager = DirectoryManager()
        mem_manager = MemoryManager()
        
        # Find data files
        data_dir = "."  # Current directory
        data_files = find_ohlc_files(data_dir)
        
        if not data_files:
            print("No data files found. Please place OHLC data files in the current directory.")
            return
        
        # Let user select files
        selected_files = select_files_to_process(data_files)
        
        if not selected_files:
            print("No files selected. Exiting.")
            return
            
        # Get parameter inputs
        params = get_parameter_inputs()
        
        # Ask user if they want to use batch processing or single-threaded mode
        print("\nProcessing Options:")
        print("1. Full batch processing (faster but less visible progress)")
        print("2. Sequential processing with detailed progress updates (slower but shows more updates)")
        choice = input("Select option (1/2): ").strip()
        
        use_sequential = choice == "2"
        
        # Process each selected file
        for file_path in selected_files:
            print(f"\nProcessing file: {os.path.basename(file_path)}")
            
            # Load data
            df = load_ohlc_data(file_path)
            
            # Consider using a subset of data for faster testing
            use_subset = input("\nWould you like to use a subset of data for faster testing? (y/n): ").lower().strip() == 'y'
            
            if use_subset:
                subset_size = get_int_input("Enter number of days for subset (e.g., 30): ", 1)
                # Get most recent subset_size days
                last_date = df.index.max()
                start_date = last_date - pd.Timedelta(days=subset_size)
                df = df[df.index >= start_date]
                print(f"Using data subset from {start_date} to {last_date} ({len(df)} rows)")
            
            # Create optimizer
            optimizer = StrategyOptimizer(df, params, dir_manager, mem_manager)
            
            # Generate parameter combinations
            param_grid = optimizer.generate_parameter_combinations()
            
            # Ask if user wants to proceed with full parameter grid
            proceed = input(f"\nProceed with testing {len(param_grid):,} parameter combinations? (y/n): ").lower().strip() == 'y'
            
            if not proceed:
                sample_size = get_int_input("Enter number of random combinations to test: ", 1, len(param_grid))
                import random
                random.seed(42)  # For reproducibility
                param_grid = random.sample(param_grid, sample_size)
                print(f"Selected {len(param_grid)} random parameter combinations")
            
            try:
                # Run optimization
                if use_sequential:
                    optimizer.optimize_batch(param_grid)
                else:
                    # Run with default parallel processing
                    optimizer.optimize_batch_parallel(param_grid)
                
            except KeyboardInterrupt:
                print("\nProcess interrupted by user. Saving current results...")
            
            # Rank results (even if interrupted)
            if optimizer.results:
                optimizer.rank_results(top_n=5)
                
                # Print elite results
                optimizer.print_elite_results()
                
                # Save results and visualizations
                optimizer.save_results_to_csv(include_all=True)
                if optimizer.elite_results:
                    optimizer.save_trades_to_csv(result_idx=0)
                    optimizer.plot_equity_curves()
                    optimizer.plot_drawdown_curves()
                    optimizer.plot_monthly_returns(result_idx=0)
            else:
                print("No valid results found during optimization.")
            
            print(f"\nProcessing for {os.path.basename(file_path)} completed.")
            
        print("\nAll files processed successfully.")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()