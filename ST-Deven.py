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
CURRENT_UTC = "2025-05-03 19:46:10"
CURRENT_USER = "arullr001"
TARGET_MAX = 15.00  # Maximum target (1500 points)


# Updated timestamp
CURRENT_UTC = "2025-05-03 19:46:33"
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
CURRENT_UTC = "2025-05-03 19:46:59"
CURRENT_USER = "arullr001"

def get_parameter_inputs():
    """Get all parameter inputs from user with validation"""
    while True:
        print("\n" + "=" * 50)
        print(" PARAMETER RANGE CONFIGURATION ".center(50, "="))
        print("=" * 50)

        # Period range inputs
        print("\nEnter Period range (Integer values):")
        period_start = get_int_input("  Start value (min 5): ", 5)
        period_end = get_int_input(f"  End value (min {period_start}): ", period_start)
        period_step = get_int_input("  Step/increment (min 1): ", 1)

        # Multiplier range inputs
        print("\nEnter Multiplier range (Float values):")
        mult_start = get_float_input("  Start value (min 0.5): ", 0.5)
        mult_end = get_float_input(f"  End value (min {mult_start}): ", mult_start)
        mult_step = get_float_input("  Step/increment (min 0.01): ", 0.01)

        # Fixed Price Range values (50-300 points)
        print("\nPrice Range Configuration:")
        print("  Fixed range: 50 to 300 points (0.50 to 3.00)")
        print("  Increment: 10 points (0.10)")
        price_ranges = [round(x, 2) for x in np.arange(0.50, 3.01, 0.10)]

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

        # Generate the ranges
        periods = list(range(period_start, period_end + 1, period_step))
        multipliers = [round(x, 2) for x in np.arange(mult_start, mult_end + (mult_step / 2), mult_step)]

        # Calculate total combinations
        total_combinations = len(periods) * len(multipliers) * len(target_combinations)

        # Summary
        print("\n" + "=" * 50)
        print(" PARAMETER SUMMARY ".center(50, "="))
        print("=" * 50)
        print(f"Period range: {period_start} to {period_end} (step {period_step}) - {len(periods)} values")
        print(f"Multiplier range: {mult_start} to {mult_end} (step {mult_step}) - {len(multipliers)} values")
        print(f"Price ranges: 50 to 300 points (step 10 points) - {len(price_ranges)} values")
        print(f"Target configurations: {len(target_combinations)} combinations")
        print(f"\nTotal parameter combinations to test: {total_combinations}")

        # Memory estimation and validation
        mem = psutil.virtual_memory()
        estimated_memory_mb = total_combinations * 0.5
        print(f"Estimated memory required: ~{estimated_memory_mb:.1f} MB")
        print(f"Available memory: {mem.available / (1024**2):.1f} MB")

        if total_combinations > 100000:
            print("\nWARNING: Very high number of combinations may cause long processing time")

        confirm = input("\nProceed with these parameters? (y/n): ").lower().strip()
        if confirm == 'y':
            return {
                'periods': periods,
                'multipliers': multipliers,
                'price_ranges': price_ranges,
                'target_combinations': target_combinations,
                'total_combinations': total_combinations
            }
        print("\nLet's reconfigure the parameters...")

class DirectoryManager:
    """Manages directory structure for the application"""
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_dir = f"master_directory_{self.timestamp}"
        self.csv_dumps_dir = os.path.join(self.base_dir, "csv_dumps")
        self.error_logs_dir = os.path.join(self.base_dir, "error_logs")
        self.final_results_dir = os.path.join(self.base_dir, "final_results")
        self.create_directory_structure()
        self.setup_logging()

    def create_directory_structure(self):
        """Creates the required directory structure"""
        directories = [
            self.base_dir,
            self.csv_dumps_dir,
            self.error_logs_dir,
            self.final_results_dir
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
CURRENT_UTC = "2025-05-03 19:47:50"
CURRENT_USER = "arullr001"

def load_ohlc_data(file_path):
    """Load and prepare OHLC data from file"""
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
	
	
# Updated timestamp
CURRENT_UTC = "2025-05-03 19:48:38"
CURRENT_USER = "arullr001"

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

def process_parameter_combination(df, params):
    """Process a single parameter combination"""
    period, multiplier, price_range_1, long_target, short_target = params
    
    try:
        trend, trailing_up_30, trailing_dn_30 = calculate_supertrend_cuda(
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
                        'price_range': price_range_1
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
                        'price_range': price_range_1
                    })
                    
                    position = 0
        
        if trades:
            trades_df = pd.DataFrame(trades)
            metrics = calculate_metrics(trades_df)
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
    """Calculate trading metrics from trades"""
    if trades_df.empty:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'max_drawdown': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0
        }
    
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['profit_pct'] > 0])
    win_rate = (winning_trades / total_trades) * 100
    
    avg_profit = trades_df['profit_pct'].mean()
    
    # Calculate drawdown
    cumulative_returns = (1 + trades_df['profit_pct']/100).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max * 100
    max_drawdown = abs(drawdowns.min())
    
    # Profit factor
    gross_profit = trades_df[trades_df['profit_pct'] > 0]['profit_pct'].sum()
    gross_loss = abs(trades_df[trades_df['profit_pct'] < 0]['profit_pct'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    # Sharpe ratio (simplified)
    returns = trades_df['profit_pct']
    sharpe_ratio = (returns.mean() / returns.std()) * math.sqrt(252) if returns.std() != 0 else 0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio
    }
	
	
	
# Updated timestamp
CURRENT_UTC = "2025-05-03 19:49:20"
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

def run_backtest(df, parameter_inputs, dir_manager):
    """Run the backtest with the given parameters"""
    periods = parameter_inputs['periods']
    multipliers = parameter_inputs['multipliers']
    target_combinations = parameter_inputs['target_combinations']
    
    # Create all parameter combinations
    parameter_combinations = [
        (period, mult, pr, long_target, short_target)
        for period in periods
        for mult in multipliers
        for pr, long_target, short_target in target_combinations
    ]
    
    memory_manager = MemoryManager()
    batch_size = memory_manager.calculate_optimal_batch_size(
        df.memory_usage().sum(),
        len(parameter_combinations)
    )
    
    all_results = []
    total_batches = math.ceil(len(parameter_combinations) / batch_size)
    
    with tqdm(total=len(parameter_combinations), desc="Processing combinations") as pbar:
        for batch_start in range(0, len(parameter_combinations), batch_size):
            batch_results = process_batch(
                df, parameter_combinations, batch_start, batch_size
            )
            all_results.extend(batch_results)
            pbar.update(min(batch_size, len(parameter_combinations) - batch_start))
    
    return compile_results(all_results, dir_manager)

def compile_results(results, dir_manager):
    """Compile and save results"""
    if not results:
        print("No valid results to compile")
        return None
    
    # Convert results to DataFrame
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'trades'}
        for r in results
    ])
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    detailed_results_path = os.path.join(
        dir_manager.csv_dumps_dir,
        f'detailed_results_{timestamp}.csv'
    )
    results_df.to_csv(detailed_results_path)
    
    # Filter and sort results
    filtered_results = results_df[
        (results_df['win_rate'] > 0) &
        (results_df['total_trades'] >= 10)
    ].sort_values(
        by=['sharpe_ratio', 'profit_factor', 'win_rate'],
        ascending=[False, False, False]
    )
    
    # Get top 5 combinations
    top_results = filtered_results.head(5)
    
    # Save top results
    top_results_path = os.path.join(
        dir_manager.final_results_dir,
        f'top_results_{timestamp}.csv'
    )
    top_results.to_csv(top_results_path)
    
    # Display top results
    print("\nTop 5 Parameter Combinations:")
    print("=" * 100)
    for idx, row in top_results.iterrows():
        print(f"\nRank {idx + 1}:")
        print(f"Period: {row['period']}")
        print(f"Multiplier: {row['multiplier']}")
        print(f"Price Range: {row['price_range_1']}")
        print(f"Long Target: {row['long_target']}")
        print(f"Short Target: {row['short_target']}")
        print(f"Total Trades: {row['total_trades']}")
        print(f"Win Rate: {row['win_rate']:.2f}%")
        print(f"Average Profit: {row['avg_profit']:.2f}%")
        print(f"Max Drawdown: {row['max_drawdown']:.2f}%")
        print(f"Profit Factor: {row['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {row['sharpe_ratio']:.2f}")
        print("-" * 50)
    
    return top_results

def main():
    """Main execution function"""
    try:
        print("\nSuperTrend Backtesting System")
        print("=" * 50)
        
        # Initialize directory manager
        dir_manager = DirectoryManager()
        
        # Find and select data files
        data_files = find_ohlc_files(".")
        selected_files = select_files_to_process(data_files)
        
        if not selected_files:
            print("No files selected for processing")
            return
        
        # Get parameter inputs
        parameter_inputs = get_parameter_inputs()
        
        # Process each selected file
        for file_path in selected_files:
            print(f"\nProcessing file: {file_path}")
            try:
                # Load data
                df = load_ohlc_data(file_path)
                
                # Run backtest
                results = run_backtest(df, parameter_inputs, dir_manager)
                
                if results is not None:
                    print(f"\nResults saved in {dir_manager.final_results_dir}")
                else:
                    print("\nNo valid results found for this file")
                
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                logging.getLogger('system_errors').error(
                    f"Error processing file {file_path}:\n{traceback.format_exc()}"
                )
    
    except Exception as e:
        print(f"System error: {str(e)}")
        logging.getLogger('system_errors').error(
            f"System error:\n{traceback.format_exc()}"
        )

if __name__ == "__main__":
    main()
