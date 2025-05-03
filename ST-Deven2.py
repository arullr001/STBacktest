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

# New optimization imports
from scipy.optimize import differential_evolution, basinhopping
from skopt import gp_minimize, space
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define all constants first
THREADS_PER_BLOCK = 256
MAX_RAM_USAGE_PERCENT = 90
BATCH_SIZE = 1000
CURRENT_UTC = "2025-05-03 21:11:19"
CURRENT_USER = "arullr001"
TARGET_MAX = 15.00  # Maximum target (1500 points)

# Parameter Space Constants
PERIOD_MIN = 10
PERIOD_MAX = 100
MULTIPLIER_MIN = 5
MULTIPLIER_MAX = 30

# Directory Structure Constants
LOG_DIR = "logs"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
CACHE_DIR = "cache"

# Optimization Constants
N_CALLS = 100  # Number of optimization steps
N_RANDOM_STARTS = 20  # Number of random starts for optimization

# Define OptimizationSpace class first
class OptimizationSpace:
    """Defines the parameter space for optimization"""
    def __init__(self):
        self.dimensions = [
            Integer(PERIOD_MIN, PERIOD_MAX, name='period'),
            Real(MULTIPLIER_MIN, MULTIPLIER_MAX, name='multiplier'),
            Real(0.50, 3.01, name='price_range')
        ]
        
    def get_space(self):
        return self.dimensions

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
        
        
def find_ohlc_files(data_dir):
    """Find all OHLC data files in the given directory"""
    files = []
    supported_extensions = ['.csv', '.xlsx', '.xls']
    try:
        for ext in supported_extensions:
            files.extend(glob.glob(os.path.join(data_dir, f'*{ext}')))
        return sorted(files)
    except Exception as e:
        logging.getLogger('system_errors').error(f"Error finding OHLC files: {str(e)}")
        return []

def select_files_to_process(data_files):
    """Let user select which files to process"""
    if not data_files:
        print("No data files found")
        return []

    print(f"\nFound {len(data_files)} data files for testing")
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
        
        


class DirectoryManager:
    """Manages directory structure for the application"""
    def __init__(self):
        current_file = os.path.basename(__file__)
        file_name = os.path.splitext(current_file)[0]
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_dir = f"{file_name}_{self.timestamp}"
        self.csv_dumps_dir = os.path.join(self.base_dir, "csv_dumps")
        self.error_logs_dir = os.path.join(self.base_dir, "error_logs")
        self.final_results_dir = os.path.join(self.base_dir, "final_results")
        self.plots_dir = os.path.join(self.base_dir, "plots")
        self.cache_dir = os.path.join(self.base_dir, "cache")
        self.optimization_dir = os.path.join(self.base_dir, "optimization")  # New directory for optimization results
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
            self.cache_dir,
            self.optimization_dir
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

        # Optimization logger
        optimization_logger = logging.getLogger('optimization')
        optimization_logger.setLevel(logging.INFO)
        fh_optimization = logging.FileHandler(
            os.path.join(self.error_logs_dir, 'optimization.log')
        )
        fh_optimization.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        optimization_logger.addHandler(fh_optimization)


class MemoryManager:
    """Manages memory usage and monitoring"""
    def __init__(self, max_ram_percent=MAX_RAM_USAGE_PERCENT):
        self.max_ram_percent = max_ram_percent
        self.total_ram = psutil.virtual_memory().total
        self.max_ram_bytes = (self.total_ram * self.max_ram_percent) / 100
        self.optimization_memory = {}  # Cache for optimization results

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

    def cache_optimization_result(self, params, result):
        """Cache optimization result for reuse"""
        param_key = tuple(params)
        self.optimization_memory[param_key] = result

    def get_cached_result(self, params):
        """Retrieve cached optimization result"""
        param_key = tuple(params)
        return self.optimization_memory.get(param_key)

    def clear_optimization_cache(self):
        """Clear optimization cache"""
        self.optimization_memory.clear()
		
		
# Update timestamp constants
CURRENT_UTC = "2025-05-03 21:00:12"
CURRENT_USER = "arullr001"

class OptimizationManager:
    """Manages different optimization strategies"""
    def __init__(self, df, memory_manager):
        self.df = df
        self.memory_manager = memory_manager
        self.opt_space = OptimizationSpace()
        self.logger = logging.getLogger('optimization')
        self.best_results = []

    @use_named_args(OptimizationSpace().get_space())
    def objective_function(self, period, multiplier, price_range):
        """Objective function for optimization"""
        try:
            # Check cache first
            cached_result = self.memory_manager.get_cached_result((period, multiplier, price_range))
            if cached_result is not None:
                return -cached_result['avg_profit']  # Negative because we're minimizing

            # Calculate SuperTrend
            params = (int(period), float(multiplier), float(price_range), 
                     price_range * 3, price_range * 3)  # Last two are targets
            result = process_parameter_combination(self.df, params)

            if result:
                self.memory_manager.cache_optimization_result(
                    (period, multiplier, price_range),
                    result
                )
                return -result['avg_profit']  # Negative because we're minimizing
            return float('inf')

        except Exception as e:
            self.logger.error(f"Error in objective function: {str(e)}")
            return float('inf')

    def bayesian_optimization(self):
        """Perform Bayesian optimization"""
        try:
            self.logger.info("Starting Bayesian optimization")
            result = gp_minimize(
                func=self.objective_function,
                dimensions=self.opt_space.get_space(),
                n_calls=N_CALLS,
                n_random_starts=N_RANDOM_STARTS,
                noise=0.1,
                verbose=True
            )
            self.best_results.append(('bayesian', result))
            return result

        except Exception as e:
            self.logger.error(f"Bayesian optimization failed: {str(e)}")
            return None

    def evolutionary_optimization(self):
        """Perform evolutionary optimization"""
        try:
            self.logger.info("Starting evolutionary optimization")
            bounds = [(PERIOD_MIN, PERIOD_MAX),
                     (MULTIPLIER_MIN, MULTIPLIER_MAX),
                     (0.5, 3.01)]
            
            result = differential_evolution(
                func=lambda x: self.objective_function(
                    period=int(x[0]),
                    multiplier=x[1],
                    price_range=x[2]
                ),
                bounds=bounds,
                popsize=15,
                maxiter=50,
                updating='deferred',
                workers=1
            )
            self.best_results.append(('evolutionary', result))
            return result

        except Exception as e:
            self.logger.error(f"Evolutionary optimization failed: {str(e)}")
            return None

    def basin_hopping_optimization(self):
        """Perform basin-hopping optimization"""
        try:
            self.logger.info("Starting basin-hopping optimization")
            x0 = [50, 15, 1.5]  # Initial guess (middle of ranges)
            
            result = basinhopping(
                func=lambda x: self.objective_function(
                    period=int(x[0]),
                    multiplier=x[1],
                    price_range=x[2]
                ),
                x0=x0,
                niter=100,
                T=1.0,
                stepsize=0.5
            )
            self.best_results.append(('basin_hopping', result))
            return result

        except Exception as e:
            self.logger.error(f"Basin-hopping optimization failed: {str(e)}")
            return None

    def run_all_optimizations(self):
        """Run all optimization strategies and return best result"""
        optimization_methods = [
            self.bayesian_optimization,
            self.evolutionary_optimization,
            self.basin_hopping_optimization
        ]

        best_result = None
        best_score = float('inf')

        for opt_method in optimization_methods:
            try:
                result = opt_method()
                if result and -result.fun < best_score:  # Converting back to profit
                    best_score = -result.fun
                    best_result = result
            except Exception as e:
                self.logger.error(f"Optimization method failed: {str(e)}")
                continue

        return best_result

    def get_best_parameters(self):
        """Get best parameters from all optimization runs"""
        if not self.best_results:
            return None

        best_params = None
        best_score = float('inf')

        for method, result in self.best_results:
            if result and -result.fun < best_score:
                best_score = -result.fun
                best_params = {
                    'method': method,
                    'period': int(result.x[0]),
                    'multiplier': float(result.x[1]),
                    'price_range': float(result.x[2]),
                    'score': -result.fun
                }

        return best_params

# Keep the existing SuperTrend CUDA implementation
@cuda.jit
def calculate_supertrend_cuda_kernel(high, low, close, period, multiplier, price_range_1, up, dn, trend, trailing_up_30, trailing_dn_30):
    """CUDA kernel for SuperTrend calculation"""
    # [Previous CUDA kernel implementation remains exactly the same]
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
	
	
# Update timestamp constants
CURRENT_UTC = "2025-05-03 21:01:40"
CURRENT_USER = "arullr001"

class BatchProcessor:
    """Enhanced batch processor with optimization support"""
    def __init__(self, directory_manager, memory_manager):
        self.dir_manager = directory_manager
        self.mem_manager = memory_manager
        self.current_batch = 0
        self.processing_logger = logging.getLogger('processing_errors')
        self.system_logger = logging.getLogger('system_errors')
        self.optimization_logger = logging.getLogger('optimization')
        self.best_params = None

    def process_optimized_batch(self, df, optimization_manager, batch_size, max_workers=4):
        """Process a batch using optimized parameters"""
        try:
            # Get optimized parameters if not already obtained
            if not self.best_params:
                self.optimization_logger.info("Starting optimization process")
                best_result = optimization_manager.run_all_optimizations()
                self.best_params = optimization_manager.get_best_parameters()
                
                if not self.best_params:
                    raise ValueError("Optimization failed to produce valid parameters")

            # Generate parameter combinations around the optimal point
            param_combinations = self.generate_parameter_combinations(self.best_params)
            
            results = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for params in param_combinations[:batch_size]:
                    futures.append(
                        executor.submit(process_parameter_combination, df.copy(), params)
                    )

                for future in tqdm(concurrent.futures.as_completed(futures), 
                                 total=len(futures), 
                                 desc=f"Processing optimized batch {self.current_batch + 1}"):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        self.processing_logger.error(
                            f"Error in optimized batch {self.current_batch + 1}: {str(e)}"
                        )

            return results

        except Exception as e:
            self.system_logger.error(f"Error in optimized batch processing: {str(e)}")
            return []

    def generate_parameter_combinations(self, best_params):
        """Generate parameter combinations around the optimal point"""
        period = best_params['period']
        multiplier = best_params['multiplier']
        price_range = best_params['price_range']

        # Generate variations around the best parameters
        periods = range(
            max(PERIOD_MIN, period - 5),
            min(PERIOD_MAX, period + 6)
        )
        multipliers = np.arange(
            max(MULTIPLIER_MIN, multiplier - 1.0),
            min(MULTIPLIER_MAX, multiplier + 1.1),
            0.1
        )
        price_ranges = np.arange(
            max(0.5, price_range - 0.2),
            min(3.01, price_range + 0.21),
            0.05
        )

        # Create combinations
        combinations = []
        for p in periods:
            for m in multipliers:
                for pr in price_ranges:
                    # Add targets based on price range
                    long_target = pr * 3
                    short_target = pr * 3
                    combinations.append((p, m, pr, long_target, short_target))

        return combinations

    def save_batch_results(self, results, batch_num, is_optimized=True):
        """Save batch results with optimization metadata"""
        try:
            batch_file = os.path.join(
                self.dir_manager.csv_dumps_dir, 
                f'batch_{batch_num}_{"optimized" if is_optimized else "regular"}_results.csv'
            )
            metadata_file = os.path.join(
                self.dir_manager.csv_dumps_dir, 
                f'batch_{batch_num}_metadata.json'
            )

            results_data = []
            for r in results:
                if r and isinstance(r, dict):
                    row = {
                        'period': r.get('period'),
                        'multiplier': r.get('multiplier'),
                        'price_range_1': r.get('price_range_1'),
                        'long_target': r.get('long_target'),
                        'short_target': r.get('short_target'),
                        'total_trades': r.get('total_trades', 0),
                        'win_rate': r.get('win_rate', 0),
                        'avg_profit': r.get('avg_profit', 0),
                        'max_drawdown': r.get('max_drawdown', 0),
                        'profit_factor': r.get('profit_factor', 0),
                        'sharpe_ratio': r.get('sharpe_ratio', 0),
                        'optimization_method': self.best_params.get('method') if self.best_params else 'none'
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
                    'optimization_used': is_optimized,
                    'best_parameters': self.best_params if self.best_params else None,
                    'batch_performance': {
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
        """Merge all batch results with optimization metadata"""
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
                
                self.save_final_results(final_df)
                return final_df
                
            return pd.DataFrame()

        except Exception as e:
            self.system_logger.error(f"Error merging batch results: {str(e)}")
            return pd.DataFrame()
			
			
# Update timestamp constants
CURRENT_UTC = "2025-05-03 21:02:38"
CURRENT_USER = "arullr001"

def get_parameter_inputs():
    """Enhanced parameter input function with optimization support"""
    print("\n" + "=" * 50)
    print(" PARAMETER CONFIGURATION ".center(50, "="))
    print("=" * 50)

    print("\nOptimization Parameter Ranges:")
    print(f"Period Range: {PERIOD_MIN} to {PERIOD_MAX}")
    print(f"Multiplier Range: {MULTIPLIER_MIN} to {MULTIPLIER_MAX}")
    print(f"Price Range: 0.50 to 3.00 (50 to 300 points)")

    print("\nOptimization Settings:")
    print(f"Number of optimization steps: {N_CALLS}")
    print(f"Random starts: {N_RANDOM_STARTS}")

    # Calculate theoretical search space
    period_space = PERIOD_MAX - PERIOD_MIN + 1
    multiplier_space = (MULTIPLIER_MAX - MULTIPLIER_MIN) * 10  # Account for decimals
    price_range_space = 26  # (3.00 - 0.50) / 0.10 + 1

    total_combinations = period_space * multiplier_space * price_range_space
    print(f"\nTheoretical parameter space: {total_combinations:,} combinations")

    # Memory estimation
    mem = psutil.virtual_memory()
    estimated_memory_mb = total_combinations * 0.5  # Rough estimate
    print(f"Estimated maximum memory required: ~{estimated_memory_mb:.1f} MB")
    print(f"Available memory: {mem.available / (1024**2):.1f} MB")

    return {
        'parameter_space': {
            'period_range': (PERIOD_MIN, PERIOD_MAX),
            'multiplier_range': (MULTIPLIER_MIN, MULTIPLIER_MAX),
            'price_range': (0.50, 3.01)
        },
        'optimization_settings': {
            'n_calls': N_CALLS,
            'n_random_starts': N_RANDOM_STARTS
        }
    }

def main():
    """Enhanced main execution function with optimization"""
    try:
        start_time = time.time()
        print("\nSuperTrend Strategy Backtester with Optimization")
        print("=" * 50)
        print(f"Started at (UTC): {CURRENT_UTC}")
        print(f"User: {CURRENT_USER}")

        # Initialize managers
        dir_manager = DirectoryManager()
        mem_manager = MemoryManager()
        batch_processor = BatchProcessor(dir_manager, mem_manager)

        print(f"\nCreated working directory: {dir_manager.base_dir}")
        print("Directory structure:")
        print(f"├── csv_dumps/")
        print(f"├── error_logs/")
        print(f"├── final_results/")
        print(f"├── plots/")
        print(f"├── cache/")
        print(f"└── optimization/")

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

        # Get parameter configuration
        params = get_parameter_inputs()

        # Process each selected file
        for file_path in selected_files:
            print(f"\nProcessing file: {os.path.basename(file_path)}")
            try:
                # Load OHLC data
                df = load_ohlc_data(file_path)
                
                # Initialize optimization manager
                optimization_manager = OptimizationManager(df, mem_manager)
                
                # Save input data metadata
                input_metadata = {
                    'filename': os.path.basename(file_path),
                    'rows': len(df),
                    'date_range': {
                        'start': str(df.index.min()),
                        'end': str(df.index.max())
                    },
                    'parameter_ranges': params['parameter_space'],
                    'optimization_settings': params['optimization_settings'],
                    'processed_at': CURRENT_UTC,
                    'processed_by': CURRENT_USER
                }
                
                with open(os.path.join(dir_manager.final_results_dir, 'input_metadata.json'), 'w') as f:
                    json.dump(input_metadata, f, indent=4)

                print("\nStarting optimization process...")
                batch_size = mem_manager.calculate_optimal_batch_size(
                    df.memory_usage().sum(),
                    N_CALLS
                )

                # Process with optimization
                results = batch_processor.process_optimized_batch(
                    df,
                    optimization_manager,
                    batch_size
                )
                
                if results:
                    batch_processor.save_batch_results(
                        results,
                        batch_processor.current_batch,
                        is_optimized=True
                    )
                    batch_processor.current_batch += 1
                
                # Merge results and generate final analysis
                print("\nGenerating final analysis...")
                final_results_df = batch_processor.merge_batch_results()
                
                results_manager = ResultsManager(dir_manager)
                results_manager.create_performance_summary(final_results_df)
                results_manager.save_detailed_results(final_results_df, df)

                # Clean up
                cleanup_gpu_memory()
                mem_manager.clear_optimization_cache()

                # Print execution summary
                end_time = time.time()
                duration = end_time - start_time
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = int(duration % 60)

                print(f"\nProcessing completed in {hours:02d}:{minutes:02d}:{seconds:02d}")
                print(f"Results saved in: {dir_manager.base_dir}")

                if not final_results_df.empty:
                    print("\nTop 5 Optimized Parameter Combinations:")
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
                        print(f"Optimization Method: {row.get('optimization_method', 'N/A')}")
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
CURRENT_USER = "arullr001"
TARGET_MAX = 15.00  # Maximum target (1500 points)

# Parameter Space Constants
PERIOD_MIN = 10
PERIOD_MAX = 100
MULTIPLIER_MIN = 5
MULTIPLIER_MAX = 30

# Directory Structure Constants
LOG_DIR = "logs"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
CACHE_DIR = "cache"

# Optimization Constants
N_CALLS = 100  # Number of optimization steps
N_RANDOM_STARTS = 20  # Number of random starts for optimization

class OptimizationSpace:
    """Defines the parameter space for optimization"""
    def __init__(self):
        self.dimensions = [
            Integer(PERIOD_MIN, PERIOD_MAX, name='period'),
            Real(MULTIPLIER_MIN, MULTIPLIER_MAX, name='multiplier'),
            Real(0.50, 3.01, name='price_range')
        ]
        
    def get_space(self):
        return self.dimensions
		
		
