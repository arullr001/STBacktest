# Standard library imports
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import glob
import time
import traceback
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import logging
from pathlib import Path
import math
import psutil
import gc

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# New performance optimization imports
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client, LocalCluster
import jax.numpy as jnp
from jax import jit, vmap
import ray
import modin.pandas as mpd
import vaex
import cudf
from rapids import cupy as cp
from numba import cuda, float64, int32
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

# Constants
THREADS_PER_BLOCK = 512  # Increased from 256
MAX_RAM_USAGE_PERCENT = 90
BATCH_SIZE = 1000
CURRENT_UTC = "2025-05-05 05:00:15"
CURRENT_USER = "arullr001"

# Fixed Parameter Ranges
PERIOD_RANGE = list(range(10, 101, 5))  # 10 to 100, step 5
MULTIPLIER_RANGE = [round(x, 2) for x in np.arange(3.0, 20.01, 0.01)]  # 3 to 20, step 0.01
PRICE_RANGE = [round(x, 2) for x in np.arange(0.05, 0.34, 0.01)]  # 0.05 to 0.33, step 0.01
TARGET_RANGE_MIN = 0.33
TARGET_RANGE_MAX = 1.1


###########ENHANCED MEMORY MANAGEMENT##############
class EnhancedMemoryManager:
    """Enhanced memory management with GPU support and optimization"""
    def __init__(self, max_ram_percent=MAX_RAM_USAGE_PERCENT):
        self.max_ram_percent = max_ram_percent
        self.total_ram = psutil.virtual_memory().total
        self.max_ram_bytes = (self.total_ram * self.max_ram_percent) / 100
        
        # Initialize GPU context if available
        self.has_gpu = cuda.is_available()
        if self.has_gpu:
            try:
                self.gpu_device = cuda.get_current_device()
                self.gpu_context = self.gpu_device.make_context()
            except Exception as e:
                logging.warning(f"GPU initialization failed: {e}")
                self.has_gpu = False

    def get_memory_status(self):
        """Get current memory status for both RAM and GPU"""
        status = {
            'ram': {
                'total': self.total_ram,
                'available': psutil.virtual_memory().available,
                'used': psutil.virtual_memory().used,
                'percent': psutil.virtual_memory().percent
            }
        }
        
        if self.has_gpu:
            try:
                free_gpu, total_gpu = self.gpu_device.get_memory_info()
                status['gpu'] = {
                    'total': total_gpu,
                    'free': free_gpu,
                    'used': total_gpu - free_gpu,
                    'percent': ((total_gpu - free_gpu) / total_gpu) * 100
                }
            except:
                status['gpu'] = None
        
        return status

    def optimize_batch_size(self, df_size, total_combinations):
        """Calculate optimal batch size based on available memory"""
        mem_status = self.get_memory_status()
        available_ram = mem_status['ram']['available']
        
        # Estimate memory per combination
        est_mem_per_combo = df_size * 1.5  # Reduced from 2.0
        
        # Calculate base batch size
        base_batch_size = int(available_ram / (est_mem_per_combo * 1.1))
        
        # Adjust for GPU if available
        if self.has_gpu:
            try:
                gpu_mem = mem_status['gpu']['free']
                gpu_batch_size = int(gpu_mem / (est_mem_per_combo * 1.2))
                base_batch_size = min(base_batch_size, gpu_batch_size)
            except:
                pass
        
        return min(base_batch_size, BATCH_SIZE, total_combinations)

    def cleanup(self):
        """Comprehensive memory cleanup"""
        # Clear Python memory
        gc.collect()
        
        # Clear GPU memory if available
        if self.has_gpu:
            try:
                self.gpu_context.pop()
                cuda.close()
                cuda.current_context().deallocations.clear()
            except:
                pass

class OptimizedDataLoader:
    """Enhanced data loading with Vaex and Modin support"""
    def __init__(self):
        self.current_engine = 'pandas'  # Default engine
        
    def determine_optimal_engine(self, file_path, file_size):
        """Determine the best engine based on file size and available memory"""
        total_ram = psutil.virtual_memory().total
        
        if file_size > total_ram * 0.5:  # If file size > 50% of RAM
            return 'vaex'
        elif file_size > total_ram * 0.25:  # If file size > 25% of RAM
            return 'modin'
        else:
            return 'pandas'
            
    def load_data(self, file_path):
        """Load data using the optimal engine"""
        try:
            # Get file size
            file_size = os.path.getsize(file_path)
            self.current_engine = self.determine_optimal_engine(file_path, file_size)
            
            print(f"Using {self.current_engine} engine for data loading...")
            
            if self.current_engine == 'vaex':
                # Use Vaex for out-of-core processing
                df = vaex.from_csv(file_path)
                required_columns = ['datetime', 'open', 'high', 'low', 'close']
                df = df[required_columns]
                return df.to_pandas_df()
                
            elif self.current_engine == 'modin':
                # Use Modin for distributed processing
                df = mpd.read_csv(file_path)
                
            else:
                # Use standard pandas
                df = pd.read_csv(file_path)
            
            # Process datetime and set index
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            
            # Optimize datatypes
            float_cols = ['open', 'high', 'low', 'close']
            for col in float_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], downcast='float')
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

class DistributedComputing:
    """Handles distributed computing setup and management"""
    def __init__(self, n_workers=None):
        self.n_workers = n_workers or max(1, os.cpu_count() - 1)
        self.dask_client = None
        self.ray_initialized = False
        
    def initialize_dask(self):
        """Initialize Dask client"""
        if self.dask_client is None:
            try:
                cluster = LocalCluster(n_workers=self.n_workers,
                                    threads_per_worker=2,
                                    memory_limit='auto')
                self.dask_client = Client(cluster)
                print(f"Dask initialized with {self.n_workers} workers")
            except Exception as e:
                logging.error(f"Dask initialization failed: {str(e)}")
                self.dask_client = None
                
    def initialize_ray(self):
        """Initialize Ray"""
        if not self.ray_initialized:
            try:
                ray.init(num_cpus=self.n_workers)
                self.ray_initialized = True
                print(f"Ray initialized with {self.n_workers} CPUs")
            except Exception as e:
                logging.error(f"Ray initialization failed: {str(e)}")
                self.ray_initialized = False
                
    def cleanup(self):
        """Cleanup distributed computing resources"""
        if self.dask_client is not None:
            try:
                self.dask_client.close()
                self.dask_client = None
            except:
                pass
                
        if self.ray_initialized:
            try:
                ray.shutdown()
                self.ray_initialized = False
            except:
                pass


##########GPU-accelerated calculations and parameter handling#########
@jit
def calculate_supertrend_jax(high, low, close, period, multiplier, price_range_1):
    """JAX-accelerated SuperTrend calculation"""
    n = len(close)
    hl2 = (high + low) / 2
    
    # Calculate TR and ATR
    tr = jnp.maximum(
        high[1:] - low[1:],
        jnp.maximum(
            jnp.abs(high[1:] - close[:-1]),
            jnp.abs(low[1:] - close[:-1])
        )
    )
    tr = jnp.concatenate([jnp.array([high[0] - low[0]]), tr])
    
    # Calculate ATR using exponential moving average
    atr = jnp.zeros_like(close)
    atr = atr.at[0].set(tr[0])
    
    def atr_step(carry, x):
        prev_atr, = carry
        curr_tr = x
        curr_atr = (prev_atr * (period - 1) + curr_tr) / period
        return (curr_atr,), curr_atr
    
    _, atr = jax.lax.scan(atr_step, (atr[0],), tr[1:])
    atr = jnp.concatenate([jnp.array([tr[0]]), atr])
    
    # Calculate basic bands
    basic_upper = hl2 - (multiplier * atr)
    basic_lower = hl2 + (multiplier * atr)
    
    return basic_upper, basic_lower, atr

class GPUAcceleratedCalculations:
    """Handles GPU-accelerated calculations using JAX and Rapids"""
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.has_gpu = cuda.is_available()
        self.use_rapids = self._check_rapids_availability()
        
    def _check_rapids_availability(self):
        """Check if Rapids (cuDF) can be used"""
        try:
            import cudf
            return True
        except ImportError:
            return False
            
    @staticmethod
    def _prepare_data_rapids(df):
        """Convert pandas DataFrame to cuDF DataFrame"""
        try:
            gpu_df = cudf.from_pandas(df)
            return gpu_df
        except Exception as e:
            logging.error(f"Rapids data preparation failed: {str(e)}")
            return None
            
    def calculate_supertrend(self, df, period, multiplier, price_range_1):
        """Calculate SuperTrend using available GPU acceleration"""
        if not self.has_gpu:
            return self._calculate_cpu(df, period, multiplier, price_range_1)
            
        try:
            if self.use_rapids:
                return self._calculate_rapids(df, period, multiplier, price_range_1)
            else:
                return self._calculate_jax(df, period, multiplier, price_range_1)
        except Exception as e:
            logging.error(f"GPU calculation failed, falling back to CPU: {str(e)}")
            return self._calculate_cpu(df, period, multiplier, price_range_1)
            
    def _calculate_rapids(self, df, period, multiplier, price_range_1):
        """Calculate SuperTrend using Rapids (cuDF)"""
        gpu_df = self._prepare_data_rapids(df)
        if gpu_df is None:
            return self._calculate_cpu(df, period, multiplier, price_range_1)
            
        try:
            # Calculate basic components
            gpu_df['hl2'] = (gpu_df['high'] + gpu_df['low']) / 2
            gpu_df['tr'] = gpu_df.eval(
                "max(high - low, "
                "abs(high - close.shift(1)), "
                "abs(low - close.shift(1)))"
            )
            
            # Calculate ATR
            gpu_df['atr'] = gpu_df['tr'].rolling(period).mean()
            
            # Calculate bands
            gpu_df['basic_upper'] = gpu_df['hl2'] - (multiplier * gpu_df['atr'])
            gpu_df['basic_lower'] = gpu_df['hl2'] + (multiplier * gpu_df['atr'])
            
            return gpu_df.to_pandas()
            
        except Exception as e:
            logging.error(f"Rapids calculation failed: {str(e)}")
            return self._calculate_cpu(df, period, multiplier, price_range_1)
            
    def _calculate_jax(self, df, period, multiplier, price_range_1):
        """Calculate SuperTrend using JAX"""
        try:
            # Convert to JAX arrays
            high = jnp.array(df['high'].values)
            low = jnp.array(df['low'].values)
            close = jnp.array(df['close'].values)
            
            # Calculate using JAX
            basic_upper, basic_lower, atr = calculate_supertrend_jax(
                high, low, close, period, multiplier, price_range_1
            )
            
            # Convert results back to numpy arrays
            df_result = df.copy()
            df_result['basic_upper'] = np.array(basic_upper)
            df_result['basic_lower'] = np.array(basic_lower)
            df_result['atr'] = np.array(atr)
            
            return df_result
            
        except Exception as e:
            logging.error(f"JAX calculation failed: {str(e)}")
            return self._calculate_cpu(df, period, multiplier, price_range_1)
            
    def _calculate_cpu(self, df, period, multiplier, price_range_1):
        """Fallback CPU calculation"""
        return calculate_supertrend_cpu(df, period, multiplier, price_range_1)

class ParameterHandler:
    """Handles parameter management and validation"""
    def __init__(self):
        self.period_range = PERIOD_RANGE
        self.multiplier_range = MULTIPLIER_RANGE
        self.price_range = PRICE_RANGE
        
    def get_target_inputs(self):
        """Get and validate target inputs from user"""
        while True:
            try:
                print("\nEnter target values (0.33 to 1.1):")
                long_target = float(input("Long target: "))
                short_target = float(input("Short target: "))
                
                if self.validate_targets(long_target, short_target):
                    return long_target, short_target
                    
            except ValueError:
                print("Please enter valid numbers")
                
    def validate_targets(self, long_target, short_target):
        """Validate target values"""
        if not (TARGET_RANGE_MIN <= long_target <= TARGET_RANGE_MAX):
            print(f"Long target must be between {TARGET_RANGE_MIN} and {TARGET_RANGE_MAX}")
            return False
            
        if not (TARGET_RANGE_MIN <= short_target <= TARGET_RANGE_MAX):
            print(f"Short target must be between {TARGET_RANGE_MIN} and {TARGET_RANGE_MAX}")
            return False
            
        return True
        
    def generate_parameter_combinations(self, long_target, short_target):
        """Generate all parameter combinations"""
        from itertools import product
        
        combinations = list(product(
            self.period_range,
            self.multiplier_range,
            self.price_range,
            [long_target],
            [short_target]
        ))
        
        return combinations
        
    def estimate_processing_time(self, n_combinations, n_workers):
        """Estimate processing time based on number of combinations"""
        # Rough estimate: 0.1 seconds per combination per worker
        estimated_seconds = (n_combinations * 0.1) / n_workers
        
        return {
            'total_combinations': n_combinations,
            'estimated_time': {
                'seconds': estimated_seconds,
                'minutes': estimated_seconds / 60,
                'hours': estimated_seconds / 3600
            }
        }



#########enhanced results processing and visualization########
class EnhancedResultsProcessor:
    """Advanced results processing with optimized performance and visualization"""
    def __init__(self, directory_manager, memory_manager):
        self.dir_manager = directory_manager
        self.mem_manager = memory_manager
        self.results_dir = Path(self.dir_manager.final_results_dir)
        self.visualization_dir = self.results_dir / 'visualizations'
        self.detailed_dir = self.results_dir / 'detailed_analysis'
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories"""
        self.visualization_dir.mkdir(exist_ok=True)
        self.detailed_dir.mkdir(exist_ok=True)
        
    def process_results(self, results_df):
        """Process and analyze results"""
        if results_df.empty:
            logging.warning("No results to process")
            return None
            
        # Sort by total profit
        results_df.sort_values('total_profit', ascending=False, inplace=True)
        
        # Get top 3 combinations
        top_3 = results_df.head(3)
        
        analysis = {
            'top_3_combinations': top_3.to_dict('records'),
            'overall_statistics': self._calculate_statistics(results_df),
            'processing_metadata': {
                'processed_at': CURRENT_UTC,
                'processed_by': CURRENT_USER,
                'total_combinations': len(results_df)
            }
        }
        
        # Save analysis
        self._save_analysis(analysis)
        
        return analysis
        
    def _calculate_statistics(self, df):
        """Calculate comprehensive statistics"""
        return {
            'profit_metrics': {
                'max_profit': float(df['total_profit'].max()),
                'min_profit': float(df['total_profit'].min()),
                'mean_profit': float(df['total_profit'].mean()),
                'median_profit': float(df['total_profit'].median()),
                'profit_std': float(df['total_profit'].std())
            },
            'win_rate_metrics': {
                'max_win_rate': float(df['win_rate'].max()),
                'min_win_rate': float(df['win_rate'].min()),
                'mean_win_rate': float(df['win_rate'].mean()),
                'median_win_rate': float(df['win_rate'].median())
            },
            'parameter_analysis': {
                'period_distribution': df['period'].value_counts().to_dict(),
                'multiplier_stats': {
                    'min': float(df['multiplier'].min()),
                    'max': float(df['multiplier'].max()),
                    'mean': float(df['multiplier'].mean())
                },
                'price_range_stats': {
                    'min': float(df['price_range_1'].min()),
                    'max': float(df['price_range_1'].max()),
                    'mean': float(df['price_range_1'].mean())
                }
            }
        }
        
    def create_visualizations(self, results_df, top_n=3):
        """Create enhanced visualizations"""
        try:
            # Set style for better visualizations
            plt.style.use('seaborn-darkgrid')
            
            self._create_profit_distribution_plot(results_df)
            self._create_parameter_correlation_plot(results_df)
            self._create_top_performers_plot(results_df, top_n)
            self._create_win_rate_analysis_plot(results_df)
            self._create_parameter_impact_plot(results_df)
            
        except Exception as e:
            logging.error(f"Error creating visualizations: {str(e)}")
            
    def _create_profit_distribution_plot(self, df):
        """Create profit distribution visualization"""
        plt.figure(figsize=(12, 6))
        
        # Create main distribution plot
        sns.histplot(data=df, x='total_profit', kde=True)
        plt.title('Profit Distribution Analysis')
        plt.xlabel('Total Profit')
        plt.ylabel('Frequency')
        
        # Add statistical annotations
        stats_text = f"Mean: {df['total_profit'].mean():.2f}\n"
        stats_text += f"Median: {df['total_profit'].median():.2f}\n"
        stats_text += f"Std: {df['total_profit'].std():.2f}"
        
        plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                    bbox=dict(facecolor='white', alpha=0.8),
                    ha='right', va='top')
                    
        plt.savefig(self.visualization_dir / 'profit_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_parameter_correlation_plot(self, df):
        """Create parameter correlation visualization"""
        plt.figure(figsize=(10, 8))
        
        # Calculate correlations
        corr_matrix = df[['period', 'multiplier', 'price_range_1', 'total_profit', 'win_rate']].corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Parameter Correlation Analysis')
        
        plt.savefig(self.visualization_dir / 'parameter_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_top_performers_plot(self, df, top_n):
        """Create top performers visualization"""
        top_df = df.nlargest(top_n, 'total_profit')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bar plot
        bars = ax.bar(range(top_n), top_df['total_profit'])
        
        # Add parameter annotations
        for idx, (i, row) in enumerate(top_df.iterrows()):
            ax.text(idx, row['total_profit'], 
                   f"P:{row['period']}\nM:{row['multiplier']:.2f}\nR:{row['price_range_1']:.2f}",
                   ha='center', va='bottom')
                   
        plt.title(f'Top {top_n} Performing Parameter Combinations')
        plt.xlabel('Rank')
        plt.ylabel('Total Profit')
        
        plt.savefig(self.visualization_dir / 'top_performers.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_detailed_results(self, results_df, trades_data):
        """Save detailed analysis of results"""
        try:
            # Save top 3 detailed analysis
            top_3 = results_df.head(3)
            for idx, row in top_3.iterrows():
                combination_trades = trades_data.get(idx, [])
                if combination_trades:
                    self._save_combination_analysis(idx, row, combination_trades)
                    
        except Exception as e:
            logging.error(f"Error saving detailed results: {str(e)}")
            
    def _save_combination_analysis(self, idx, params, trades):
        """Save detailed analysis for a parameter combination"""
        analysis_dir = self.detailed_dir / f'combination_{idx}'
        analysis_dir.mkdir(exist_ok=True)
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Calculate trade metrics
        metrics = {
            'parameters': params.to_dict(),
            'trade_metrics': {
                'total_trades': len(trades_df),
                'winning_trades': len(trades_df[trades_df['points'] > 0]),
                'losing_trades': len(trades_df[trades_df['points'] < 0]),
                'win_rate': len(trades_df[trades_df['points'] > 0]) / len(trades_df),
                'average_win': trades_df[trades_df['points'] > 0]['points'].mean(),
                'average_loss': trades_df[trades_df['points'] < 0]['points'].mean(),
                'largest_win': trades_df['points'].max(),
                'largest_loss': trades_df['points'].min(),
                'profit_factor': abs(trades_df[trades_df['points'] > 0]['points'].sum() / 
                                   trades_df[trades_df['points'] < 0]['points'].sum())
            }
        }
        
        # Save detailed analysis
        with open(analysis_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
            
        # Save trades
        trades_df.to_csv(analysis_dir / 'trades.csv', index=False)
        
        # Create trade analysis visualizations
        self._create_trade_visualizations(trades_df, analysis_dir)
        
    def _create_trade_visualizations(self, trades_df, save_dir):
        """Create detailed trade analysis visualizations"""
        # Profit distribution over time
        plt.figure(figsize=(12, 6))
        trades_df['cumulative_profit'] = trades_df['points'].cumsum()
        plt.plot(trades_df.index, trades_df['cumulative_profit'])
        plt.title('Cumulative Profit Over Time')
        plt.xlabel('Trade Number')
        plt.ylabel('Cumulative Profit')
        plt.savefig(save_dir / 'cumulative_profit.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        
        
###########main execution flow##########
class SupertrendOptimizer:
    """Main class coordinating the optimization process"""
    def __init__(self):
        self.dir_manager = DirectoryManager()
        self.mem_manager = EnhancedMemoryManager()
        self.data_loader = OptimizedDataLoader()
        self.distributed_computing = DistributedComputing()
        self.gpu_calculator = GPUAcceleratedCalculations(self.mem_manager)
        self.param_handler = ParameterHandler()
        self.results_processor = EnhancedResultsProcessor(self.dir_manager, self.mem_manager)
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Setup logging configuration"""
        logger = logging.getLogger('supertrend_optimizer')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        file_handler = logging.FileHandler(
            Path(self.dir_manager.error_logs_dir) / 'optimization.log'
        )
        console_handler = logging.StreamHandler()
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def run_optimization(self, data_file):
        """Main optimization process"""
        try:
            self.logger.info("Starting optimization process")
            
            # 1. Initialize distributed computing
            self.logger.info("Initializing distributed computing")
            self.distributed_computing.initialize_dask()
            self.distributed_computing.initialize_ray()
            
            # 2. Load and prepare data
            self.logger.info(f"Loading data from {data_file}")
            df = self.data_loader.load_data(data_file)
            
            # 3. Get target inputs
            self.logger.info("Getting target inputs")
            long_target, short_target = self.param_handler.get_target_inputs()
            
            # 4. Generate parameter combinations
            self.logger.info("Generating parameter combinations")
            combinations = self.param_handler.generate_parameter_combinations(
                long_target, short_target
            )
            
            # 5. Estimate processing time
            estimate = self.param_handler.estimate_processing_time(
                len(combinations),
                self.distributed_computing.n_workers
            )
            self.logger.info(
                f"Estimated processing time: {estimate['estimated_time']['minutes']:.2f} minutes"
            )
            
            # 6. Process combinations in batches
            results = self._process_combinations(df, combinations)
            
            # 7. Process and visualize results
            self.logger.info("Processing results")
            analysis = self.results_processor.process_results(results)
            
            # 8. Create visualizations
            self.logger.info("Creating visualizations")
            self.results_processor.create_visualizations(results)
            
            # 9. Save detailed results
            self.logger.info("Saving detailed results")
            self.results_processor.save_detailed_results(
                results,
                self._get_trades_for_top_combinations(df, results)
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Optimization process failed: {str(e)}")
            raise
        finally:
            self.cleanup()

    def _process_combinations(self, df, combinations):
        """Process parameter combinations using distributed computing"""
        total_combinations = len(combinations)
        batch_size = self.mem_manager.optimize_batch_size(
            df.memory_usage().sum(),
            total_combinations
        )
        
        results_list = []
        
        with tqdm(total=total_combinations, desc="Processing combinations") as pbar:
            for i in range(0, total_combinations, batch_size):
                batch = combinations[i:i + batch_size]
                
                # Process batch using Ray
                @ray.remote
                def process_batch(batch_combinations):
                    return [
                        self._process_single_combination(df, *params)
                        for params in batch_combinations
                    ]
                
                batch_results = ray.get(process_batch.remote(batch))
                results_list.extend(batch_results)
                
                # Update progress
                pbar.update(len(batch))
                
                # Cleanup after batch
                self.mem_manager.cleanup()
        
        return pd.DataFrame(results_list)

    def _process_single_combination(self, df, period, multiplier, price_range, 
                                  long_target, short_target):
        """Process a single parameter combination"""
        try:
            # Calculate SuperTrend
            st_df = self.gpu_calculator.calculate_supertrend(
                df, period, multiplier, price_range
            )
            
            # Run backtest
            result = backtest_supertrend(
                st_df, period, multiplier, price_range,
                long_target, short_target
            )
            
            return {
                'period': period,
                'multiplier': multiplier,
                'price_range_1': price_range,
                'long_target': long_target,
                'short_target': short_target,
                'total_profit': result['total_profit'],
                'trade_count': result['trade_count'],
                'win_rate': result['win_rate']
            }
            
        except Exception as e:
            self.logger.error(
                f"Error processing combination: {period}, {multiplier}, "
                f"{price_range}, {long_target}, {short_target}: {str(e)}"
            )
            return None

    def _get_trades_for_top_combinations(self, df, results_df, top_n=3):
        """Get detailed trade data for top combinations"""
        trades_data = {}
        top_combinations = results_df.head(top_n)
        
        for idx, row in top_combinations.iterrows():
            # Calculate SuperTrend
            st_df = self.gpu_calculator.calculate_supertrend(
                df,
                row['period'],
                row['multiplier'],
                row['price_range_1']
            )
            
            # Run backtest
            result = backtest_supertrend(
                st_df,
                row['period'],
                row['multiplier'],
                row['price_range_1'],
                row['long_target'],
                row['short_target']
            )
            
            trades_data[idx] = result['trades']
            
        return trades_data

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.distributed_computing.cleanup()
            self.mem_manager.cleanup()
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")

def main():
    """Main execution function"""
    try:
        # Initialize optimizer
        optimizer = SupertrendOptimizer()
        
        # Get data file
        print("\nEnter the path to your OHLC data file:")
        data_file = input("> ").strip()
        
        if not os.path.exists(data_file):
            print("Error: File not found")
            return
            
        # Run optimization
        analysis = optimizer.run_optimization(data_file)
        
        # Display results
        print("\nOptimization completed!")
        print("\nTop 3 Parameter Combinations:")
        for idx, combo in enumerate(analysis['top_3_combinations'], 1):
            print(f"\n{idx}. Combination:")
            print(f"   Period: {combo['period']}")
            print(f"   Multiplier: {combo['multiplier']:.2f}")
            print(f"   Price Range: {combo['price_range_1']:.2f}")
            print(f"   Total Profit: {combo['total_profit']:.2f}")
            print(f"   Win Rate: {combo['win_rate']*100:.2f}%")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
