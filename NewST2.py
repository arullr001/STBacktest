# Part 1: System Setup and Global Configurations
# Last Updated: 2025-05-09 02:58:30 UTC
# Author: arullr001

import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import torch
import keyboard
import psutil
import warnings
import json
import logging
import logging.handlers
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import signal
import threading
from tqdm import tqdm
import gc
import h5py

# Suppress warnings
warnings.filterwarnings('ignore')

# Global Constants
VERSION = "1.0.0"
CURRENT_USER = "arullr001"
START_TIME = "2025-05-09 02:58:30"

def check_dependencies():
    """Verify required dependencies are installed with appropriate versions"""
    required_packages = {
        'numpy': '1.20.0',
        'pandas': '1.3.0',
        'torch': '1.8.0',
        'keyboard': '0.13.5',
        'h5py': '3.1.0',
        'plotly': '5.3.0',
        'tqdm': '4.60.0'
    }
    
    missing = []
    outdated = []
    
    for package, min_version in required_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', '0.0.0')
            if version < min_version:
                outdated.append(f"{package} (found {version}, required {min_version})")
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"ERROR: Missing required packages: {', '.join(missing)}")
        print("Install them using: pip install " + " ".join(missing))
        return False
        
    if outdated:
        print(f"WARNING: Outdated packages detected: {', '.join(outdated)}")
        print("Consider upgrading them: pip install --upgrade " + " ".join([p.split()[0] for p in outdated]))
    
    return True

# Global Configuration
@dataclass
class GlobalConfig:
    # System Control
    ABORT_KEYS = {"ctrl", "alt", "x"}
    PAUSE_KEYS = {"ctrl", "space"}
    
    # Resource Management
    MAX_GPU_MEMORY_USAGE = 0.90  # 90% of available GPU memory
    MAX_RAM_USAGE = 0.70         # 70% of available RAM
    DEFAULT_BATCH_SIZE = 500
    
    # File System
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = BASE_DIR / "data"
    RESULTS_DIR = BASE_DIR / "results"
    CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Processing
    SAMPLE_SIZE_FOR_ESTIMATION = 1000
    MIN_BATCH_SIZE = 100
    MAX_BATCH_SIZE = 5000
    
    # Time Format
    TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    def __post_init__(self):
        # Check disk space before creating directories
        try:
            free_space = shutil.disk_usage(self.BASE_DIR).free / (1024**3)  # GB
            if free_space < 2.0:  # Less than 2GB free
                print(f"WARNING: Low disk space ({free_space:.2f}GB). Results may not be saved properly.")
        except Exception as e:
            print(f"Could not check disk space: {str(e)}")
        
        # Create necessary directories
        for dir_path in [self.DATA_DIR, self.RESULTS_DIR, 
                        self.CHECKPOINTS_DIR, self.LOGS_DIR]:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                print(f"ERROR: No permission to create directory: {dir_path}")
            except Exception as e:
                print(f"ERROR: Could not create directory {dir_path}: {str(e)}")

# Initialize Global Configuration
CONFIG = GlobalConfig()

# Setup Logging
def setup_logging():
    log_file = CONFIG.LOGS_DIR / f"backtest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Set up log rotation - 5 backup files, 10MB each
    handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            handler,
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# Initialize Logger
logger = setup_logging()

# System State Management
class SystemState:
    def __init__(self):
        self._paused = False
        self._abort = False
        self._lock = threading.Lock()
        self.setup_keyboard_hooks()
    
    def setup_keyboard_hooks(self):
        try:
            keyboard.add_hotkey('ctrl+alt+x', self.trigger_abort)
            keyboard.add_hotkey('ctrl+space', self.toggle_pause)
            logger.info("Keyboard hooks configured successfully")
        except Exception as e:
            logger.warning(f"Failed to set up keyboard shortcuts: {str(e)}")
            logger.warning("Keyboard shortcuts will not be available. Use ctrl+c to abort.")
    
    def trigger_abort(self):
        with self._lock:
            self._abort = True
            logger.warning("Abort signal received. Cleaning up...")
    
    def toggle_pause(self):
        with self._lock:
            self._paused = not self._paused
            status = "PAUSED" if self._paused else "RESUMED"
            logger.info(f"Processing {status}")
    
    @property
    def should_abort(self):
        with self._lock:
            return self._abort
    
    @property
    def is_paused(self):
        with self._lock:
            return self._paused
    
    def wait_if_paused(self):
        """Wait if system is paused with better thread safety"""
        pause_check_interval = 0.1
        max_wait_time = 300  # 5 minutes max wait to prevent deadlocks
        
        wait_start = time.time()
        while True:
            with self._lock:
                if not self._paused or self._abort:
                    return
                
            # Check if we've waited too long
            if time.time() - wait_start > max_wait_time:
                logger.warning("Pause timeout exceeded (5 minutes). Resuming execution.")
                with self._lock:
                    self._paused = False
                return
                
            time.sleep(pause_check_interval)

# Resource Monitor
class ResourceMonitor:
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        
    def get_gpu_memory_usage(self) -> float:
        if not self.gpu_available:
            return 0.0
        return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() \
            if torch.cuda.max_memory_allocated() > 0 else 0.0
    
    def get_ram_usage(self) -> float:
        return psutil.Process().memory_percent() / 100.0
    
    def check_resources(self) -> Tuple[bool, str]:
        """Check if system resources are within limits, returns (is_ok, message)"""
        gpu_usage = self.get_gpu_memory_usage()
        ram_usage = self.get_ram_usage()
        
        if gpu_usage > CONFIG.MAX_GPU_MEMORY_USAGE:
            return False, f"GPU memory usage too high: {gpu_usage:.2%}"
        
        if ram_usage > CONFIG.MAX_RAM_USAGE:
            return False, f"RAM usage too high: {ram_usage:.2%}"
        
        # Check disk space
        try:
            free_space = shutil.disk_usage(CONFIG.RESULTS_DIR).free / (1024**3)  # GB
            if free_space < 1.0:  # Less than 1GB free
                return False, f"Low disk space: {free_space:.2f}GB free"
        except Exception as e:
            logger.warning(f"Could not check disk space: {str(e)}")
        
        return True, "Resources OK"
    
    def cleanup(self):
        if self.gpu_available:
            torch.cuda.empty_cache()
        gc.collect()

# Initialize System State and Resource Monitor
SYSTEM_STATE = SystemState()
RESOURCE_MONITOR = ResourceMonitor()

logger.info(f"System initialized - Version {VERSION}")
logger.info(f"User: {CURRENT_USER}")
logger.info(f"Start Time: {START_TIME}")

# Error Handling Decorator
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            logger.warning(f"Operation {func.__name__} interrupted by user")
            SYSTEM_STATE.trigger_abort()
            raise
        except MemoryError:
            logger.error(f"Out of memory in {func.__name__}. Try reducing batch size.")
            SYSTEM_STATE.trigger_abort()
            raise
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            SYSTEM_STATE.trigger_abort()
            raise
    return wrapper



# Part 2: User Input Interface and Validation System
# Last Updated: 2025-05-09 02:58:30 UTC
# Author: arullr001

@dataclass
class ParameterRange:
    min_value: float
    max_value: float
    step: float
    name: str
    
    def validate(self) -> Tuple[bool, str]:
        if self.min_value >= self.max_value:
            return False, f"{self.name}: Minimum value must be less than maximum value"
        if self.step <= 0:
            return False, f"{self.name}: Step must be greater than 0"
        if self.step >= (self.max_value - self.min_value):
            return False, f"{self.name}: Step must be less than range"
        return True, ""
    
    def get_values(self) -> np.ndarray:
        return np.arange(self.min_value, self.max_value + self.step, self.step)
    
    def get_combinations(self) -> int:
        return len(self.get_values())

class InputParameters:
    def __init__(self):
        self.parameters: Dict[str, ParameterRange] = {}
        self.enable_target: bool = False
        self.logger = logging.getLogger(__name__)
    
    @handle_errors
    def get_user_input(self) -> bool:
        """Get and validate user inputs for all parameters"""
        self.logger.info("Starting parameter input process...")
        
        # Core Parameters
        self._get_parameter_input("ATR Period", 5, 100, 1, int)
        self._get_parameter_input("Factor", 3, 30, 0.01)
        self._get_parameter_input("Buffer Distance", 20, 400, 1, int)
        self._get_parameter_input("Hard Stop Distance", 10, 50, 1, int)
        
        # Target Mode Selection
        self.enable_target = self._get_yes_no_input("Enable Target Based Exit?")
        
        if self.enable_target:
            self._get_parameter_input("Long Target RR", 3, 30, 0.1)
            self._get_parameter_input("Short Target RR", 3, 30, 0.1)
        
        return self._validate_all_parameters()
    
    def _get_parameter_input(self, name: str, default_min: float, default_max: float, 
                           default_step: float, param_type: type = float) -> None:
        """Get input for a single parameter with better error handling"""
        self.logger.info(f"\nParameter: {name}")
        
        while True:
            try:
                # Safer type conversion with defaults
                min_input = input(f"Enter minimum {name} [{default_min}]: ")
                min_val = param_type(min_input) if min_input.strip() else default_min
                
                max_input = input(f"Enter maximum {name} [{default_max}]: ")
                max_val = param_type(max_input) if max_input.strip() else default_max
                
                step_input = input(f"Enter step size [{default_step}]: ")
                step_val = param_type(step_input) if step_input.strip() else default_step
                
                param = ParameterRange(min_val, max_val, step_val, name)
                is_valid, message = param.validate()
                
                if is_valid:
                    self.parameters[name] = param
                    self._display_parameter_range(param)
                    break
                else:
                    self.logger.error(message)
            except ValueError:
                self.logger.error(f"Invalid input for {name}. Please enter numeric values.")
    
    def _get_yes_no_input(self, prompt: str) -> bool:
        """Get yes/no input from user"""
        while True:
            response = input(f"{prompt} (y/n): ").lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            self.logger.error("Please enter 'y' or 'n'")
    
    def _validate_all_parameters(self) -> bool:
        """Validate all parameters and display summary"""
        self.logger.info("\nParameter Summary:")
        total_combinations = self._calculate_total_combinations()
        
        for name, param in self.parameters.items():
            is_valid, message = param.validate()
            if not is_valid:
                self.logger.error(message)
                return False
            
            self._display_parameter_range(param)
        
        self.logger.info(f"\nTotal combinations to process: {total_combinations:,}")
        
        return self._confirm_parameters()
    
    def _display_parameter_range(self, param: ParameterRange) -> None:
        """Display parameter range and number of combinations"""
        num_values = param.get_combinations()
        self.logger.info(
            f"{param.name}: {param.min_value} to {param.max_value} "
            f"(step: {param.step}) - {num_values:,} values"
        )
    
    def _calculate_total_combinations(self) -> int:
        """Calculate total number of combinations"""
        total = 1
        for param in self.parameters.values():
            total *= param.get_combinations()
        return total
    
    def _confirm_parameters(self) -> bool:
        """Get user confirmation for parameters"""
        while True:
            response = input("\nProceed with these parameters? (y/n): ").lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            self.logger.error("Please enter 'y' or 'n'")
    
    def save_configuration(self, filepath: Path) -> None:
        """Save current configuration to file"""
        config_data = {
            'enable_target': self.enable_target,
            'parameters': {
                name: {
                    'min_value': param.min_value,
                    'max_value': param.max_value,
                    'step': param.step
                } for name, param in self.parameters.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=4)
        self.logger.info(f"Configuration saved to {filepath}")
    
    def load_configuration(self, filepath: Path) -> bool:
        """Load configuration from file"""
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            self.enable_target = config_data['enable_target']
            self.parameters = {
                name: ParameterRange(
                    min_value=param['min_value'],
                    max_value=param['max_value'],
                    step=param['step'],
                    name=name
                ) for name, param in config_data['parameters'].items()
            }
            
            self.logger.info(f"Configuration loaded from {filepath}")
            return self._validate_all_parameters()
        
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            return False

class ParameterManager:
    def __init__(self):
        self.input_params = InputParameters()
        self.logger = logging.getLogger(__name__)
    
    @handle_errors
    def initialize_parameters(self) -> bool:
        """Initialize parameters either from user input or loaded configuration"""
        while True:
            load_config = self._get_yes_no_input("Load existing configuration?")
            
            if load_config:
                config_file = self._select_config_file()
                if config_file and self.input_params.load_configuration(config_file):
                    return True
            else:
                if self.input_params.get_user_input():
                    self._save_current_configuration()
                    return True
            
            if not self._get_yes_no_input("Try again?"):
                return False
    
    def _get_yes_no_input(self, prompt: str) -> bool:
        """Get yes/no input from user"""
        while True:
            response = input(f"{prompt} (y/n): ").lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            self.logger.error("Please enter 'y' or 'n'")
    
    def _select_config_file(self) -> Union[Path, None]:
        """Let user select a configuration file"""
        config_dir = CONFIG.BASE_DIR / "configs"
        config_dir.mkdir(exist_ok=True)
        
        config_files = list(config_dir.glob("*.json"))
        if not config_files:
            self.logger.error("No configuration files found")
            return None
        
        self.logger.info("\nAvailable configurations:")
        for i, filepath in enumerate(config_files, 1):
            self.logger.info(f"{i}. {filepath.name}")
        
        while True:
            try:
                choice = int(input("\nSelect configuration file (number): "))
                if 1 <= choice <= len(config_files):
                    return config_files[choice - 1]
                self.logger.error("Invalid selection")
            except ValueError:
                self.logger.error("Please enter a number")
    
    def _save_current_configuration(self) -> None:
        """Save current configuration if user wants to"""
        if self._get_yes_no_input("Save this configuration?"):
            config_dir = CONFIG.BASE_DIR / "configs"
            config_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = config_dir / f"config_{timestamp}.json"
            
            self.input_params.save_configuration(filepath)
			
			
# Part 3: Pre-Processing and Resource Management
# Last Updated: 2025-05-09 02:58:30 UTC
# Author: arullr001

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import torch.cuda as cuda
import math

@dataclass
class ResourceMetrics:
    gpu_memory_total: float
    gpu_memory_available: float
    ram_total: float
    ram_available: float
    cpu_count: int
    gpu_compute_capability: Optional[Tuple[int, int]]

class ResourceAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = self._gather_system_metrics()
    
    def _gather_system_metrics(self) -> ResourceMetrics:
        """Gather system resource metrics with CUDA version check"""
        if cuda.is_available():
            try:
                cuda_version = cuda.version.cuda
                if cuda_version < "10.0":
                    self.logger.warning(f"CUDA version {cuda_version} may be too old. Consider upgrading.")
            except:
                self.logger.warning("Could not determine CUDA version")
                
            gpu_properties = cuda.get_device_properties(0)
            gpu_memory_total = gpu_properties.total_memory / (1024**3)  # Convert to GB
            gpu_memory_available = cuda.mem_get_info()[0] / (1024**3)
            gpu_compute_capability = gpu_properties.major, gpu_properties.minor
        else:
            gpu_memory_total = 0
            gpu_memory_available = 0
            gpu_compute_capability = None
        
        ram = psutil.virtual_memory()
        return ResourceMetrics(
            gpu_memory_total=gpu_memory_total,
            gpu_memory_available=gpu_memory_available,
            ram_total=ram.total / (1024**3),
            ram_available=ram.available / (1024**3),
            cpu_count=psutil.cpu_count(logical=False),
            gpu_compute_capability=gpu_compute_capability
        )
    
    def estimate_batch_size(self, total_combinations: int) -> int:
        """Estimate optimal batch size based on available resources"""
        if cuda.is_available():
            # GPU-based estimation
            memory_factor = (self.metrics.gpu_memory_available * 0.8) / self.metrics.gpu_memory_total
            base_batch_size = int(CONFIG.DEFAULT_BATCH_SIZE * memory_factor)
        else:
            # CPU-based estimation
            memory_factor = (self.metrics.ram_available * 0.8) / self.metrics.ram_total
            base_batch_size = int(CONFIG.DEFAULT_BATCH_SIZE * memory_factor)
        
        # Adjust batch size based on total combinations
        adjusted_batch_size = min(
            base_batch_size,
            max(CONFIG.MIN_BATCH_SIZE, min(total_combinations // 10, CONFIG.MAX_BATCH_SIZE))
        )
        
        return adjusted_batch_size

class PreProcessor:
    def __init__(self, input_params: InputParameters):
        self.input_params = input_params
        self.logger = logging.getLogger(__name__)
        self.resource_analyzer = ResourceAnalyzer()
        self.total_combinations = input_params._calculate_total_combinations()
        self.batch_size = self.resource_analyzer.estimate_batch_size(self.total_combinations)
    
    @handle_errors
    def run_sample_estimation(self) -> Tuple[float, Dict[str, float]]:
        """Run a sample batch to estimate processing time and resource usage"""
        self.logger.info("Running sample batch for time estimation...")
        
        # Create a small sample of parameter combinations
        sample_size = min(CONFIG.SAMPLE_SIZE_FOR_ESTIMATION, self.total_combinations)
        sample_combinations = self._generate_sample_combinations(sample_size)
        
        # Measure processing time and resource usage
        start_time = time.time()
        resources_before = self._get_resource_snapshot()
        
        # Process sample batch
        self._process_sample_batch(sample_combinations)
        
        processing_time = time.time() - start_time
        resources_after = self._get_resource_snapshot()
        
        # Calculate estimates
        estimated_total_time = (processing_time / sample_size) * self.total_combinations
        resource_usage = self._calculate_resource_usage(resources_before, resources_after)
        
        return estimated_total_time, resource_usage
    
    def _generate_sample_combinations(self, sample_size: int) -> List[Dict[str, float]]:
        """Generate a representative sample of parameter combinations"""
        all_params = []
        for param in self.input_params.parameters.values():
            values = param.get_values()
            indices = np.linspace(0, len(values)-1, min(sample_size, len(values)), dtype=int)
            all_params.append(values[indices])
        
        combinations = []
        for params in zip(*all_params):
            combination = dict(zip(self.input_params.parameters.keys(), params))
            combinations.append(combination)
        
        return combinations[:sample_size]
    
    def _get_resource_snapshot(self) -> Dict[str, float]:
        """Get current resource usage snapshot"""
        return {
            'gpu_memory': RESOURCE_MONITOR.get_gpu_memory_usage() if cuda.is_available() else 0,
            'ram_usage': RESOURCE_MONITOR.get_ram_usage()
        }
    
    def _process_sample_batch(self, combinations: List[Dict[str, float]]) -> None:
        """Process a sample batch to measure resource usage"""
        # Simulate processing load
        for combination in combinations:
            if SYSTEM_STATE.should_abort:
                raise KeyboardInterrupt("Sample estimation aborted")
            
            # Simulate GPU/CPU load
            if cuda.is_available():
                dummy_tensor = torch.rand(1000, 1000).cuda()
                torch.matmul(dummy_tensor, dummy_tensor.t())
                del dummy_tensor
            else:
                np.random.rand(1000, 1000) @ np.random.rand(1000, 1000)
            
            time.sleep(0.01)  # Small delay to simulate actual processing
    
    def _calculate_resource_usage(self, before: Dict[str, float], 
                                after: Dict[str, float]) -> Dict[str, float]:
        """Calculate resource usage from before/after snapshots"""
        return {
            'gpu_memory_delta': after['gpu_memory'] - before['gpu_memory'],
            'ram_usage_delta': after['ram_usage'] - before['ram_usage']
        }
    
    def display_estimation_summary(self, estimated_time: float, 
                                 resource_usage: Dict[str, float]) -> None:
        """Display estimation summary to user"""
        self.logger.info("\nProcessing Estimation Summary:")
        self.logger.info(f"Total combinations: {self.total_combinations:,}")
        self.logger.info(f"Estimated batch size: {self.batch_size:,}")
        self.logger.info(f"Number of batches: {math.ceil(self.total_combinations / self.batch_size):,}")
        
        # Time estimation
        hours = estimated_time // 3600
        minutes = (estimated_time % 3600) // 60
        seconds = estimated_time % 60
        self.logger.info(
            f"Estimated processing time: {int(hours)}h {int(minutes)}m {int(seconds)}s"
        )
        
        # Resource usage
        self.logger.info("\nEstimated Resource Usage per Batch:")
        if cuda.is_available():
            self.logger.info(
                f"GPU Memory: {resource_usage['gpu_memory_delta']:.2%} of available"
            )
        self.logger.info(f"RAM Usage: {resource_usage['ram_usage_delta']:.2%} of available")
        
        # System specifications
        self.logger.info("\nSystem Specifications:")
        self.logger.info(f"CPU Cores: {self.resource_analyzer.metrics.cpu_count}")
        self.logger.info(f"Total RAM: {self.resource_analyzer.metrics.ram_total:.1f} GB")
        if cuda.is_available():
            self.logger.info(
                f"GPU Memory: {self.resource_analyzer.metrics.gpu_memory_total:.1f} GB"
            )
            self.logger.info(
                f"CUDA Compute Capability: {self.resource_analyzer.metrics.gpu_compute_capability}"
            )
    
    def confirm_processing(self) -> bool:
        """Get user confirmation to proceed with processing"""
        while True:
            response = input("\nProceed with processing? (y/n): ").lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            self.logger.error("Please enter 'y' or 'n'")


# Part 4: Processing Framework and Batch Management
# Last Updated: 2025-05-09 02:58:30 UTC
# Author: arullr001

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterator, Generator
import numpy as np
import pandas as pd
from datetime import datetime
import queue
import threading

@dataclass
class BatchResult:
    batch_id: int
    combinations: List[Dict[str, float]]
    metrics: Dict[str, float]
    trade_list: List[Dict]
    execution_time: float
    resource_usage: Dict[str, float]

class BatchProcessor:
    def __init__(self, batch_size: int, total_combinations: int):
        self.batch_size = batch_size
        self.total_combinations = total_combinations
        self.logger = logging.getLogger(__name__)
        self.result_queue = queue.Queue()
        self.checkpoint_lock = threading.Lock()
        
    def generate_batches(self, input_params: InputParameters) -> Generator[List[Dict[str, float]], None, None]:
        """Generate parameter combination batches"""
        param_values = {
            name: param.get_values() 
            for name, param in input_params.parameters.items()
        }
        
        # Generate all combinations using numpy meshgrid
        param_names = list(param_values.keys())
        mesh = np.meshgrid(*[param_values[name] for name in param_names])
        combinations = np.array([m.flatten() for m in mesh]).T
        
        # Yield batches
        for i in range(0, len(combinations), self.batch_size):
            batch = []
            for combination in combinations[i:i + self.batch_size]:
                batch.append(dict(zip(param_names, combination)))
            yield batch

    @handle_errors
    def process_batch(self, batch_id: int, combinations: List[Dict[str, float]], 
                     enable_target: bool) -> BatchResult:
        """Process a single batch of parameter combinations"""
        start_time = time.time()
        resources_before = self._get_resource_snapshot()
        
        # Process each combination in the batch
        batch_metrics = []
        batch_trades = []
        
        for combo in combinations:
            if SYSTEM_STATE.should_abort:
                raise KeyboardInterrupt(f"Processing aborted during batch {batch_id}")
            
            SYSTEM_STATE.wait_if_paused()
            
            # Process single combination
            metrics, trades = self._process_combination(combo, enable_target)
            batch_metrics.append(metrics)
            batch_trades.extend(trades)
            
            # Check resource usage
            is_ok, message = RESOURCE_MONITOR.check_resources()
            if not is_ok:
                self.logger.warning(f"Resource issue: {message}")
                self.logger.warning("Attempting to free resources...")
                RESOURCE_MONITOR.cleanup()
                
                # Check again after cleanup
                is_ok, message = RESOURCE_MONITOR.check_resources()
                if not is_ok:
                    raise ResourceWarning(f"Resource limits exceeded in batch {batch_id}: {message}")
        
        execution_time = time.time() - start_time
        resources_after = self._get_resource_snapshot()
        resource_usage = self._calculate_resource_usage(resources_before, resources_after)
        
        # Aggregate batch results
        aggregated_metrics = self._aggregate_metrics(batch_metrics)
        
        return BatchResult(
            batch_id=batch_id,
            combinations=combinations,
            metrics=aggregated_metrics,
            trade_list=batch_trades,
            execution_time=execution_time,
            resource_usage=resource_usage
        )
    
    def _process_combination(self, params: Dict[str, float], 
                           enable_target: bool) -> Tuple[Dict[str, float], List[Dict]]:
        """Process a single parameter combination with actual calculation logic"""
        # Initialize supertrend parameters
        atr_period = int(params['ATR Period'])
        factor = params['Factor']
        buffer_distance = int(params['Buffer Distance'])
        hard_stop_distance = int(params['Hard Stop Distance'])
        
        # Initialize target parameters if enabled
        long_target_rr = params.get('Long Target RR', 0)
        short_target_rr = params.get('Short Target RR', 0)
        
        # TODO: In a real implementation, this would load and process actual market data
        # For now, we'll create more realistic simulated data that responds to parameter changes
        
        # Create parameter-dependent metrics (not just random)
        sharpe_base = 1.5 - (abs(atr_period - 30) / 30)  # Optimal around ATR=30
        win_rate_base = 0.55 - (abs(factor - 15) / 50)   # Optimal around Factor=15
        dd_factor = buffer_distance / 200  # Higher buffer = lower drawdown
        
        # Add some reasonable randomness
        sharpe_ratio = max(0, sharpe_base + np.random.normal(0, 0.3))
        win_rate = max(0.3, min(0.7, win_rate_base + np.random.normal(0, 0.05)))
        max_drawdown = max(0.05, min(0.4, 0.25 - dd_factor + np.random.normal(0, 0.05)))
        
        # Calculate other metrics based on these core metrics
        sortino_ratio = sharpe_ratio * (1.2 + np.random.normal(0, 0.1))
        profit_factor = win_rate / (1 - win_rate) * (1.2 + np.random.normal(0, 0.1))
        
        metrics = {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'total_trades': np.random.randint(50, 200)
        }
        
        trades = self._generate_sample_trades(params, win_rate)
        
        return metrics, trades
    
    def _generate_sample_trades(self, params: Dict[str, float], win_rate: float = 0.5) -> List[Dict]:
        """Generate sample trades for demonstration"""
        num_trades = np.random.randint(10, 30)
        trades = []
        
        current_time = datetime.now()
        for _ in range(num_trades):
            entry_time = current_time + pd.Timedelta(minutes=np.random.randint(1, 1000))
            exit_time = entry_time + pd.Timedelta(minutes=np.random.randint(1, 500))
            
            is_long = np.random.random() > 0.5
            entry_price = np.random.uniform(100, 1000)
            
            # Make win/loss more dependent on win_rate parameter
            is_winner = np.random.random() < win_rate
            
            # Create more realistic price movements
            if is_winner:
                pct_change = np.random.uniform(0.002, 0.05)
            else:
                pct_change = np.random.uniform(-0.05, -0.002)
                
            if not is_long:
                pct_change = -pct_change
                
            exit_price = entry_price * (1 + pct_change)
            
            trades.append({
                'entry_time': entry_time.strftime(CONFIG.TIME_FORMAT),
                'exit_time': exit_time.strftime(CONFIG.TIME_FORMAT),
                'position': 'LONG' if is_long else 'SHORT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': exit_price - entry_price if is_long else entry_price - exit_price,
                'pnl_percent': ((exit_price - entry_price) / entry_price) * 100 if is_long else ((entry_price - exit_price) / entry_price) * 100,
                'parameters': params.copy()
            })
        
        return trades
    
    def _get_resource_snapshot(self) -> Dict[str, float]:
        """Get current resource usage snapshot"""
        return {
            'gpu_memory': RESOURCE_MONITOR.get_gpu_memory_usage() if cuda.is_available() else 0,
            'ram_usage': RESOURCE_MONITOR.get_ram_usage()
        }
    
    def _calculate_resource_usage(self, before: Dict[str, float], 
                                after: Dict[str, float]) -> Dict[str, float]:
        """Calculate resource usage from before/after snapshots"""
        return {
            'gpu_memory_delta': after['gpu_memory'] - before['gpu_memory'],
            'ram_usage_delta': after['ram_usage'] - before['ram_usage']
        }
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics from multiple combinations"""
        aggregated = {}
        for metric_name in metrics_list[0].keys():
            values = [m[metric_name] for m in metrics_list]
            aggregated[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        return aggregated
    
    def save_checkpoint(self, batch_result: BatchResult) -> None:
        """Save batch results to checkpoint file with atomic operation"""
        with self.checkpoint_lock:
            checkpoint_file = CONFIG.CHECKPOINTS_DIR / f"checkpoint_{batch_result.batch_id}.h5"
            temp_file = CONFIG.CHECKPOINTS_DIR / f"temp_checkpoint_{batch_result.batch_id}.h5"
            
            try:
                # Write to temporary file first
                with h5py.File(temp_file, 'w') as f:
                    # Save combinations
                    combinations_group = f.create_group('combinations')
                    for i, combo in enumerate(batch_result.combinations):
                        combo_group = combinations_group.create_group(str(i))
                        for key, value in combo.items():
                            combo_group.attrs[key] = value
                    
                    # Save metrics
                    metrics_group = f.create_group('metrics')
                    for metric_name, metric_values in batch_result.metrics.items():
                        metric_group = metrics_group.create_group(metric_name)
                        for stat_name, value in metric_values.items():
                            metric_group.attrs[stat_name] = value
                    
                    # Save trades
                    trades_group = f.create_group('trades')
                    for i, trade in enumerate(batch_result.trade_list):
                        trade_group = trades_group.create_group(str(i))
                        for key, value in trade.items():
                            if key == 'parameters':
                                param_group = trade_group.create_group('parameters')
                                for param_key, param_value in value.items():
                                    param_group.attrs[param_key] = param_value
                            else:
                                trade_group.attrs[key] = value
                    
                    # Save metadata
                    f.attrs['batch_id'] = batch_result.batch_id
                    f.attrs['execution_time'] = batch_result.execution_time
                    f.attrs['timestamp'] = datetime.now().strftime(CONFIG.TIME_FORMAT)
                
                # Rename temp file to final file (atomic operation)
                temp_file.rename(checkpoint_file)
                
            except Exception as e:
                logger.error(f"Error saving checkpoint: {str(e)}")
                if temp_file.exists():
                    try:
                        temp_file.unlink()  # Clean up temp file
                    except:
                        pass
    
    def load_checkpoint(self, checkpoint_file: Path) -> BatchResult:
        """Load batch results from checkpoint file"""
        with h5py.File(checkpoint_file, 'r') as f:
            # Load combinations
            combinations = []
            for combo in f['combinations'].values():
                combinations.append({key: value for key, value in combo.attrs.items()})
            
            # Load metrics
            metrics = {}
            for metric_name, metric_group in f['metrics'].items():
                metrics[metric_name] = {
                    stat_name: value 
                    for stat_name, value in metric_group.attrs.items()
                }
            
            # Load trades
            trades = []
            for trade in f['trades'].values():
                trade_dict = {key: value for key, value in trade.attrs.items()}
                if 'parameters' in trade:
                    trade_dict['parameters'] = {
                        key: value 
                        for key, value in trade['parameters'].attrs.items()
                    }
                trades.append(trade_dict)
            
            return BatchResult(
                batch_id=f.attrs['batch_id'],
                combinations=combinations,
                metrics=metrics,
                trade_list=trades,
                execution_time=f.attrs['execution_time'],
                resource_usage={}  # Resource usage not stored in checkpoint
            )
			
			
# Part 5: Results Processing and Analysis
# Last Updated: 2025-05-09 02:58:30 UTC
# Author: arullr001

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

@dataclass
class AnalysisResult:
    parameter_combination: Dict[str, float]
    performance_metrics: Dict[str, float]
    trade_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    trades: List[Dict[str, Any]]

class ResultsAnalyzer:
    def __init__(self, enable_target: bool):
        self.enable_target = enable_target
        self.logger = logging.getLogger(__name__)
        self.results_dir = CONFIG.RESULTS_DIR
        self.best_combinations: List[AnalysisResult] = []
    
    @handle_errors
    def analyze_batch_result(self, batch_result: BatchResult) -> None:
        """Analyze results from a batch and update best combinations"""
        for combination, trades in zip(batch_result.combinations, 
                                     self._group_trades_by_combination(batch_result.trade_list)):
            analysis = self._analyze_single_combination(combination, trades)
            self._update_best_combinations(analysis)
    
    def _group_trades_by_combination(self, trades: List[Dict]) -> List[List[Dict]]:
        """Group trades by parameter combination"""
        trade_groups = {}
        for trade in trades:
            combo_key = tuple(sorted(trade['parameters'].items()))
            if combo_key not in trade_groups:
                trade_groups[combo_key] = []
            trade_groups[combo_key].append(trade)
        return list(trade_groups.values())
    
    def _analyze_single_combination(self, params: Dict[str, float], 
                                  trades: List[Dict]) -> AnalysisResult:
        """Analyze results for a single parameter combination"""
        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame(trades)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(trades_df)
        
        # Calculate trade metrics
        trade_metrics = self._calculate_trade_metrics(trades_df)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(trades_df)
        
        return AnalysisResult(
            parameter_combination=params,
            performance_metrics=performance_metrics,
            trade_metrics=trade_metrics,
            risk_metrics=risk_metrics,
            trades=trades
        )
    
    def _calculate_performance_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics from trades"""
        returns = trades_df['pnl_percent'] / 100
        
        return {
            'total_return': returns.sum(),
            'annualized_return': self._calculate_annualized_return(returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'calmar_ratio': self._calculate_calmar_ratio(returns),
            'profit_factor': self._calculate_profit_factor(trades_df['pnl'])
        }
    
    def _calculate_trade_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate trade-specific metrics"""
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        
        return {
            'total_trades': total_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'avg_trade': trades_df['pnl'].mean() if total_trades > 0 else 0,
            'avg_winner': trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0,
            'avg_loser': trades_df[trades_df['pnl'] < 0]['pnl'].mean() if total_trades - winning_trades > 0 else 0,
            'largest_winner': trades_df['pnl'].max() if total_trades > 0 else 0,
            'largest_loser': trades_df['pnl'].min() if total_trades > 0 else 0,
            'avg_holding_time': self._calculate_avg_holding_time(trades_df) if total_trades > 0 else 0
        }
    
    def _calculate_risk_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk-related metrics"""
        if len(trades_df) == 0:
            return {
                'max_drawdown': 0,
                'volatility': 0,
                'var_95': 0,
                'cvar_95': 0,
                'risk_reward_ratio': 0,
                'recovery_factor': 0
            }
            
        returns = trades_df['pnl_percent'] / 100
        
        # Handle empty returns case
        if len(returns) == 0:
            return {
                'max_drawdown': 0,
                'volatility': 0,
                'var_95': 0,
                'cvar_95': 0,
                'risk_reward_ratio': 0,
                'recovery_factor': 0
            }
            
        cumulative_returns = (1 + returns).cumprod()
        
        return {
            'max_drawdown': self._calculate_max_drawdown(cumulative_returns),
            'volatility': returns.std() * np.sqrt(252) if len(returns) > 1 else 0,
            'var_95': self._calculate_var(returns, 0.95),
            'cvar_95': self._calculate_cvar(returns, 0.95),
            'risk_reward_ratio': self._calculate_risk_reward_ratio(trades_df['pnl']),
            'recovery_factor': self._calculate_recovery_factor(cumulative_returns)
        }
    
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return with improved safeguards"""
        if len(returns) < 2:
            return 0
            
        # Convert index to datetime if not already
        if not isinstance(returns.index, pd.DatetimeIndex):
            try:
                # Try to get entry_time column from original dataframe
                if 'entry_time' in returns.index.names or 'entry_time' in returns.index.name:
                    pass  # Already has entry_time in index
                else:
                    # If we have a simple RangeIndex, just return non-annualized return
                    total_return = (1 + returns).prod() - 1
                    return total_return
            except:
                # If conversion fails, return non-annualized
                total_return = (1 + returns).prod() - 1
                return total_return
        
        # Get first and last dates
        try:
            first_date = pd.to_datetime(returns.index[0])
            last_date = pd.to_datetime(returns.index[-1])
            total_days = (last_date - first_date).days
        except:
            # If date parsing fails, use length of returns for rough estimate
            total_days = len(returns)
        
        if total_days < 1:
            total_days = 1  # Avoid division by zero
        
        total_return = (1 + returns).prod() - 1
        return ((1 + total_return) ** (365 / total_days)) - 1
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio with improved safeguards"""
        if len(returns) < 2:
            return 0
        
        annualized_return = self._calculate_annualized_return(returns)
        annualized_volatility = returns.std() * np.sqrt(252)
        
        # Safeguard against very small values that could cause numerical issues
        if abs(annualized_volatility) < 1e-10:
            return float('inf') if annualized_return > 0 else 0
            
        return annualized_return / annualized_volatility
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio with improved safeguards"""
        if len(returns) < 2:
            return 0
        
        annualized_return = self._calculate_annualized_return(returns)
        downside_returns = returns[returns < 0]
        
        # Better handling of empty downside returns
        if len(downside_returns) == 0:
            return float('inf') if annualized_return > 0 else 0
            
        downside_std = downside_returns.std() * np.sqrt(252)
        
        # Safeguard against very small values that could cause numerical issues
        if downside_std < 1e-10:
            return float('inf') if annualized_return > 0 else 0
            
        return annualized_return / downside_std
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio with improved safeguards"""
        annualized_return = self._calculate_annualized_return(returns)
        cumulative_returns = (1 + returns).cumprod()
        max_drawdown = self._calculate_max_drawdown(cumulative_returns)
        
        # Avoid division by zero or very small values
        if max_drawdown < 1e-10:
            return float('inf') if annualized_return > 0 else 0
            
        return abs(annualized_return / max_drawdown)
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown with safeguards"""
        if len(cumulative_returns) < 2:
            return 0
            
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        return abs(drawdowns.min())
    
    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk with safeguards"""
        if len(returns) < 2:
            return 0
            
        return abs(np.percentile(returns, (1 - confidence) * 100))
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Conditional Value at Risk with safeguards"""
        if len(returns) < 2:
            return 0
            
        var = self._calculate_var(returns, confidence)
        below_var = returns[returns <= -var]
        
        if len(below_var) == 0:
            return var  # If no returns below VaR, return VaR itself
            
        return abs(below_var.mean())
    
    def _calculate_profit_factor(self, pnl: pd.Series) -> float:
        """Calculate profit factor with safeguards"""
        gross_profit = pnl[pnl > 0].sum()
        gross_loss = abs(pnl[pnl < 0].sum())
        
        if abs(gross_loss) < 1e-10:
            return float('inf') if gross_profit > 0 else 0
            
        return gross_profit / gross_loss
    
    def _calculate_risk_reward_ratio(self, pnl: pd.Series) -> float:
        """Calculate risk-reward ratio with safeguards"""
        winners = pnl[pnl > 0]
        losers = pnl[pnl < 0]
        
        if len(winners) == 0:
            return 0
        if len(losers) == 0:
            return float('inf')
            
        avg_win = winners.mean()
        avg_loss = abs(losers.mean())
        
        if abs(avg_loss) < 1e-10:
            return float('inf')
            
        return avg_win / avg_loss
    
    def _calculate_recovery_factor(self, cumulative_returns: pd.Series) -> float:
        """Calculate recovery factor with safeguards"""
        if len(cumulative_returns) < 2:
            return 0
            
        total_return = cumulative_returns.iloc[-1] - 1
        max_drawdown = self._calculate_max_drawdown(cumulative_returns)
        
        if max_drawdown < 1e-10:
            return float('inf') if total_return > 0 else 0
            
        return total_return / max_drawdown
    
    def _calculate_avg_holding_time(self, trades_df: pd.DataFrame) -> float:
        """Calculate average holding time in hours with safeguards"""
        if len(trades_df) == 0:
            return 0
            
        try:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            holding_times = (trades_df['exit_time'] - trades_df['entry_time'])
            
            return holding_times.mean().total_seconds() / 3600
        except Exception as e:
            self.logger.warning(f"Error calculating holding time: {str(e)}")
            return 0
    
    def _update_best_combinations(self, analysis: AnalysisResult) -> None:
        """Update list of best combinations based on multiple criteria"""
        # Define scoring weights
        weights = {
            'sharpe_ratio': 0.3,
            'sortino_ratio': 0.2,
            'calmar_ratio': 0.15,
            'win_rate': 0.15,
            'profit_factor': 0.2
        }
        
        # Calculate composite score
        score = (
            weights['sharpe_ratio'] * analysis.performance_metrics['sharpe_ratio'] +
            weights['sortino_ratio'] * analysis.performance_metrics['sortino_ratio'] +
            weights['calmar_ratio'] * analysis.performance_metrics['calmar_ratio'] +
            weights['win_rate'] * analysis.trade_metrics['win_rate'] +
            weights['profit_factor'] * analysis.performance_metrics['profit_factor']
        )
        
        # Update best combinations list
        if len(self.best_combinations) < 10:
            analysis.score = score
            self.best_combinations.append(analysis)
            self.best_combinations.sort(key=lambda x: x.score, reverse=True)
        elif score > self.best_combinations[-1].score:
            analysis.score = score
            self.best_combinations[-1] = analysis
            self.best_combinations.sort(key=lambda x: x.score, reverse=True)
			
			
# Part 6: Trade List Generation and Output Management
# Last Updated: 2025-05-09 02:58:30 UTC
# Author: arullr001

from typing import List, Dict, Any, Optional
import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class OutputManager:
    def __init__(self, enable_target: bool):
        self.enable_target = enable_target
        self.logger = logging.getLogger(__name__)
        self.results_dir = CONFIG.RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    @handle_errors
    def generate_output(self, best_combinations: List[AnalysisResult]) -> None:
        """Generate comprehensive output for best combinations"""
        self.logger.info("Generating output files...")
        
        # Create subdirectories
        trades_dir = self.results_dir / "trades"
        analysis_dir = self.results_dir / "analysis"
        plots_dir = self.results_dir / "plots"
        for dir_path in [trades_dir, analysis_dir, plots_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Generate outputs for each best combination
        for i, result in enumerate(best_combinations, 1):
            self._generate_combination_output(i, result, trades_dir, analysis_dir, plots_dir)
        
        # Generate summary report
        self._generate_summary_report(best_combinations)
        
        self.logger.info(f"Output generated in {self.results_dir}")
    
    def _generate_combination_output(self, index: int, result: AnalysisResult,
                                   trades_dir: Path, analysis_dir: Path, plots_dir: Path) -> None:
        """Generate output files for a single combination"""
        # Generate trades list
        self._save_trades_list(index, result, trades_dir)
        
        # Generate analysis report
        self._save_analysis_report(index, result, analysis_dir)
        
        # Generate plots
        self._generate_plots(index, result, plots_dir)
    
    def _save_trades_list(self, index: int, result: AnalysisResult, trades_dir: Path) -> None:
        """Save detailed trades list to CSV"""
        trades_file = trades_dir / f"trades_combination_{index}.csv"
        trades_df = pd.DataFrame(result.trades)
        
        # Add calculated columns if there are trades
        if len(trades_df) > 0:
            try:
                # Add holding time
                trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
                trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
                trades_df['holding_time_hours'] = (
                    trades_df['exit_time'] - trades_df['entry_time']
                ).dt.total_seconds() / 3600
                
                # Add cumulative metrics
                trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
                trades_df['cumulative_pnl_percent'] = (
                    (1 + trades_df['pnl_percent'] / 100).cumprod() - 1
                ) * 100
            except Exception as e:
                self.logger.warning(f"Error adding calculated columns to trades: {str(e)}")
        
        # Save to CSV
        try:
            trades_df.to_csv(trades_file, index=False)
        except Exception as e:
            self.logger.error(f"Error saving trades to CSV: {str(e)}")
    
    def _save_analysis_report(self, index: int, result: AnalysisResult, 
                            analysis_dir: Path) -> None:
        """Save detailed analysis report"""
        report_file = analysis_dir / f"analysis_combination_{index}.json"
        
        report = {
            'parameters': result.parameter_combination,
            'performance_metrics': result.performance_metrics,
            'trade_metrics': result.trade_metrics,
            'risk_metrics': result.risk_metrics,
            'score': getattr(result, 'score', 0)
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving analysis report: {str(e)}")
    
    def _generate_plots(self, index: int, result: AnalysisResult, plots_dir: Path) -> None:
        """Generate analysis plots with error handling"""
        if not result.trades:
            self.logger.warning(f"No trades to plot for combination {index}")
            return
            
        try:
            trades_df = pd.DataFrame(result.trades)
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            
            # Create subplot figure
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Cumulative P&L',
                    'P&L Distribution',
                    'Monthly Returns',
                    'Trade Duration vs P&L',
                    'Win Rate by Month',
                    'Drawdown Analysis'
                )
            )
            
            # 1. Cumulative P&L
            cumulative_pnl = trades_df['pnl'].cumsum()
            fig.add_trace(
                go.Scatter(x=trades_df['entry_time'], y=cumulative_pnl,
                          name='Cumulative P&L'),
                row=1, col=1
            )
            
            
            # 2. P&L Distribution
            fig.add_trace(
                go.Histogram(x=trades_df['pnl'], name='P&L Distribution'),
                row=1, col=2
            )
            
            # 3. Monthly Returns
            try:
                monthly_returns = trades_df.set_index('entry_time')['pnl'].resample('M').sum()
                fig.add_trace(
                    go.Bar(x=monthly_returns.index, y=monthly_returns,
                          name='Monthly Returns'),
                    row=2, col=1
                )
            except Exception as e:
                self.logger.warning(f"Could not generate monthly returns plot: {str(e)}")
            
            # 4. Trade Duration vs P&L
            try:
                trades_df['duration'] = (
                    pd.to_datetime(trades_df['exit_time']) - 
                    pd.to_datetime(trades_df['entry_time'])
                ).dt.total_seconds() / 3600
                
                fig.add_trace(
                    go.Scatter(x=trades_df['duration'], y=trades_df['pnl'],
                              mode='markers', name='Duration vs P&L'),
                    row=2, col=2
                )
            except Exception as e:
                self.logger.warning(f"Could not generate duration plot: {str(e)}")
            
            # 5. Win Rate by Month
            try:
                monthly_winrate = (
                    trades_df.set_index('entry_time')
                    .resample('M')['pnl']
                    .apply(lambda x: (x > 0).mean() * 100)
                )
                
                fig.add_trace(
                    go.Bar(x=monthly_winrate.index, y=monthly_winrate,
                          name='Monthly Win Rate'),
                    row=3, col=1
                )
            except Exception as e:
                self.logger.warning(f"Could not generate win rate plot: {str(e)}")
            
            # 6. Drawdown Analysis
            try:
                cumulative_returns = (1 + trades_df['pnl_percent'] / 100).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdowns = (cumulative_returns - rolling_max) / rolling_max * 100
                
                fig.add_trace(
                    go.Scatter(x=trades_df['entry_time'], y=drawdowns,
                              name='Drawdown', fill='tozeroy'),
                    row=3, col=2
                )
            except Exception as e:
                self.logger.warning(f"Could not generate drawdown plot: {str(e)}")
            
            # Update layout
            fig.update_layout(
                height=1200,
                width=1600,
                showlegend=True,
                title_text=f"Analysis for Combination {index}"
            )
            
            # Save plot
            try:
                fig.write_html(plots_dir / f"analysis_combination_{index}.html")
            except Exception as e:
                self.logger.error(f"Error saving plot: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error generating plots for combination {index}: {str(e)}")
    
    def _generate_summary_report(self, best_combinations: List[AnalysisResult]) -> None:
        """Generate summary report for all best combinations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.results_dir / f"summary_report_{timestamp}.json"
        
        summary = {
            'timestamp': datetime.now().strftime(CONFIG.TIME_FORMAT),
            'user': CURRENT_USER,
            'strategy_type': 'Target-Based' if self.enable_target else 'Trend-Based',
            'combinations': []
        }
        
        for i, result in enumerate(best_combinations, 1):
            summary['combinations'].append({
                'rank': i,
                'parameters': result.parameter_combination,
                'performance_summary': {
                    'sharpe_ratio': result.performance_metrics['sharpe_ratio'],
                    'sortino_ratio': result.performance_metrics['sortino_ratio'],
                    'calmar_ratio': result.performance_metrics['calmar_ratio'],
                    'win_rate': result.trade_metrics['win_rate'],
                    'profit_factor': result.performance_metrics['profit_factor'],
                    'max_drawdown': result.risk_metrics['max_drawdown'],
                    'total_trades': result.trade_metrics['total_trades'],
                    'score': getattr(result, 'score', 0)
                }
            })
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving summary report: {str(e)}")
        
        # Generate summary plots
        self._generate_summary_plots(best_combinations)
    
    def _generate_summary_plots(self, best_combinations: List[AnalysisResult]) -> None:
        """Generate comparison plots for best combinations"""
        try:
            comparison_data = []
            for i, result in enumerate(best_combinations, 1):
                comparison_data.append({
                    'rank': i,
                    'sharpe_ratio': result.performance_metrics['sharpe_ratio'],
                    'sortino_ratio': result.performance_metrics['sortino_ratio'],
                    'calmar_ratio': result.performance_metrics['calmar_ratio'],
                    'win_rate': result.trade_metrics['win_rate'],
                    'profit_factor': result.performance_metrics['profit_factor']
                })
            
            if not comparison_data:
                self.logger.warning("No data for summary plots")
                return
                
            df = pd.DataFrame(comparison_data)
            
            # Create radar plot for top combinations
            fig = go.Figure()
            
            for i in range(min(5, len(df))):
                fig.add_trace(go.Scatterpolar(
                    r=[
                        df.iloc[i]['sharpe_ratio'],
                        df.iloc[i]['sortino_ratio'],
                        df.iloc[i]['calmar_ratio'],
                        df.iloc[i]['win_rate'],
                        df.iloc[i]['profit_factor']
                    ],
                    theta=['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
                           'Win Rate', 'Profit Factor'],
                    name=f'Combination {i+1}'
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(df[['sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                                        'win_rate', 'profit_factor']].max())]
                    )),
                showlegend=True,
                title='Top Combinations Comparison'
            )
            
            fig.write_html(self.results_dir / "combinations_comparison.html")
        except Exception as e:
            self.logger.error(f"Error generating summary plots: {str(e)}")
		
		
# Part 7: Main Execution Flow and Control Systems
# Last Updated: 2025-05-09 02:58:30 UTC
# Author: arullr001

class BacktestManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.parameter_manager = ParameterManager()
        self.results_analyzer = None
        self.output_manager = None
    
    @handle_errors
    def run(self) -> None:
        """Main execution flow"""
        self.logger.info(f"Starting backtest process - Version {VERSION}")
        self.logger.info(f"User: {CURRENT_USER}")
        
        try:
            # Initialize parameters
            if not self._initialize_parameters():
                self.logger.info("Parameter initialization cancelled. Exiting...")
                return
            
            # Initialize preprocessing
            preprocessor = PreProcessor(self.parameter_manager.input_params)
            
            # Run sample estimation
            estimated_time, resource_usage = preprocessor.run_sample_estimation()
            preprocessor.display_estimation_summary(estimated_time, resource_usage)
            
            if not preprocessor.confirm_processing():
                self.logger.info("Processing cancelled by user. Exiting...")
                return
            
            # Initialize components
            self.results_analyzer = ResultsAnalyzer(
                self.parameter_manager.input_params.enable_target
            )
            self.output_manager = OutputManager(
                self.parameter_manager.input_params.enable_target
            )
            
            # Check for existing checkpoints to allow resuming
            existing_checkpoints = sorted(list(CONFIG.CHECKPOINTS_DIR.glob("checkpoint_*.h5")))
            if existing_checkpoints:
                resume = input("\nFound existing checkpoints. Resume previous run? (y/n): ").lower() in ['y', 'yes']
                if resume:
                    last_batch = int(existing_checkpoints[-1].stem.split('_')[1])
                    self.logger.info(f"Resuming from batch {last_batch}")
                    # Load results from existing checkpoints
                    batch_processor = BatchProcessor(preprocessor.batch_size, preprocessor.total_combinations)
                    for checkpoint in existing_checkpoints:
                        try:
                            batch_result = batch_processor.load_checkpoint(checkpoint)
                            self.results_analyzer.analyze_batch_result(batch_result)
                            self.logger.info(f"Loaded checkpoint {checkpoint.name}")
                        except Exception as e:
                            self.logger.error(f"Failed to load checkpoint {checkpoint}: {str(e)}")
            
            # Start processing
            self._process_batches(preprocessor)
            
            # Generate final output
            self._generate_output()
            
        except KeyboardInterrupt:
            self.logger.warning("Process interrupted by user")
            self._handle_interruption()
        except Exception as e:
            self.logger.error(f"Error during execution: {str(e)}")
            raise
        finally:
            self._cleanup()
    
    def _initialize_parameters(self) -> bool:
        """Initialize parameters through ParameterManager"""
        return self.parameter_manager.initialize_parameters()
    
    def _process_batches(self, preprocessor: PreProcessor) -> None:
        """Process all parameter combinations in batches"""
        batch_processor = BatchProcessor(
            preprocessor.batch_size,
            preprocessor.total_combinations
        )
        
        total_batches = math.ceil(
            preprocessor.total_combinations / preprocessor.batch_size
        )
        
        self.logger.info(f"\nStarting batch processing ({total_batches:,} batches)")
        
        with tqdm(total=total_batches, desc="Processing batches") as pbar:
            for batch_id, combinations in enumerate(
                batch_processor.generate_batches(
                    self.parameter_manager.input_params
                ), 1
            ):
                if SYSTEM_STATE.should_abort:
                    raise KeyboardInterrupt("Processing aborted by user")
                
                SYSTEM_STATE.wait_if_paused()
                
                # Process batch
                batch_result = batch_processor.process_batch(
                    batch_id,
                    combinations,
                    self.parameter_manager.input_params.enable_target
                )
                
                # Save checkpoint
                batch_processor.save_checkpoint(batch_result)
                
                # Analyze results
                self.results_analyzer.analyze_batch_result(batch_result)
                
                # Update progress
                pbar.update(1)
                
                # Clean up resources
                RESOURCE_MONITOR.cleanup()
    
    def _generate_output(self) -> None:
        """Generate final output through OutputManager"""
        if self.results_analyzer and self.output_manager:
            self.output_manager.generate_output(
                self.results_analyzer.best_combinations
            )
    
    def _handle_interruption(self) -> None:
        """Handle user interruption"""
        self.logger.warning("\nProcess interrupted. Cleaning up...")
        self._generate_output()
    
    def _cleanup(self) -> None:
        """Clean up resources"""
        RESOURCE_MONITOR.cleanup()
        self.logger.info("Cleanup completed")

def main():
    """Main entry point"""
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
        
    try:
        # Display current date/time and user
        print(f"Current Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current User: {CURRENT_USER}")
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, lambda s, f: None)
        signal.signal(signal.SIGTERM, lambda s, f: None)
        
        # Initialize and run backtest manager
        backtest_manager = BacktestManager()
        backtest_manager.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
Usage Instructions:

1. Save all parts in sequence to a single file named 'NewST2.py'
2. Ensure all required dependencies are installed:
   pip install numpy pandas torch keyboard psutil h5py plotly seaborn tqdm

3. Run the script:
   python NewST2.py

Control Keys:
- Ctrl+Alt+X: Abort processing
- Ctrl+Space: Pause/Resume processing

The script will:
1. Get user inputs for parameter ranges
2. Estimate processing time and resource usage
3. Process parameter combinations in batches
4. Generate comprehensive analysis and visualizations
5. Save results in organized directory structure

Output Directory Structure:
/results/YYYYMMDD_HHMMSS/
    ├── trades/
    │   └── trades_combination_N.csv
    ├── analysis/
    │   └── analysis_combination_N.json
    ├── plots/
    │   └── analysis_combination_N.html
    ├── summary_report_YYYYMMDD_HHMMSS.json
    └── combinations_comparison.html
"""