#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supertrend Backtester v14 - Optimized GPU-accelerated backtesting with GUI
Created on: 2025-06-24
Author: arullr001
"""

# Standard libraries
import os
import sys
import json
import glob
import time
import logging
import traceback
import warnings
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional
from itertools import product, combinations
from functools import partial, lru_cache
from PyQt5.QtWidgets import QStyleFactory
import threading
import queue
import uuid
import multiprocessing
import concurrent.futures
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import random

from collections import Counter
from copy import deepcopy
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import itertools  # For parameter combinations in optimization

# Modern data processing libraries
# Polars - Fast DataFrame library (replacement for pandas)
HAS_POLARS = False
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    pass  # Polars not available

# Vaex - Out-of-core DataFrames for large datasets
HAS_VAEX = False
try:
    import vaex
    HAS_VAEX = True
except ImportError:
    pass  # Vaex not available

# PyTorch - For GPU-accelerated tensor operations
HAS_TORCH = False
try:
    import torch
    HAS_TORCH = cuda_available = torch.cuda.is_available()
    HAS_TORCH = True
except ImportError:
    pass  # PyTorch not available

# Datatable - Fast data manipulation library
HAS_DATATABLE = False
try:
    import datatable as dt
    HAS_DATATABLE = True
except ImportError:
    pass  # datatable not available

# Data processing
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# GPU Computing 
import numba
from numba import cuda, jit, vectorize, prange
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

# Bayesian Optimization
try:
    from skopt import Optimizer
    from skopt.space import Real, Integer
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False

# Dask for distributed computing
try:
    import dask
    import dask.dataframe as dd
    import dask.bag as db
    from dask.diagnostics import ProgressBar
    HAS_DASK = True
except ImportError:
    HAS_DASK = False


# New GPU Implementation
HAS_CUDF = False
try:
    import cudf
    HAS_CUDF = True
except ImportError:
    pass  # cuDF not available

HAS_NUMBA = False
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    pass  # Numba not available

HAS_CUDA = False
try:
    from numba import cuda
    test_device = cuda.get_current_device()
    HAS_CUDA = True
    del test_device  # Clean up the variable
except (ImportError, cuda.CudaSupportError):
    pass  # CUDA not available or not working

# Optional visualization/analysis libraries
HAS_PLOTLY = False
try:
    import plotly
    import plotly.graph_objects
    HAS_PLOTLY = True
except ImportError:
    pass  # Plotly not available

HAS_SKLEARN = False
try:
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    pass  # scikit-learn not available

HAS_SKOPT = False
try:
    import skopt
    HAS_SKOPT = True
except ImportError:
    pass  # scikit-optimize not available

HAS_DEAP = False
try:
    import deap
    HAS_DEAP = True
except ImportError:
    pass  # DEAP genetic algorithm library not available    

# Memory monitoring
import psutil

# Visualization
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns

# GUI Framework - PyQt5
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QProgressBar, QTableWidget, QTableWidgetItem,
    QSplitter, QFrame, QGroupBox, QFormLayout, QSizePolicy, QTextEdit, QMenu,
    QAction, QToolBar, QStatusBar, QDockWidget, QListWidget, QScrollArea,
    QStyleFactory, QGridLayout, QInputDialog, QProgressDialog, QHeaderView
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QSettings, QSize, QRect, QPoint, QUrl
)
from PyQt5.QtGui import (
    QIcon, QPixmap, QColor, QFont, QPalette, QDesktopServices, QTextCursor
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Suppress warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Application metadata
APP_NAME = "SuperTrend Backtester"
APP_VERSION = "14.0"
APP_DATE = "2025-06-24"
CURRENT_USER = "arullr001"
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 800

# Performance settings
MAX_RAM_USAGE_PERCENT = 90
GPU_MEMORY_SAFETY_FACTOR = 0.8  # Use only 80% of available GPU memory
THREADS_PER_BLOCK = 256
CPU_THREADS = max(1, os.cpu_count() - 1)  # Leave one CPU core free
BATCH_SIZE = 1000

# File and directory constants
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO, 
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# Default parameters
DEFAULT_ATR_LENGTH_RANGE = (8, 24, 1)     # start, end, step
DEFAULT_FACTOR_RANGE = (0.3, 1.5, 0.1)    # start, end, step
DEFAULT_BUFFER_RANGE = (0.1, 0.5, 0.05)   # start, end, step
DEFAULT_STOP_RANGE = (10, 50, 5)          # start, end, step

# Colors for GUI
COLORS = {
    'primary': '#2C3E50',           # Dark blue-gray
    'secondary': '#3498DB',         # Bright blue
    'accent': '#E74C3C',            # Red
    'background': '#ECF0F1',        # Light gray
    'text': '#2C3E50',              # Dark blue-gray
    'success': '#2ECC71',           # Green
    'warning': '#F39C12',           # Orange
    'chart_background': '#FFFFFF',  # White
    'profit': '#18BC9C',            # Teal
    'loss': '#E74C3C',              # Red
}

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_current_time() -> str:
    """Return current UTC time as formatted string"""
    return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

def format_time_delta(seconds: float) -> str:
    """Format seconds into hours:minutes:seconds"""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def generate_unique_id() -> str:
    """Generate a unique ID for runs and sessions"""
    return str(uuid.uuid4())[:8]

def generate_run_directory() -> str:
    """Generate a unique directory name for this run"""
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    script_name = os.path.basename(__file__).split('.')[0]  # Get filename without extension
    uid = generate_unique_id()
    return f"{script_name}_{timestamp}_{uid}"

def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format"""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def is_valid_float(val: str) -> bool:
    """Check if string can be converted to float"""
    try:
        float(val)
        return True
    except ValueError:
        return False

def is_valid_int(val: str) -> bool:
    """Check if string can be converted to integer"""
    try:
        int(val)
        return True
    except ValueError:
        return False

def check_gpu_availability() -> bool:
    """Check if a CUDA-compatible GPU is available"""
    try:
        if cuda.is_available():
            cuda.select_device(0)
            device = cuda.get_current_device()
            return True
        return False
    except Exception:
        return False

def get_gpu_info() -> Dict[str, Any]:
    """Get information about available GPU"""
    gpu_info = {
        'available': False,
        'name': 'None',
        'memory_total': 0,
        'memory_free': 0,
        'compute_capability': None
    }
    
    try:
        if cuda.is_available():
            cuda.select_device(0)
            device = cuda.get_current_device()
            name = device.name
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            ctx = cuda.current_context()
            free_mem = ctx.get_memory_info()[0]
            total_mem = device.total_memory
            cc_major, cc_minor = device.compute_capability
            
            gpu_info.update({
                'available': True,
                'name': name,
                'memory_total': total_mem,
                'memory_free': free_mem,
                'compute_capability': f"{cc_major}.{cc_minor}"
            })
    except Exception:
        pass
    
    return gpu_info

def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    memory = psutil.virtual_memory()
    return {
        'cpu_count': os.cpu_count(),
        'ram_total': memory.total,
        'ram_available': memory.available,
        'platform': sys.platform,
        'python_version': sys.version.split()[0],
        'gpu_info': get_gpu_info()
    }

def format_number(num: float, precision: int = 2) -> str:
    """Format a number with specified precision and thousands separator"""
    return f"{num:,.{precision}f}"

def create_arange(start: float, end: float, step: float) -> np.ndarray:
    """Create a numpy array with values from start to end with step increment"""
    # Add small epsilon for floating point precision issues
    eps = 1e-10
    return np.arange(start, end + step/2 + eps, step)

def get_parameter_ranges(atr_range: Tuple[float, float, float],
                         factor_range: Tuple[float, float, float],
                         buffer_range: Tuple[float, float, float],
                         stop_range: Tuple[float, float, float]) -> Dict[str, List]:
    """Generate parameter ranges from tuples of (start, end, step)"""
    atr_lengths = list(range(int(atr_range[0]), int(atr_range[1]) + 1, int(atr_range[2])))
    factors = [round(x, 2) for x in create_arange(factor_range[0], factor_range[1], factor_range[2])]
    buffers = [round(x, 2) for x in create_arange(buffer_range[0], buffer_range[1], buffer_range[2])]
    stops = list(range(int(stop_range[0]), int(stop_range[1]) + 1, int(stop_range[2])))
    
    return {
        'atr_lengths': atr_lengths,
        'factors': factors,
        'buffers': buffers,
        'stops': stops,
        'total_combinations': len(atr_lengths) * len(factors) * len(buffers) * len(stops)
    }

def cleanup_gpu_memory():
    """Clean up GPU memory to prevent leaks"""
    try:
        if cuda.is_available():
            import gc
            gc.collect()
            cuda.current_context().deallocations.clear()
    except Exception:
        pass

def set_plot_style():
    """Set consistent style for matplotlib plots"""
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

# GUI Utility Functions
def show_error_message(parent, title, message):
    """Show error message box"""
    QMessageBox.critical(parent, title, message)

def show_info_message(parent, title, message):
    """Show info message box"""
    QMessageBox.information(parent, title, message)

def show_question_message(parent, title, message):
    """Show question message box with Yes/No buttons"""
    return QMessageBox.question(parent, title, message, 
                              QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes

def create_separator(orientation=Qt.Horizontal):
    """Create a visual separator line (horizontal or vertical)"""
    line = QFrame()
    line.setFrameShape(QFrame.HLine if orientation == Qt.Horizontal else QFrame.VLine)
    line.setFrameShadow(QFrame.Sunken)
    return line

def style_widget(widget, background_color=None, text_color=None, border=None):
    """Apply custom styling to a widget"""
    style_sheet = ""
    if background_color:
        style_sheet += f"background-color: {background_color};"
    if text_color:
        style_sheet += f"color: {text_color};"
    if border:
        style_sheet += f"border: {border};"
    
    if style_sheet:
        widget.setStyleSheet(style_sheet)
		
# ==============================================================================
# LOGGING FRAMEWORK
# ==============================================================================

class LogManager:
    """Advanced logging framework with support for console, file, and GUI output"""
    
    def __init__(self, base_dir: str, app_name: str = APP_NAME):
        self.base_dir = base_dir
        self.app_name = app_name
        self.log_dir = os.path.join(base_dir, "logs")
        self.debug_dir = os.path.join(base_dir, "debug")
        self.main_log_file = os.path.join(self.log_dir, "main.log")
        self.debug_log_file = os.path.join(self.debug_dir, "debug.log")
        self.error_log_file = os.path.join(self.log_dir, "errors.log")
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Configure loggers
        self._setup_main_logger()
        self._setup_error_logger()
        self._setup_debug_logger()
        
        # Store logs in memory for GUI access
        self.log_queue = queue.Queue(maxsize=1000)
        self.debug_entries = []
        
        # Log initial information
        self.info(f"=== {self.app_name} v{APP_VERSION} ===")
        self.info(f"Session started at: {get_current_time()}")
        self.info(f"User: {CURRENT_USER}")
        
        # Log system information
        system_info = get_system_info()
        self.debug("System Information:")
        self.debug(f"- CPU: {system_info['cpu_count']} cores")
        self.debug(f"- RAM: {human_readable_size(system_info['ram_total'])}")
        self.debug(f"- Platform: {system_info['platform']}")
        self.debug(f"- Python: {system_info['python_version']}")
        
        # Log GPU information
        gpu_info = system_info['gpu_info']
        if gpu_info['available']:
            self.info(f"GPU Available: {gpu_info['name']}")
            self.debug(f"- GPU Memory: {human_readable_size(gpu_info['memory_total'])}")
            self.debug(f"- Free Memory: {human_readable_size(gpu_info['memory_free'])}")
            self.debug(f"- Compute Capability: {gpu_info['compute_capability']}")
        else:
            self.warning("No CUDA-compatible GPU detected. Using CPU only.")
    
    def _setup_main_logger(self):
        """Set up the main logger"""
        self.main_logger = logging.getLogger('main')
        self.main_logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.main_log_file)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_format)
        
        # Add handlers
        self.main_logger.addHandler(file_handler)
        self.main_logger.addHandler(console_handler)
    
    def _setup_error_logger(self):
        """Set up the error logger"""
        self.error_logger = logging.getLogger('errors')
        self.error_logger.setLevel(logging.ERROR)
        
        # File handler for errors
        error_handler = logging.FileHandler(self.error_log_file)
        error_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n')
        error_handler.setFormatter(error_format)
        
        # Add handler
        self.error_logger.addHandler(error_handler)
    
    def _setup_debug_logger(self):
        """Set up the debug logger"""
        self.debug_logger = logging.getLogger('debug')
        self.debug_logger.setLevel(logging.DEBUG)
        
        # File handler for debug
        debug_handler = logging.FileHandler(self.debug_log_file)
        debug_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - [%(pathname)s:%(lineno)d]')
        debug_handler.setFormatter(debug_format)
        
        # Add handler
        self.debug_logger.addHandler(debug_handler)
    
    def setup_logging(self):
        """Set up and configure the logging system"""
        # Create base directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Configure root logger (for third-party libraries)
        logging.basicConfig(
            level=logging.WARNING,  # Only show warnings and above from third-party libs
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, "system.log")),
                logging.StreamHandler()
            ]
        )
        
        # Call the setup methods for specialized loggers
        self._setup_main_logger()
        self._setup_error_logger()
        self._setup_debug_logger()
        
        self.debug("Logging system initialized")
    
    def _add_to_queue(self, level: str, message: str):
        """Add log entry to queue for GUI access"""
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        entry = {
            'time': timestamp,
            'level': level,
            'message': message
        }
        
        # Add to queue, removing oldest if full
        if self.log_queue.full():
            try:
                self.log_queue.get_nowait()
            except queue.Empty:
                pass
        
        try:
            self.log_queue.put_nowait(entry)
        except queue.Full:
            pass
    
    def info(self, message: str):
        """Log info message"""
        self.main_logger.info(message)
        self._add_to_queue('INFO', message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.debug_logger.debug(message)
        self._add_to_queue('DEBUG', message)
        
        # Store debug entries (limited to most recent 1000)
        self.debug_entries.append({
            'time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'message': message
        })
        if len(self.debug_entries) > 1000:
            self.debug_entries.pop(0)
    
    def warning(self, message: str):
        """Log warning message"""
        self.main_logger.warning(message)
        self._add_to_queue('WARNING', message)
    
    def error(self, message: str, exc_info=None):
        """Log error message"""
        if exc_info:
            self.error_logger.error(message, exc_info=exc_info)
            # Add traceback to message for GUI display
            tb = traceback.format_exc()
            self.main_logger.error(f"{message}\n{tb}")
            self._add_to_queue('ERROR', f"{message}\n{tb}")
        else:
            self.error_logger.error(message)
            self.main_logger.error(message)
            self._add_to_queue('ERROR', message)
    
    def critical(self, message: str, exc_info=None):
        """Log critical message"""
        if exc_info:
            self.error_logger.critical(message, exc_info=exc_info)
            # Add traceback to message for GUI display
            tb = traceback.format_exc()
            self.main_logger.critical(f"{message}\n{tb}")
            self._add_to_queue('CRITICAL', f"{message}\n{tb}")
        else:
            self.error_logger.critical(message)
            self.main_logger.critical(message)
            self._add_to_queue('CRITICAL', message)
    
    def get_log_entries(self, max_entries=100):
        """Get recent log entries for GUI display"""
        entries = []
        temp_queue = queue.Queue()
        
        # Transfer items to a temporary queue
        while not self.log_queue.empty() and len(entries) < max_entries:
            try:
                item = self.log_queue.get_nowait()
                entries.append(item)
                temp_queue.put(item)
            except queue.Empty:
                break
        
        # Return items to the original queue
        while not temp_queue.empty():
            try:
                self.log_queue.put(temp_queue.get_nowait())
            except queue.Full:
                break
        
        return list(reversed(entries))  # Most recent first
    
    def save_debug_report(self, output_file=None):
        """Save comprehensive debug information to a file"""
        if output_file is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(self.debug_dir, f"debug_report_{timestamp}.txt")
        
        with open(output_file, 'w') as f:
            # Write header
            f.write(f"=== {self.app_name} v{APP_VERSION} Debug Report ===\n")
            f.write(f"Generated: {get_current_time()}\n")
            f.write(f"User: {CURRENT_USER}\n\n")
            
            # System information
            f.write("System Information:\n")
            f.write("-" * 50 + "\n")
            system_info = get_system_info()
            f.write(f"CPU: {system_info['cpu_count']} cores\n")
            f.write(f"RAM: {human_readable_size(system_info['ram_total'])}\n")
            f.write(f"Available RAM: {human_readable_size(system_info['ram_available'])}\n")
            f.write(f"Platform: {system_info['platform']}\n")
            f.write(f"Python Version: {system_info['python_version']}\n")
            
            # GPU information
            f.write("\nGPU Information:\n")
            f.write("-" * 50 + "\n")
            gpu_info = system_info['gpu_info']
            if gpu_info['available']:
                f.write(f"GPU: {gpu_info['name']}\n")
                f.write(f"Memory: {human_readable_size(gpu_info['memory_total'])}\n")
                f.write(f"Free Memory: {human_readable_size(gpu_info['memory_free'])}\n")
                f.write(f"Compute Capability: {gpu_info['compute_capability']}\n")
            else:
                f.write("No CUDA-compatible GPU detected\n")
            
            # Library versions
            f.write("\nLibrary Versions:\n")
            f.write("-" * 50 + "\n")
            f.write(f"NumPy: {np.__version__}\n")
            f.write(f"Pandas: {pd.__version__}\n")
            f.write(f"Matplotlib: {matplotlib.__version__}\n")
            f.write(f"Numba: {numba.__version__}\n")
            if HAS_CUPY:
                f.write(f"CuPy: {cp.__version__}\n")
            else:
                f.write("CuPy: Not installed\n")
            if HAS_DASK:
                f.write(f"Dask: {dask.__version__}\n")
            else:
                f.write("Dask: Not installed\n")
            
            # Debug log entries
            f.write("\nDebug Log Entries:\n")
            f.write("-" * 50 + "\n")
            for entry in self.debug_entries:
                f.write(f"{entry['time']} - {entry['message']}\n")
        
        return output_file
 
    def shutdown(self):
        """Clean up logging resources"""
        logging.shutdown()

# ==============================================================================
# DATA MANAGEMENT
# ==============================================================================

class DataLoader:
    """Load and prepare OHLC data from various file formats"""
    
    def __init__(self, log_manager: LogManager = None):
        self.log = log_manager
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.parquet', '.feather']
    
    def _log(self, level: str, message: str):
        """Log message if log_manager is available"""
        if self.log:
            if level == 'info':
                self.log.info(message)
            elif level == 'debug':
                self.log.debug(message)
            elif level == 'warning':
                self.log.warning(message)
            elif level == 'error':
                self.log.error(message)
    
    def get_supported_formats(self):
        """Return list of supported file formats"""
        return self.supported_formats
    
    def load_file(self, file_path: str) -> pd.DataFrame:
        """Load data file based on extension"""
        self._log('info', f"Loading data from: {os.path.basename(file_path)}")
        
        if not os.path.exists(file_path):
            error_msg = f"File does not exist: {file_path}"
            self._log('error', error_msg)
            raise FileNotFoundError(error_msg)
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                return self._load_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                return self._load_excel(file_path)
            elif file_ext == '.parquet':
                return self._load_parquet(file_path)
            elif file_ext == '.feather':
                return self._load_feather(file_path)
            else:
                error_msg = f"Unsupported file format: {file_ext}"
                self._log('error', error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            self._log('error', f"Error loading file: {str(e)}")
            raise
    
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load and process CSV file"""
        # Try to detect the separator and read a sample
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        if ',' in first_line:
            sep = ','
        elif ';' in first_line:
            sep = ';'
        elif '\t' in first_line:
            sep = '\t'
        else:
            sep = ','  # Default
        
        # Read a small sample to examine columns
        sample = pd.read_csv(file_path, sep=sep, nrows=5)
        self._log('debug', f"CSV columns detected: {list(sample.columns)}")
        
        # Load full file
        df = pd.read_csv(file_path, sep=sep)
        
        # Process columns
        return self._process_dataframe(df)
    
    def _load_excel(self, file_path: str) -> pd.DataFrame:
        """Load and process Excel file"""
        # Read a small sample to examine sheet structure
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        self._log('debug', f"Excel sheets detected: {sheet_names}")
        
        # Read first sheet or prompt user to choose if multiple sheets
        # For now, we'll just take the first sheet
        sheet_name = sheet_names[0]
        self._log('debug', f"Loading sheet: {sheet_name}")
        
        # Load the data
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Process columns
        return self._process_dataframe(df)
    
    def _load_parquet(self, file_path: str) -> pd.DataFrame:
        """Load and process Parquet file"""
        df = pd.read_parquet(file_path)
        return self._process_dataframe(df)
    
    def _load_feather(self, file_path: str) -> pd.DataFrame:
        """Load and process Feather file"""
        df = pd.read_feather(file_path)
        return self._process_dataframe(df)
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process loaded DataFrame to standardize column names and format"""
        self._log('debug', f"Processing DataFrame with shape: {df.shape}")
    
        # Map columns to standard names
        column_mapping = {}
        date_col = None
        time_col = None
    
        # Identify columns based on common naming patterns
        for col in df.columns:
            col_lower = col.lower()
        
            # OHLC columns
            if col in ['o', 'O', 'open'] or 'open' in col_lower:
                column_mapping[col] = 'open'
            elif col in ['h', 'H', 'high'] or 'high' in col_lower:
                column_mapping[col] = 'high'
            elif col in ['l', 'L', 'low'] or 'low' in col_lower:
                column_mapping[col] = 'low'
            elif col in ['c', 'C', 'close'] or 'close' in col_lower:
                column_mapping[col] = 'close'
            elif col in ['v', 'V', 'vol', 'volume'] or 'volume' in col_lower:
                column_mapping[col] = 'volume'
        
            # Date/time columns
            elif 'date' in col_lower and 'datetime' not in col_lower:
                date_col = col
            elif 'time' in col_lower and 'datetime' not in col_lower:
                time_col = col
            elif any(x in col_lower for x in ['datetime', 'timestamp']):
                column_mapping[col] = 'datetime'
    
        self._log('debug', f"Column mapping: {column_mapping}")
        self._log('debug', f"Date column: {date_col}, Time column: {time_col}")
    
        # Apply column mapping
        df = df.rename(columns=column_mapping)
    
        # Handle datetime creation if needed
        if 'datetime' not in df.columns:
            if date_col and time_col:
                self._log('debug', f"Creating datetime from {date_col} and {time_col}")
                try:
                    df['datetime'] = pd.to_datetime(df[date_col] + ' ' + df[time_col], errors='coerce')
                except Exception as e:
                    self._log('warning', f"Error combining date and time: {str(e)}")
                    try:
                        # Alternative approach
                        df['datetime'] = pd.to_datetime(df[date_col]) + pd.to_timedelta(df[time_col])
                    except Exception as e2:
                        self._log('error', f"Failed to create datetime: {str(e2)}")
                        raise ValueError(f"Could not create datetime from columns: {date_col} and {time_col}")
            elif date_col:
                self._log('debug', f"Creating datetime from {date_col}")
                df['datetime'] = pd.to_datetime(df[date_col], errors='coerce')
    
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'datetime']
        missing_cols = [col for col in required_cols if col not in df.columns]
    
        if missing_cols:
            self._log('error', f"Missing required columns: {missing_cols}")
            raise ValueError(f"Could not find all required columns. Missing: {missing_cols}")
    
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
        # Convert volume if available
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
    
        # Remove rows with NaN in critical columns
        df.dropna(subset=numeric_cols, inplace=True)
    
        self._log('info', f"Successfully processed data with shape: {df.shape}")
        self._log('debug', f"Date range: {df.index.min()} to {df.index.max()}")
    
        return df

class DataAnalyzer:
    """Analyze OHLC data to detect characteristics and generate insights"""
    
    def __init__(self, log_manager: LogManager = None):
        self.log = log_manager
    
    def _log(self, level: str, message: str):
        """Log message if log_manager is available"""
        if self.log:
            if level == 'info':
                self.log.info(message)
            elif level == 'debug':
                self.log.debug(message)
            elif level == 'warning':
                self.log.warning(message)
            elif level == 'error':
                self.log.error(message)
    
    def analyze_ohlc_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze OHLC data and return comprehensive insights"""
        self._log('info', "Analyzing OHLC data...")
        
        analysis = {}
        
        # Basic information
        analysis['shape'] = df.shape
        analysis['date_range'] = {
            'start': df.index.min().strftime('%Y-%m-%d %H:%M:%S'),
            'end': df.index.max().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Calculate total days and trading days
        try:
            total_days = (df.index.max() - df.index.min()).days + 1
            analysis['total_calendar_days'] = total_days
            
            # Count trading days
            if hasattr(df.index, 'date'):
                date_list = [d.date() for d in df.index]
                analysis['trading_days'] = len(set(date_list))
            else:
                # Alternative approach for numpy arrays
                date_strings = [d.split(' ')[0] for d in df.index.astype(str)]
                analysis['trading_days'] = len(set(date_strings))
        except Exception as e:
            self._log('warning', f"Could not calculate trading days: {str(e)}")
            analysis['trading_days'] = "Unknown"
        
        # Detect timeframe
        analysis['timeframe'] = self._detect_timeframe(df)
        analysis['timeframe_confidence'] = self._calculate_timeframe_confidence(df)
        
        # Basic statistics
        analysis['candle_count'] = len(df)
        analysis['price_statistics'] = {
            'min': df['low'].min(),
            'max': df['high'].max(),
            'open_first': df['open'].iloc[0],
            'close_last': df['close'].iloc[-1],
            'price_change': df['close'].iloc[-1] - df['open'].iloc[0],
            'price_change_pct': (df['close'].iloc[-1] / df['open'].iloc[0] - 1) * 100
        }
        
        # Volatility metrics
        analysis['volatility'] = {
            'avg_true_range': self._calculate_atr(df, 14),
            'avg_daily_range_pct': ((df['high'] - df['low']) / df['close']).mean() * 100,
            'std_dev_daily_returns': (df['close'].pct_change().std() * 100)
        }
        
        # Volume analysis if available
        if 'volume' in df.columns:
            analysis['volume'] = {
                'total': df['volume'].sum(),
                'daily_avg': df['volume'].mean(),
                'max': df['volume'].max(),
                'min': df['volume'].min()
            }
        
        # Gap analysis
        analysis['gaps'] = self._detect_gaps(df)
        
        # Trend analysis
        analysis['trend'] = self._analyze_trend(df)
        
        # Data quality checks
        analysis['data_quality'] = {
            'missing_values': df.isna().sum().sum(),
            'zero_volume_days': (df['volume'] == 0).sum() if 'volume' in df.columns else 'N/A',
            'has_gaps': analysis['gaps']['has_gaps'],
            'unusual_values': self._check_unusual_values(df)
        }
        
        # Parameter recommendations based on data characteristics
        analysis['parameter_recommendations'] = self._generate_parameter_recommendations(df)
        
        self._log('info', "Data analysis complete")
        return analysis
    
    def _detect_timeframe(self, df: pd.DataFrame) -> str:
        """Detect the timeframe of the data"""
        if len(df) <= 1:
            return "Unknown (insufficient data points)"
        
        # Calculate time differences between consecutive rows
        time_diffs = []
        for i in range(min(100, len(df) - 1)):  # Sample up to 100 differences
            diff = (df.index[i+1] - df.index[i]).total_seconds()
            time_diffs.append(diff)
        
        # Find most common difference
        most_common_diff = max(set(time_diffs), key=time_diffs.count)
        
        # Convert to human-readable format
        timeframe_map = {
            60: "1 minute",
            300: "5 minutes",
            900: "15 minutes",
            1800: "30 minutes",
            3600: "1 hour",
            7200: "2 hours",
            14400: "4 hours",
            21600: "6 hours",
            43200: "12 hours",
            86400: "1 day",
            604800: "1 week"
        }
        
        return timeframe_map.get(most_common_diff, f"{most_common_diff} seconds")
    
    def _calculate_timeframe_confidence(self, df: pd.DataFrame) -> float:
        """Calculate confidence in the detected timeframe"""
        if len(df) <= 1:
            return 0.0
        
        # Calculate all time differences
        time_diffs = [(df.index[i+1] - df.index[i]).total_seconds() for i in range(min(1000, len(df) - 1))]
        
        # Count frequency of most common difference
        most_common = max(set(time_diffs), key=time_diffs.count)
        matches = time_diffs.count(most_common)
        
        # Return confidence as percentage
        return (matches / len(time_diffs)) * 100
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.zeros(len(df))
        tr[0] = high[0] - low[0]  # First TR is simply high - low
        
        # Calculate TR values
        for i in range(1, len(df)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr[i] = max(tr1, tr2, tr3)
        
        # Calculate ATR using simple moving average
        atr = np.mean(tr[-period:])
        return atr
    
    def _detect_gaps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect gaps in the data"""
        result = {'has_gaps': False, 'gap_count': 0, 'largest_gap_seconds': 0}
        
        if len(df) <= 1:
            return result
        
        # Calculate timeframe
        timeframe_seconds = (df.index[1] - df.index[0]).total_seconds()
        if timeframe_seconds == 0:
            return result
        
        # Check for gaps
        gaps = []
        for i in range(1, len(df)):
            gap_seconds = (df.index[i] - df.index[i-1]).total_seconds()
            if gap_seconds > (timeframe_seconds * 1.5):  # Gap if > 1.5x normal interval
                gaps.append({
                    'start': df.index[i-1].strftime('%Y-%m-%d %H:%M:%S'),
                    'end': df.index[i].strftime('%Y-%m-%d %H:%M:%S'),
                    'duration_seconds': gap_seconds
                })
        
        if gaps:
            result['has_gaps'] = True
            result['gap_count'] = len(gaps)
            result['largest_gap_seconds'] = max(g['duration_seconds'] for g in gaps)
            result['largest_gap_hours'] = result['largest_gap_seconds'] / 3600
            result['gaps'] = gaps[:10]  # Return at most 10 gaps
        
        return result
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend characteristics"""
        # Calculate simple moving averages
        try:
            df_copy = df.copy()
            df_copy['sma20'] = df_copy['close'].rolling(window=20).mean()
            df_copy['sma50'] = df_copy['close'].rolling(window=50).mean()
            
            # Skip first 50 candles that have NaN moving averages
            df_valid = df_copy.iloc[50:].copy()
            
            # Determine trend direction
            last_sma20 = df_valid['sma20'].iloc[-1]
            last_sma50 = df_valid['sma50'].iloc[-1]
            
            if last_sma20 > last_sma50:
                primary_trend = "Bullish"
            elif last_sma20 < last_sma50:
                primary_trend = "Bearish"
            else:
                primary_trend = "Neutral"
            
            # Calculate trend strength using ADX-like measure
            df_valid['up_move'] = df_valid['high'] - df_valid['high'].shift(1)
            df_valid['down_move'] = df_valid['low'].shift(1) - df_valid['low']
            
            df_valid['plus_dm'] = np.where(
                (df_valid['up_move'] > df_valid['down_move']) & (df_valid['up_move'] > 0),
                df_valid['up_move'],
                0
            )
            df_valid['minus_dm'] = np.where(
                (df_valid['down_move'] > df_valid['up_move']) & (df_valid['down_move'] > 0),
                df_valid['down_move'],
                0
            )
            
            # Smooth directional movement
            window = 14
            df_valid['plus_di'] = 100 * (df_valid['plus_dm'].rolling(window=window).mean() / 
                                        self._calculate_atr(df_valid, window))
            df_valid['minus_di'] = 100 * (df_valid['minus_dm'].rolling(window=window).mean() / 
                                         self._calculate_atr(df_valid, window))
            
            # Calculate directional movement index
            df_valid['dx'] = 100 * abs(df_valid['plus_di'] - df_valid['minus_di']) / (df_valid['plus_di'] + df_valid['minus_di'])
            
            # Calculate ADX
            df_valid['adx'] = df_valid['dx'].rolling(window=window).mean()
            
            last_adx = df_valid['adx'].iloc[-1]
            
            # Interpret trend strength
            if last_adx < 20:
                trend_strength = "Weak"
            elif last_adx < 40:
                trend_strength = "Moderate"
            elif last_adx < 60:
                trend_strength = "Strong"
            else:
                trend_strength = "Very Strong"
            
            # Calculate price volatility
            volatility = df_valid['close'].pct_change().std() * 100  # in percent
            
            return {
                'primary_trend': primary_trend,
                'trend_strength': trend_strength,
                'adx': last_adx,
                'volatility_pct': volatility,
                'last_sma20': last_sma20,
                'last_sma50': last_sma50
            }
            
        except Exception as e:
            self._log('warning', f"Error in trend analysis: {str(e)}")
            return {
                'primary_trend': 'Unknown',
                'trend_strength': 'Unknown',
                'error': str(e)
            }
    
    def _check_unusual_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for unusual values in the data"""
        result = {'has_unusual_values': False}
        
        # Check for zeros in prices
        zero_prices = ((df['open'] == 0) | (df['high'] == 0) | 
                      (df['low'] == 0) | (df['close'] == 0)).sum()
        
        # Check for negative prices
        negative_prices = ((df['open'] < 0) | (df['high'] < 0) | 
                          (df['low'] < 0) | (df['close'] < 0)).sum()
        
        # Check for high-low relationship
        invalid_hl = (df['high'] < df['low']).sum()
        
        # Check for price anomalies (very large changes)
        pct_changes = df['close'].pct_change().abs()
        outliers = pct_changes[pct_changes > 0.2].count()  # Consider >20% move as outlier
        
        issues = {
            'zero_prices': int(zero_prices),
            'negative_prices': int(negative_prices),
            'high_lower_than_low': int(invalid_hl),
            'price_outliers': int(outliers)
        }
        
        if any(issues.values()):
            result['has_unusual_values'] = True
            result['issues'] = issues
        
        return result
    
    def _generate_parameter_recommendations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate recommended parameters based on data characteristics"""
        # Volatility-based recommendations
        price_volatility = df['close'].pct_change().std()
        
        # Detect timeframe
        timeframe = self._detect_timeframe(df)
        is_intraday = any(tf in timeframe for tf in ['minute', 'hour'])
        
        # ATR calculation for reference
        atr_period = 14
        atr = self._calculate_atr(df, atr_period)
        avg_price = df['close'].mean()
        atr_percent = (atr / avg_price) * 100  # ATR as percentage of price
        
        # Adapt ATR period based on timeframe
        if '1 minute' in timeframe:
            atr_recommendation = (8, 16)
        elif '5 minutes' in timeframe:
            atr_recommendation = (10, 20)
        elif '15 minutes' in timeframe:
            atr_recommendation = (10, 24)
        elif '30 minutes' in timeframe:
            atr_recommendation = (12, 28)
        elif '1 hour' in timeframe:
            atr_recommendation = (14, 30)
        elif '1 day' in timeframe:
            atr_recommendation = (14, 30)
        else:
            atr_recommendation = (14, 28)  # Default
        
        # Adjust factor based on volatility
        if atr_percent < 0.5:  # Low volatility
            factor_recommendation = (0.8, 2.0)
            buffer_recommendation = (0.3, 0.7)
        elif atr_percent < 1.0:  # Moderate volatility
            factor_recommendation = (0.6, 1.5)
            buffer_recommendation = (0.2, 0.5)
        else:  # High volatility
            factor_recommendation = (0.3, 1.2)
            buffer_recommendation = (0.1, 0.4)
            
        # Hard stop recommendation based on ATR and volatility
        if is_intraday:
            # For intraday, base hard stop on average candle range
            avg_range = (df['high'] - df['low']).mean()
            stop_recommendation = (int(avg_range * 1.5), int(avg_range * 5))
        else:
            # For daily and above, use a percentage based approach
            stop_recommendation = (int(avg_price * 0.01), int(avg_price * 0.05))
        
        # Ensure minimum values
        stop_recommendation = (max(5, stop_recommendation[0]), max(20, stop_recommendation[1]))
        
        return {
            'atr_length': {
                'min': atr_recommendation[0], 
                'max': atr_recommendation[1],
                'optimal': int((atr_recommendation[0] + atr_recommendation[1]) / 2)
            },
            'factor': {
                'min': factor_recommendation[0],
                'max': factor_recommendation[1],
                'optimal': (factor_recommendation[0] + factor_recommendation[1]) / 2
            },
            'buffer_multiplier': {
                'min': buffer_recommendation[0],
                'max': buffer_recommendation[1],
                'optimal': (buffer_recommendation[0] + buffer_recommendation[1]) / 2
            },
            'hard_stop_distance': {
                'min': stop_recommendation[0],
                'max': stop_recommendation[1],
                'optimal': int((stop_recommendation[0] + stop_recommendation[1]) / 2)
            },
            'notes': {
                'volatility_pct': atr_percent,
                'timeframe': timeframe,
                'is_intraday': is_intraday
            }
        }
		
# ==============================================================================
# SUPERTREND ALGORITHM
# ==============================================================================

class SuperTrendCalculator:
    """Base class for SuperTrend calculations"""
    
    def __init__(self, log_manager: LogManager = None):
        self.log = log_manager
        self.current_utc = "2025-06-24 12:54:35"
        self.current_user = "arullr001"
    
    def _log(self, level: str, message: str):
        """Log message if log_manager is available"""
        if self.log:
            if level == 'info':
                self.log.info(message)
            elif level == 'debug':
                self.log.debug(message)
            elif level == 'warning':
                self.log.warning(message)
            elif level == 'error':
                self.log.error(message)
    
    def calculate(self, df: pd.DataFrame, atr_length: int, factor: float, 
                 buffer_multiplier: float) -> pd.DataFrame:
        """
        Calculate SuperTrend indicator.
        This is a placeholder to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement calculate()")
    
    def verify_result(self, result_df: pd.DataFrame) -> bool:
        """Verify the computed SuperTrend values"""
        required_columns = [
            'supertrend', 'direction', 'up_trend_buffer', 'down_trend_buffer',
            'buy_signal', 'sell_signal'
        ]
        
        # Check if all required columns exist
        if not all(col in result_df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in result_df.columns]
            self._log('error', f"Missing columns in result: {missing}")
            return False
        
        # Check if direction has expected values (-1, 0, 1)
        direction_values = result_df['direction'].unique()
        if not all(d in [-1, 0, 1] for d in direction_values):
            self._log('error', f"Invalid direction values: {direction_values}")
            return False
        
        # Check signals are boolean
        if not np.issubdtype(result_df['buy_signal'].dtype, np.bool_):
            self._log('warning', "Buy signals not boolean type, converting")
            # Not returning False as this can be fixed
        
        if not np.issubdtype(result_df['sell_signal'].dtype, np.bool_):
            self._log('warning', "Sell signals not boolean type, converting")
            # Not returning False as this can be fixed
        
        # Check for NaN values in critical columns
        critical_cols = ['supertrend', 'direction']
        nan_count = result_df[critical_cols].isna().sum().sum()
        if nan_count > 0:
            self._log('error', f"Found {nan_count} NaN values in critical columns")
            return False
        
        return True


class SuperTrendCPU(SuperTrendCalculator):
    """CPU-optimized vectorized implementation of SuperTrend"""
    
    def calculate(self, df: pd.DataFrame, atr_length: int, factor: float, 
                 buffer_multiplier: float) -> pd.DataFrame:
        """
        Calculate SuperTrend using vectorized operations
        
        Args:
            df: DataFrame with OHLC data
            atr_length: Period for ATR calculation
            factor: Multiplier for ATR to set band distance
            buffer_multiplier: Multiplier for setting buffer zones
            
        Returns:
            DataFrame with SuperTrend calculations and signals
        """
        self._log('info', f"Calculating SuperTrend (CPU) with ATR={atr_length}, Factor={factor}, Buffer={buffer_multiplier}")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Extract price data as numpy arrays for better performance
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        n = len(close)
        
        # Calculate True Range using vectorized operations
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        # Stack and find max along axis
        tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
        # First TR is just high-low
        tr[0] = tr1[0]
        
        # Calculate ATR using numba-optimized function if available, otherwise use numpy
        atr = self._calculate_atr(tr, atr_length)
        
        # Calculate basic bands
        hl2 = (high + low) / 2
        basic_upperband = hl2 + (factor * atr)
        basic_lowerband = hl2 - (factor * atr)
        
        # Initialize arrays for final bands, supertrend, and direction
        final_upperband = np.zeros_like(close)
        final_lowerband = np.zeros_like(close)
        supertrend = np.zeros_like(close)
        direction = np.zeros_like(close)
        
        # Set initial values
        final_upperband[0] = basic_upperband[0]
        final_lowerband[0] = basic_lowerband[0]
        supertrend[0] = (final_upperband[0] + final_lowerband[0]) / 2  # Start in the middle
        direction[0] = 0  # No initial direction
        
        # Calculate SuperTrend using vectorized operations where possible
        for i in range(1, n):
            # Upper band rules
            if basic_upperband[i] < final_upperband[i-1] or close[i-1] > final_upperband[i-1]:
                final_upperband[i] = basic_upperband[i]
            else:
                final_upperband[i] = final_upperband[i-1]
                
            # Lower band rules
            if basic_lowerband[i] > final_lowerband[i-1] or close[i-1] < final_lowerband[i-1]:
                final_lowerband[i] = basic_lowerband[i]
            else:
                final_lowerband[i] = final_lowerband[i-1]
                
            # Determine trend direction
            if close[i] > final_upperband[i-1]:
                direction[i] = -1  # Uptrend (matching Pinescript's convention)
                supertrend[i] = final_lowerband[i]
            elif close[i] < final_lowerband[i-1]:
                direction[i] = 1   # Downtrend (matching Pinescript's convention)
                supertrend[i] = final_upperband[i]
            else:
                direction[i] = direction[i-1]  # Continue previous trend
                supertrend[i] = supertrend[i-1]
        
        # Calculate dynamic buffer zones
        dynamic_buffer = atr * buffer_multiplier
        
        # Add results to DataFrame
        result_df['supertrend'] = supertrend
        result_df['direction'] = direction
        
        # Calculate buffer zones exactly as in Pinescript (using vectorized numpy where)
        result_df['up_trend_buffer'] = np.where(direction < 0, 
                                               supertrend + dynamic_buffer,
                                               np.nan)
        result_df['down_trend_buffer'] = np.where(direction > 0,
                                                 supertrend - dynamic_buffer,
                                                 np.nan)
        
        # Generate signals using Pinescript logic
        result_df['buy_signal'] = (
            (result_df['direction'] < 0) &
            (result_df['close'] >= result_df['supertrend']) & 
            (result_df['close'] <= result_df['up_trend_buffer'])
        )
        
        result_df['sell_signal'] = (
            (result_df['direction'] > 0) & 
            (result_df['close'] < result_df['supertrend']) &
            (result_df['close'] >= result_df['down_trend_buffer'])
        )
        
        # Log stats
        self._log('debug', f"Buy signals: {result_df['buy_signal'].sum()} ({result_df['buy_signal'].mean()*100:.2f}%)")
        self._log('debug', f"Sell signals: {result_df['sell_signal'].sum()} ({result_df['sell_signal'].mean()*100:.2f}%)")
        
        return result_df
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_atr(tr, atr_length):
        """JIT-compiled ATR calculation for better performance"""
        n = len(tr)
        atr = np.zeros_like(tr)
        atr[0] = tr[0]
        
        for i in range(1, n):
            atr[i] = ((atr_length - 1) * atr[i-1] + tr[i]) / atr_length
            
        return atr


# Optimized CUDA kernel for SuperTrend with shared memory
@cuda.jit
def supertrend_cuda_kernel_optimized(high, low, close, atr_length_f, factor_f, buffer_multiplier_f,
                                   atr, supertrend, direction, up_trend_buffer, down_trend_buffer):
    """Enhanced CUDA kernel for SuperTrend calculation with shared memory optimizations"""
    # Get block and thread indices
    i = cuda.grid(1)
    n = len(close)
    
    # Allocate shared memory for this thread block
    thread_id = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    # Create shared memory arrays for commonly accessed data
    shared_high = cuda.shared.array(shape=THREADS_PER_BLOCK, dtype=numba.float64)
    shared_low = cuda.shared.array(shape=THREADS_PER_BLOCK, dtype=numba.float64)
    shared_close = cuda.shared.array(shape=THREADS_PER_BLOCK, dtype=numba.float64)
    shared_prev_close = cuda.shared.array(shape=THREADS_PER_BLOCK, dtype=numba.float64)
    
    # Load data into shared memory
    if i < n:
        shared_high[thread_id] = high[i]
        shared_low[thread_id] = low[i]
        shared_close[thread_id] = close[i]
        if i > 0:
            shared_prev_close[thread_id] = close[i-1]
        else:
            shared_prev_close[thread_id] = close[0]  # Handle edge case
    
    # Synchronize threads to ensure all shared memory is loaded
    cuda.syncthreads()
    
    if i >= n:
        return
    
    # Explicit type conversion for parameters
    atr_length = float(atr_length_f)
    factor = float(factor_f)
    buffer_multiplier = float(buffer_multiplier_f)
    
    # Calculate TR with shared memory
    if i == 0:
        tr = shared_high[thread_id] - shared_low[thread_id]
    else:
        tr1 = shared_high[thread_id] - shared_low[thread_id]
        tr2 = abs(shared_high[thread_id] - shared_prev_close[thread_id])
        tr3 = abs(shared_low[thread_id] - shared_prev_close[thread_id])
        tr = max(tr1, max(tr2, tr3))
    
    # Calculate ATR
    if i == 0:
        atr[i] = tr
    else:
        atr[i] = ((atr_length - 1.0) * atr[i-1] + tr) / atr_length
    
    # Calculate basic bands
    hl2 = (shared_high[thread_id] + shared_low[thread_id]) / 2.0
    basic_upperband = hl2 + (factor * atr[i])
    basic_lowerband = hl2 - (factor * atr[i])
    
    # First value initialization
    if i == 0:
        supertrend[i] = hl2
        direction[i] = 0.0
        up_trend_buffer[i] = 0.0
        down_trend_buffer[i] = 0.0
        return
    
    # Calculate final upper and lower bands
    final_upperband = basic_upperband
    final_lowerband = basic_lowerband
    
    # Optimize band calculation
    prev_supertrend = supertrend[i-1]
    prev_close = shared_prev_close[thread_id]
    
    # Upper band logic
    if basic_upperband < prev_supertrend and prev_close > prev_supertrend:
        final_upperband = basic_upperband
    elif prev_close <= prev_supertrend:
        final_upperband = basic_upperband
    else:
        final_upperband = min(basic_upperband, prev_supertrend)
    
    # Lower band logic
    if basic_lowerband > prev_supertrend and prev_close < prev_supertrend:
        final_lowerband = basic_lowerband
    elif prev_close >= prev_supertrend:
        final_lowerband = basic_lowerband
    else:
        final_lowerband = max(basic_lowerband, prev_supertrend)
    
    # Determine trend direction
    if shared_close[thread_id] > prev_supertrend:
        direction[i] = -1.0  # Uptrend
        supertrend[i] = final_lowerband
    elif shared_close[thread_id] < prev_supertrend:
        direction[i] = 1.0   # Downtrend
        supertrend[i] = final_upperband
    else:
        direction[i] = direction[i-1]
        supertrend[i] = prev_supertrend
    
    # Calculate buffer zones
    dynamic_buffer = atr[i] * buffer_multiplier
    if direction[i] < 0:
        up_trend_buffer[i] = supertrend[i] + dynamic_buffer
        down_trend_buffer[i] = 0.0  # nan equivalent
    else:  # direction[i] > 0
        up_trend_buffer[i] = 0.0    # nan equivalent
        down_trend_buffer[i] = supertrend[i] - dynamic_buffer


class SuperTrendGPU(SuperTrendCalculator):
    """Enhanced GPU-accelerated implementation of SuperTrend using CUDA"""
    
    def calculate(self, df: pd.DataFrame, atr_length: int, factor: float, 
                 buffer_multiplier: float) -> pd.DataFrame:
        """
        Calculate SuperTrend using optimized GPU acceleration
        
        Args:
            df: DataFrame with OHLC data
            atr_length: Period for ATR calculation
            factor: Multiplier for ATR to set band distance
            buffer_multiplier: Multiplier for setting buffer zones
            
        Returns:
            DataFrame with SuperTrend calculations and signals
        """
        self._log('info', f"Calculating SuperTrend (GPU) with ATR={atr_length}, Factor={factor}, Buffer={buffer_multiplier}")
        
        if not cuda.is_available():
            self._log('warning', "CUDA not available, falling back to CPU implementation")
            cpu_calculator = SuperTrendCPU(self.log)
            return cpu_calculator.calculate(df, atr_length, factor, buffer_multiplier)
        
        try:
            # Make a copy and extract arrays
            result_df = df.copy()
            n = len(df)
            
            # Check GPU memory and optimize batch size
            ctx = cuda.current_context()
            free_mem = ctx.get_memory_info().free
            required_mem = n * 8 * 9  # 9 arrays of float64
            
            self._log('debug', f"GPU memory check: Required={required_mem/1e6:.1f}MB, Available={free_mem/1e6:.1f}MB")
            
            if required_mem > free_mem * 0.8:  # Leave 20% buffer
                # Try to process in batches if dataset is too large
                if n > 50000 and required_mem > free_mem * 0.3:
                    return self._process_in_batches(df, atr_length, factor, buffer_multiplier)
                else:
                    self._log('warning', "Insufficient GPU memory, falling back to CPU")
                    cpu_calculator = SuperTrendCPU(self.log)
                    return cpu_calculator.calculate(df, atr_length, factor, buffer_multiplier)
            
            # Prepare input arrays with optimal memory layout
            high = cuda.to_device(df['high'].values.astype(np.float64))
            low = cuda.to_device(df['low'].values.astype(np.float64))
            close = cuda.to_device(df['close'].values.astype(np.float64))
            
            # Pre-allocate output arrays on device with pinned memory for faster transfer
            atr = cuda.device_array(n, dtype=np.float64)
            supertrend = cuda.device_array(n, dtype=np.float64)
            direction = cuda.device_array(n, dtype=np.float64)
            up_trend_buffer = cuda.device_array(n, dtype=np.float64)
            down_trend_buffer = cuda.device_array(n, dtype=np.float64)
            
            # Configure kernel launch parameters for optimal performance
            threads_per_block = THREADS_PER_BLOCK
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
            
            self._log('debug', f"Launching optimized CUDA kernel with {blocks_per_grid} blocks of {threads_per_block} threads")
            
            # Launch optimized kernel
            supertrend_cuda_kernel_optimized[blocks_per_grid, threads_per_block](
                high, low, close, float(atr_length), float(factor), float(buffer_multiplier),
                atr, supertrend, direction, up_trend_buffer, down_trend_buffer
            )
            
            # Synchronize to ensure kernel completion
            cuda.synchronize()
            
            # Copy results back to host using asynchronous transfers where possible
            h_supertrend = supertrend.copy_to_host()
            h_direction = direction.copy_to_host()
            h_up_buffer = up_trend_buffer.copy_to_host()
            h_down_buffer = down_trend_buffer.copy_to_host()
            
            # Store results in DataFrame
            result_df['supertrend'] = h_supertrend
            result_df['direction'] = h_direction
            result_df['up_trend_buffer'] = h_up_buffer
            result_df['down_trend_buffer'] = h_down_buffer
            
            # Convert 0.0 in buffer arrays to NaN (CUDA can't handle NaN directly)
            result_df['up_trend_buffer'] = result_df['up_trend_buffer'].replace(0.0, np.nan)
            result_df['down_trend_buffer'] = result_df['down_trend_buffer'].replace(0.0, np.nan)
            
            # Generate signals using vectorized operations for performance
            result_df['buy_signal'] = (
                (result_df['direction'] < 0) &
                (result_df['close'] >= result_df['supertrend']) & 
                (result_df['close'] <= result_df['up_trend_buffer'])
            )
            
            result_df['sell_signal'] = (
                (result_df['direction'] > 0) & 
                (result_df['close'] < result_df['supertrend']) &
                (result_df['close'] >= result_df['down_trend_buffer'])
            )
            
            # Log stats
            buy_count = result_df['buy_signal'].sum()
            sell_count = result_df['sell_signal'].sum()
            self._log('debug', f"Buy signals: {buy_count} ({buy_count/len(df)*100:.2f}%)")
            self._log('debug', f"Sell signals: {sell_count} ({sell_count/len(df)*100:.2f}%)")
            
            # Clean up GPU memory explicitly
            del high, low, close, atr, supertrend, direction, up_trend_buffer, down_trend_buffer
            
            # Force cleanup of GPU memory
            cuda.current_context().deallocations.clear()
            
            return result_df
            
        except Exception as e:
            self._log('error', f"GPU calculation error: {str(e)}", exc_info=e)
            self._log('warning', "Falling back to CPU implementation")
            cpu_calculator = SuperTrendCPU(self.log)
            return cpu_calculator.calculate(df, atr_length, factor, buffer_multiplier)
    
    def _process_in_batches(self, df: pd.DataFrame, atr_length: int, factor: float, 
                           buffer_multiplier: float) -> pd.DataFrame:
        """Process large datasets in batches to avoid GPU memory limitations"""
        self._log('info', "Processing large dataset in batches")
        
        # Determine batch size based on available memory
        n = len(df)
        ctx = cuda.current_context()
        free_mem = ctx.get_memory_info().free
        
        # Calculate safe batch size (allocating ~40% of free memory)
        bytes_per_row = 8 * 9  # 9 float64 arrays
        max_rows = int((free_mem * 0.4) / bytes_per_row)
        batch_size = min(50000, max_rows)  # Cap at 50k rows per batch
        
        self._log('debug', f"Using batch size of {batch_size} rows")
        
        # Initialize result DataFrame
        result_df = df.copy()
        result_df['supertrend'] = np.nan
        result_df['direction'] = np.nan
        result_df['up_trend_buffer'] = np.nan
        result_df['down_trend_buffer'] = np.nan
        
        # First batch needs special handling to initialize values
        first_batch_size = min(batch_size, n)
        first_batch = df.iloc[:first_batch_size].copy()
        
        # Process first batch
        self._log('debug', f"Processing first batch of {first_batch_size} rows")
        batch_result = SuperTrendCPU(self.log).calculate(first_batch, atr_length, factor, buffer_multiplier)
        
        # Copy results from first batch
        result_df.iloc[:first_batch_size, result_df.columns.get_indexer(['supertrend', 'direction', 
                                                                        'up_trend_buffer', 'down_trend_buffer'])] = \
            batch_result[['supertrend', 'direction', 'up_trend_buffer', 'down_trend_buffer']]
        
        # Process remaining batches, carrying over state from previous batch
        for start_idx in range(batch_size, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            self._log('debug', f"Processing batch from {start_idx} to {end_idx}")
            
            # Create batch with one extra row at the beginning for state carryover
            batch_with_prev = df.iloc[start_idx-1:end_idx].copy()
            
            # Process this batch
            batch_result = SuperTrendCPU(self.log).calculate(batch_with_prev, atr_length, factor, buffer_multiplier)
            
            # Copy results, skipping the first row which was just for state carryover
            result_df.iloc[start_idx:end_idx, result_df.columns.get_indexer(['supertrend', 'direction', 
                                                                            'up_trend_buffer', 'down_trend_buffer'])] = \
                batch_result.iloc[1:][['supertrend', 'direction', 'up_trend_buffer', 'down_trend_buffer']]
        
        # Generate signals for the entire dataset
        result_df['buy_signal'] = (
            (result_df['direction'] < 0) &
            (result_df['close'] >= result_df['supertrend']) & 
            (result_df['close'] <= result_df['up_trend_buffer'])
        )
        
        result_df['sell_signal'] = (
            (result_df['direction'] > 0) & 
            (result_df['close'] < result_df['supertrend']) &
            (result_df['close'] >= result_df['down_trend_buffer'])
        )
        
        return result_df


class SuperTrendCuPy(SuperTrendCalculator):
    """CuPy-based implementation of SuperTrend for maximum performance"""
    
    def calculate(self, df: pd.DataFrame, atr_length: int, factor: float, 
                 buffer_multiplier: float) -> pd.DataFrame:
        """
        Calculate SuperTrend using CuPy for maximum GPU performance
        
        Args:
            df: DataFrame with OHLC data
            atr_length: Period for ATR calculation
            factor: Multiplier for ATR to set band distance
            buffer_multiplier: Multiplier for setting buffer zones
            
        Returns:
            DataFrame with SuperTrend calculations and signals
        """
        self._log('info', f"Calculating SuperTrend (CuPy) with ATR={atr_length}, Factor={factor}, Buffer={buffer_multiplier}")
        
        if not HAS_CUPY:
            self._log('warning', "CuPy not available, falling back to CUDA implementation")
            gpu_calculator = SuperTrendGPU(self.log)
            return gpu_calculator.calculate(df, atr_length, factor, buffer_multiplier)
        
        try:
            # Make a copy for the result
            result_df = df.copy()
            
            # Extract arrays and transfer to GPU
            high = cp.array(df['high'].values)
            low = cp.array(df['low'].values)
            close = cp.array(df['close'].values)
            n = len(close)
            
            # Calculate True Range
            tr1 = high - low
            tr2 = cp.abs(high - cp.roll(close, 1))
            tr3 = cp.abs(low - cp.roll(close, 1))
            
            # Stack and find max along axis
            tr = cp.maximum(tr1, cp.maximum(tr2, tr3))
            tr[0] = tr1[0]  # First TR is just high-low
            
            # Calculate ATR
            atr = cp.zeros_like(close)
            atr[0] = tr[0]
            for i in range(1, n):
                atr[i] = ((atr_length - 1) * atr[i-1] + tr[i]) / atr_length
            
            # Calculate basic bands
            hl2 = (high + low) / 2
            basic_upperband = hl2 + (factor * atr)
            basic_lowerband = hl2 - (factor * atr)
            
            # Initialize arrays for final bands, supertrend, and direction
            final_upperband = cp.zeros_like(close)
            final_lowerband = cp.zeros_like(close)
            supertrend = cp.zeros_like(close)
            direction = cp.zeros_like(close)
            
            # Set initial values
            final_upperband[0] = basic_upperband[0]
            final_lowerband[0] = basic_lowerband[0]
            supertrend[0] = (final_upperband[0] + final_lowerband[0]) / 2  # Start in the middle
            direction[0] = 0  # No initial direction
            
            # Calculate SuperTrend (this loop is still needed for correct calculation)
            for i in range(1, n):
                # Upper band rules
                if basic_upperband[i] < final_upperband[i-1] or close[i-1] > final_upperband[i-1]:
                    final_upperband[i] = basic_upperband[i]
                else:
                    final_upperband[i] = final_upperband[i-1]
                    
                # Lower band rules
                if basic_lowerband[i] > final_lowerband[i-1] or close[i-1] < final_lowerband[i-1]:
                    final_lowerband[i] = basic_lowerband[i]
                else:
                    final_lowerband[i] = final_lowerband[i-1]
                    
                # Determine trend direction
                if close[i] > final_upperband[i-1]:
                    direction[i] = -1  # Uptrend
                    supertrend[i] = final_lowerband[i]
                elif close[i] < final_lowerband[i-1]:
                    direction[i] = 1   # Downtrend
                    supertrend[i] = final_upperband[i]
                else:
                    direction[i] = direction[i-1]  # Continue previous trend
                    supertrend[i] = supertrend[i-1]
            
            # Calculate dynamic buffer zones
            dynamic_buffer = atr * buffer_multiplier
            
            # Convert back to numpy and store in the DataFrame
            result_df['supertrend'] = cp.asnumpy(supertrend)
            result_df['direction'] = cp.asnumpy(direction)
            
            # Create buffer zones (this part in numpy)
            direction_np = cp.asnumpy(direction)
            supertrend_np = cp.asnumpy(supertrend)
            dynamic_buffer_np = cp.asnumpy(dynamic_buffer)
            
            # Transfer everything back to CPU for DataFrame operations
            result_df['up_trend_buffer'] = np.where(
                direction_np < 0, 
                supertrend_np + dynamic_buffer_np,
                np.nan
            )
            
            result_df['down_trend_buffer'] = np.where(
                direction_np > 0,
                supertrend_np - dynamic_buffer_np,
                np.nan
            )
            
            # Generate signals using Pinescript logic
            result_df['buy_signal'] = (
                (result_df['direction'] < 0) &
                (result_df['close'] >= result_df['supertrend']) & 
                (result_df['close'] <= result_df['up_trend_buffer'])
            )
            
            result_df['sell_signal'] = (
                (result_df['direction'] > 0) & 
                (result_df['close'] < result_df['supertrend']) &
                (result_df['close'] >= result_df['down_trend_buffer'])
            )
            
            # Log stats
            self._log('debug', f"Buy signals: {result_df['buy_signal'].sum()} ({result_df['buy_signal'].mean()*100:.2f}%)")
            self._log('debug', f"Sell signals: {result_df['sell_signal'].sum()} ({result_df['sell_signal'].mean()*100:.2f}%)")
            
            # Explicitly clear GPU memory
            del high, low, close, tr1, tr2, tr3, tr, atr
            del hl2, basic_upperband, basic_lowerband
            del final_upperband, final_lowerband, supertrend, direction, dynamic_buffer
            cp.get_default_memory_pool().free_all_blocks()
            
            return result_df
            
        except Exception as e:
            self._log('error', f"CuPy calculation error: {str(e)}", exc_info=e)
            self._log('warning', "Falling back to CUDA implementation")
            gpu_calculator = SuperTrendGPU(self.log)
            return gpu_calculator.calculate(df, atr_length, factor, buffer_multiplier)


class SuperTrend:
    """
    SuperTrend indicator calculator with GPU-optimized implementation
    """
    
    def __init__(self, log_manager: LogManager = None):
        self.log = log_manager
        self.current_utc = "2025-06-27 19:36:52"  # Updated timestamp
        self.current_user = "arullr001"           # Updated username
        
        # Track GPU context
        self._gpu_context = None
        self._has_cudf = HAS_CUDF
        self._has_numba = HAS_NUMBA
        
        if self._has_cudf:
            self._log('info', "CUDF is available for GPU acceleration")
        else:
            self._log('debug', "CUDF not available, using fallback implementations")
            
        if self._has_numba:
            self._log('info', "Numba is available for JIT compilation")
        else:
            self._log('debug', "Numba not available, using fallback implementations")
    
    def _log(self, level: str, message: str):
        """Log message if log_manager is available"""
        if self.log:
            if level == 'info':
                self.log.info(message)
            elif level == 'debug':
                self.log.debug(message)
            elif level == 'warning':
                self.log.warning(message)
            elif level == 'error':
                self.log.error(message)
    
    def calculate(self, df: pd.DataFrame, 
                atr_length: int = 14,
                factor: float = 3.0,
                buffer_multiplier: float = 0.3,
                force_implementation: str = None,
                precomputed_patterns: Dict = None) -> pd.DataFrame:
        """
        Calculate SuperTrend and return dataframe with added columns
        
        Args:
            df: DataFrame with OHLC data
            atr_length: Period for ATR calculation
            factor: Multiplier for ATR
            buffer_multiplier: Additional buffer for trend changes
            force_implementation: Force specific implementation ('cpu', 'cudf', 'numba', 'gpu')
            precomputed_patterns: Dictionary with precomputed elements for optimization
            
        Returns:
            DataFrame with SuperTrend columns:
                - supertrend: SuperTrend value
                - trend: 1 for uptrend, -1 for downtrend
                - trend_changed: True on trend change candle
                - supertrend_signal: Signal (1=buy, -1=sell, 0=no signal)
                - supertrend_upper: Upper band
                - supertrend_lower: Lower band
        """
        # If we have precomputed patterns and they match the current parameters, use them
        if precomputed_patterns and self._can_use_precomputed(precomputed_patterns, atr_length):
            self._log('debug', "Using precomputed patterns for SuperTrend calculation")
            return self._calculate_with_precomputed(df, atr_length, factor, buffer_multiplier, precomputed_patterns)
        
        # Choose implementation
        implementation = self._choose_implementation(force_implementation)
        
        # Log the implementation choice
        self._log('debug', f"Using {implementation} implementation for SuperTrend calculation")
        
        # Create a copy of input dataframe to avoid modifying the original
        result_df = df.copy()
        
        # Calculate based on chosen implementation
        if implementation == 'cudf':
            result_df = self._calculate_cudf(result_df, atr_length, factor, buffer_multiplier)
        elif implementation == 'numba':
            result_df = self._calculate_numba(result_df, atr_length, factor, buffer_multiplier)
        else:
            # Fallback to CPU implementation
            result_df = self._calculate_cpu(result_df, atr_length, factor, buffer_multiplier)
        
        return result_df

    def precompute_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Precompute common elements for multiple SuperTrend calculations to optimize GPU usage
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dictionary with precomputed elements
        """
        try:
            if not HAS_CUDF:
                self._log('warning', "CUDF not available, skipping precomputation")
                return None
                
            # Import as needed
            import cudf
            import numpy as np
            
            # Create or reuse GPU context
            self._ensure_gpu_context()
            
            # Start by moving data to GPU
            self._log('debug', "Moving data to GPU for precomputation")
            gdf = cudf.DataFrame.from_pandas(df)
            
            # Precompute high-low difference and ATR components (these are independent of atr_length)
            gdf['hl_diff'] = gdf['high'] - gdf['low']
            gdf['hc_diff'] = (gdf['high'] - gdf['close'].shift(1)).abs()
            gdf['lc_diff'] = (gdf['low'] - gdf['close'].shift(1)).abs()
            
            # Precompute common ATR lengths
            atr_patterns = {}
            common_lengths = [7, 10, 14, 20, 30]
            
            for length in common_lengths:
                # Calculate true range
                gdf['tr'] = gdf[['hl_diff', 'hc_diff', 'lc_diff']].max(axis=1)
                
                # Calculate ATR
                gdf[f'atr_{length}'] = gdf['tr'].rolling(window=length).mean()
                
                # Store the ATR column in our patterns dictionary
                atr_patterns[length] = gdf[f'atr_{length}'].to_pandas()
            
            # Calculate median price
            gdf['median_price'] = (gdf['high'] + gdf['low']) / 2
            
            # Store the required columns in our patterns dictionary
            patterns = {
                'high': gdf['high'].to_array(),
                'low': gdf['low'].to_array(),
                'close': gdf['close'].to_array(),
                'median_price': gdf['median_price'].to_pandas(),
                'atr_patterns': atr_patterns,
                'data_fingerprint': self._get_data_fingerprint(df)
            }
            
            self._log('info', f"Successfully precomputed patterns for SuperTrend with {len(common_lengths)} ATR lengths")
            
            return patterns
            
        except Exception as e:
            self._log('error', f"Error in precomputing patterns: {str(e)}")
            # Don't raise exception, just return None to allow fallback to regular calculation
            return None
    
    def _get_data_fingerprint(self, df: pd.DataFrame) -> str:
        """Get a fingerprint of the data to verify compatibility with precomputed patterns"""
        return f"{len(df)}_{df.index[0]}_{df.index[-1]}"
    
    def _can_use_precomputed(self, precomputed_patterns: Dict, atr_length: int) -> bool:
        """Check if precomputed patterns can be used for the requested parameters"""
        if not precomputed_patterns:
            return False
            
        # Check if we have the requested ATR length
        if 'atr_patterns' not in precomputed_patterns:
            return False
            
        if atr_length not in precomputed_patterns['atr_patterns']:
            return False
            
        return True
    
    def _calculate_with_precomputed(self, df: pd.DataFrame, 
                                  atr_length: int,
                                  factor: float,
                                  buffer_multiplier: float,
                                  precomputed_patterns: Dict) -> pd.DataFrame:
        """Calculate SuperTrend using precomputed patterns"""
        try:
            # Create a copy of input dataframe
            result_df = df.copy()
            
            # Get precomputed ATR
            atr = precomputed_patterns['atr_patterns'][atr_length]
            
            # Get median price
            median_price = precomputed_patterns['median_price']
            
            # Calculate basic and final bands
            result_df['basic_upper_band'] = median_price + factor * atr
            result_df['basic_lower_band'] = median_price - factor * atr
            
            # Calculate with buffers
            result_df['final_upper_band'] = self._calculate_final_bands(
                result_df['basic_upper_band'], 
                result_df['close'], 
                'upper',
                buffer_multiplier
            )
            
            result_df['final_lower_band'] = self._calculate_final_bands(
                result_df['basic_lower_band'], 
                result_df['close'], 
                'lower',
                buffer_multiplier
            )
            
            # Calculate trend and supertrend
            result_df['trend'] = self._calculate_trend(
                result_df['close'], 
                result_df['final_upper_band'], 
                result_df['final_lower_band']
            )
            
            # Calculate supertrend values and signals
            self._calculate_supertrend_signals(result_df)
            
            # Drop intermediate columns
            result_df.drop(['basic_upper_band', 'basic_lower_band'], axis=1, inplace=True)
            
            return result_df
            
        except Exception as e:
            self._log('error', f"Error in precomputed calculation: {str(e)}")
            # Fallback to CPU implementation
            self._log('info', "Falling back to CPU implementation")
            return self._calculate_cpu(df, atr_length, factor, buffer_multiplier)
    
    def _choose_implementation(self, force_implementation: str = None) -> str:
        """Choose the appropriate implementation based on availability and preferences"""
        # If implementation is forced, try to use it
        if force_implementation:
            if force_implementation.lower() == 'cudf' and not HAS_CUDF:
                self._log('warning', "CUDF implementation requested but not available, falling back")
            elif force_implementation.lower() == 'numba' and not HAS_NUMBA:
                self._log('warning', "Numba implementation requested but not available, falling back")
            elif force_implementation.lower() == 'gpu' and not (HAS_CUDF or HAS_NUMBA):
                self._log('warning', "GPU implementation requested but neither CUDF nor Numba available, falling back")
            elif force_implementation.lower() in ['cudf', 'numba', 'gpu', 'cpu']:
                if force_implementation.lower() == 'gpu':
                    # For 'gpu', prioritize CUDF over Numba
                    return 'cudf' if HAS_CUDF else ('numba' if HAS_NUMBA else 'cpu')
                return force_implementation.lower()
        
        # Auto-select best available implementation
        if HAS_CUDF:
            return 'cudf'
        elif HAS_NUMBA:
            return 'numba'
        else:
            return 'cpu'
    
    def _calculate_cpu(self, df: pd.DataFrame, 
                     atr_length: int,
                     factor: float,
                     buffer_multiplier: float) -> pd.DataFrame:
        """Calculate SuperTrend using CPU implementation"""
        # Create a copy of input dataframe
        result_df = df.copy()
        
        # Calculate true range
        tr1 = result_df['high'] - result_df['low']
        tr2 = (result_df['high'] - result_df['close'].shift(1)).abs()
        tr3 = (result_df['low'] - result_df['close'].shift(1)).abs()
        result_df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        result_df['atr'] = result_df['tr'].rolling(window=atr_length).mean()
        
        # Calculate median price
        result_df['median_price'] = (result_df['high'] + result_df['low']) / 2
        
        # Calculate basic bands
        result_df['basic_upper_band'] = result_df['median_price'] + factor * result_df['atr']
        result_df['basic_lower_band'] = result_df['median_price'] - factor * result_df['atr']
        
        # Calculate final bands with buffer
        result_df['final_upper_band'] = self._calculate_final_bands(
            result_df['basic_upper_band'], 
            result_df['close'], 
            'upper',
            buffer_multiplier
        )
        
        result_df['final_lower_band'] = self._calculate_final_bands(
            result_df['basic_lower_band'], 
            result_df['close'], 
            'lower',
            buffer_multiplier
        )
        
        # Calculate trend
        result_df['trend'] = self._calculate_trend(
            result_df['close'], 
            result_df['final_upper_band'], 
            result_df['final_lower_band']
        )
        
        # Calculate supertrend
        self._calculate_supertrend_signals(result_df)
        
        # Clean up intermediate columns
        result_df.drop(['tr', 'atr', 'median_price', 'basic_upper_band', 'basic_lower_band'], axis=1, inplace=True)
        
        return result_df
    
    def _calculate_cudf(self, df: pd.DataFrame, 
                      atr_length: int,
                      factor: float,
                      buffer_multiplier: float) -> pd.DataFrame:
        """Calculate SuperTrend using CUDF (GPU) implementation"""
        try:
            # Import cudf
            import cudf
            
            # Create or reuse GPU context
            self._ensure_gpu_context()
            
            # Convert pandas DataFrame to cuDF DataFrame
            gdf = cudf.DataFrame.from_pandas(df)
            
            # Calculate true range
            gdf['hl_diff'] = gdf['high'] - gdf['low']
            gdf['hc_diff'] = (gdf['high'] - gdf['close'].shift(1)).abs()
            gdf['lc_diff'] = (gdf['low'] - gdf['close'].shift(1)).abs()
            gdf['tr'] = gdf[['hl_diff', 'hc_diff', 'lc_diff']].max(axis=1)
            
            # Calculate ATR
            gdf['atr'] = gdf['tr'].rolling(window=atr_length).mean()
            
            # Calculate median price
            gdf['median_price'] = (gdf['high'] + gdf['low']) / 2
            
            # Calculate basic bands
            gdf['basic_upper_band'] = gdf['median_price'] + factor * gdf['atr']
            gdf['basic_lower_band'] = gdf['median_price'] - factor * gdf['atr']
            
            # Cannot directly use _calculate_final_bands on GPU, do calculation inline
            # Final upper band
            gdf['final_upper_band'] = 0.0
            for i in range(1, len(gdf)):
                if (gdf['basic_upper_band'][i] < gdf['final_upper_band'][i-1] or 
                    gdf['close'][i-1] > gdf['final_upper_band'][i-1]):
                    gdf['final_upper_band'][i] = gdf['basic_upper_band'][i]
                else:
                    gdf['final_upper_band'][i] = gdf['final_upper_band'][i-1]
                    
                # Apply buffer
                if buffer_multiplier > 0:
                    buffer_val = buffer_multiplier * gdf['atr'][i]
                    if gdf['close'][i-1] > gdf['final_upper_band'][i-1]:
                        # Allow more room when switching from below to above
                        gdf['final_upper_band'][i] = gdf['final_upper_band'][i] + buffer_val
            
            # Final lower band
            gdf['final_lower_band'] = 0.0
            for i in range(1, len(gdf)):
                if (gdf['basic_lower_band'][i] > gdf['final_lower_band'][i-1] or 
                    gdf['close'][i-1] < gdf['final_lower_band'][i-1]):
                    gdf['final_lower_band'][i] = gdf['basic_lower_band'][i]
                else:
                    gdf['final_lower_band'][i] = gdf['final_lower_band'][i-1]
                    
                # Apply buffer
                if buffer_multiplier > 0:
                    buffer_val = buffer_multiplier * gdf['atr'][i]
                    if gdf['close'][i-1] < gdf['final_lower_band'][i-1]:
                        # Allow more room when switching from above to below
                        gdf['final_lower_band'][i] = gdf['final_lower_band'][i] - buffer_val
            
            # Calculate trend
            gdf['trend'] = 0
            for i in range(1, len(gdf)):
                if gdf['close'][i] > gdf['final_upper_band'][i-1]:
                    gdf['trend'][i] = 1
                elif gdf['close'][i] < gdf['final_lower_band'][i-1]:
                    gdf['trend'][i] = -1
                else:
                    gdf['trend'][i] = gdf['trend'][i-1]
            
            # Calculate supertrend
            gdf['supertrend'] = 0.0
            for i in range(1, len(gdf)):
                if gdf['trend'][i] == 1:
                    gdf['supertrend'][i] = gdf['final_lower_band'][i]
                else:
                    gdf['supertrend'][i] = gdf['final_upper_band'][i]
            
            # Calculate trend changed and signals
            gdf['trend_changed'] = gdf['trend'] != gdf['trend'].shift(1)
            gdf['supertrend_signal'] = 0
            
            # Trend change signals
            for i in range(1, len(gdf)):
                if gdf['trend_changed'][i]:
                    if gdf['trend'][i] == 1:
                        gdf['supertrend_signal'][i] = 1  # Buy signal
                    else:
                        gdf['supertrend_signal'][i] = -1  # Sell signal
            
            # Store the upper and lower bands
            gdf['supertrend_upper'] = gdf['final_upper_band'] 
            gdf['supertrend_lower'] = gdf['final_lower_band']
            
            # Convert back to pandas
            result_df = gdf[[
                'open', 'high', 'low', 'close', 'volume',
                'trend', 'trend_changed', 'supertrend', 
                'supertrend_signal', 'supertrend_upper', 'supertrend_lower'
            ]].to_pandas()
            
            return result_df
            
        except Exception as e:
            self._log('error', f"CUDF calculation error: {str(e)}")
            self._log('info', "Falling back to CPU implementation")
            return self._calculate_cpu(df, atr_length, factor, buffer_multiplier)
    
    def _calculate_numba(self, df: pd.DataFrame, 
                       atr_length: int,
                       factor: float,
                       buffer_multiplier: float) -> pd.DataFrame:
        """Calculate SuperTrend using Numba JIT compilation"""
        try:
            import numpy as np
            from numba import njit, prange
            
            # Create a copy of input dataframe
            result_df = df.copy()
            
            # Calculate ATR using pandas first
            tr1 = result_df['high'] - result_df['low']
            tr2 = abs(result_df['high'] - result_df['close'].shift(1))
            tr3 = abs(result_df['low'] - result_df['close'].shift(1))
            result_df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            result_df['atr'] = result_df['tr'].rolling(window=atr_length).mean()
            
            # Calculate median price
            result_df['median_price'] = (result_df['high'] + result_df['low']) / 2
            
            # Basic bands
            result_df['basic_upper_band'] = result_df['median_price'] + factor * result_df['atr']
            result_df['basic_lower_band'] = result_df['median_price'] - factor * result_df['atr']
            
            # Get numpy arrays for numba processing
            close = result_df['close'].values
            basic_upper = result_df['basic_upper_band'].values
            basic_lower = result_df['basic_lower_band'].values
            atr = result_df['atr'].values
            
            # Define numba functions
            @njit
            def calculate_final_bands_numba(basic_bands, close_values, atr_values, buffer_mult, is_upper=True):
                n = len(basic_bands)
                final_bands = np.zeros(n)
                
                # First value
                final_bands[0] = basic_bands[0]
                
                # Calculate rest
                for i in range(1, n):
                    if is_upper:
                        if ((basic_bands[i] < final_bands[i-1]) or 
                            (close_values[i-1] > final_bands[i-1])):
                            final_bands[i] = basic_bands[i]
                        else:
                            final_bands[i] = final_bands[i-1]
                            
                        # Apply buffer when switching trends
                        if buffer_mult > 0 and close_values[i-1] > final_bands[i-1]:
                            buffer_val = buffer_mult * atr_values[i]
                            final_bands[i] = final_bands[i] + buffer_val
                    else:
                        if ((basic_bands[i] > final_bands[i-1]) or 
                            (close_values[i-1] < final_bands[i-1])):
                            final_bands[i] = basic_bands[i]
                        else:
                            final_bands[i] = final_bands[i-1]
                            
                        # Apply buffer when switching trends
                        if buffer_mult > 0 and close_values[i-1] < final_bands[i-1]:
                            buffer_val = buffer_mult * atr_values[i]
                            final_bands[i] = final_bands[i] - buffer_val
                
                return final_bands
            
            @njit
            def calculate_trend_numba(close_values, upper_bands, lower_bands):
                n = len(close_values)
                trend = np.zeros(n, dtype=np.int32)
                
                # First value initialization
                if close_values[0] > upper_bands[0]:
                    trend[0] = 1
                elif close_values[0] < lower_bands[0]:
                    trend[0] = -1
                    
                # Calculate rest
                for i in range(1, n):
                    if close_values[i] > upper_bands[i-1]:
                        trend[i] = 1
                    elif close_values[i] < lower_bands[i-1]:
                        trend[i] = -1
                    else:
                        trend[i] = trend[i-1]
                
                return trend
            
            @njit
            def calculate_supertrend_numba(trend, upper_bands, lower_bands):
                n = len(trend)
                supertrend = np.zeros(n)
                
                for i in range(n):
                    if trend[i] == 1:
                        supertrend[i] = lower_bands[i]
                    else:
                        supertrend[i] = upper_bands[i]
                
                return supertrend
            
            @njit
            def calculate_signals_numba(trend):
                n = len(trend)
                changed = np.zeros(n, dtype=np.bool_)
                signals = np.zeros(n, dtype=np.int32)
                
                # First value can't have changed
                changed[0] = False
                
                # Calculate trend changes and signals
                for i in range(1, n):
                    if trend[i] != trend[i-1]:
                        changed[i] = True
                        if trend[i] == 1:
                            signals[i] = 1  # Buy signal
                        else:
                            signals[i] = -1  # Sell signal
                
                return changed, signals
            
            # Execute numba functions
            final_upper = calculate_final_bands_numba(basic_upper, close, atr, buffer_multiplier, True)
            final_lower = calculate_final_bands_numba(basic_lower, close, atr, buffer_multiplier, False)
            trend = calculate_trend_numba(close, final_upper, final_lower)
            supertrend = calculate_supertrend_numba(trend, final_upper, final_lower)
            trend_changed, signals = calculate_signals_numba(trend)
            
            # Assign results to dataframe
            result_df['final_upper_band'] = final_upper
            result_df['final_lower_band'] = final_lower
            result_df['trend'] = trend
            result_df['supertrend'] = supertrend
            result_df['trend_changed'] = trend_changed
            result_df['supertrend_signal'] = signals
            
            # Store the upper and lower bands
            result_df['supertrend_upper'] = result_df['final_upper_band']
            result_df['supertrend_lower'] = result_df['final_lower_band']
            
            # Clean up intermediate columns
            result_df.drop(['tr', 'atr', 'median_price', 'basic_upper_band', 
                          'basic_lower_band', 'final_upper_band', 'final_lower_band'], 
                         axis=1, inplace=True)
            
            return result_df
            
        except Exception as e:
            self._log('error', f"Numba calculation error: {str(e)}")
            self._log('info', "Falling back to CPU implementation")
            return self._calculate_cpu(df, atr_length, factor, buffer_multiplier)
    
    def _calculate_final_bands(self, basic_band: pd.Series, 
                             close: pd.Series, 
                             band_type: str,
                             buffer_multiplier: float = 0) -> pd.Series:
        """
        Calculate final bands with logical conditions and buffers
        """
        final_band = pd.Series(0.0, index=basic_band.index)
        
        # First value initialization
        final_band.iloc[0] = basic_band.iloc[0]
        
        # Process the series
        for i in range(1, len(basic_band)):
            if band_type == 'upper':
                if ((basic_band.iloc[i] < final_band.iloc[i-1]) or 
                    (close.iloc[i-1] > final_band.iloc[i-1])):
                    final_band.iloc[i] = basic_band.iloc[i]
                else:
                    final_band.iloc[i] = final_band.iloc[i-1]
                    
                # Apply buffer when switching trends
                if buffer_multiplier > 0 and close.iloc[i-1] > final_band.iloc[i-1]:
                    buffer_val = buffer_multiplier * (basic_band.iloc[i] - basic_band.iloc[i] + 
                                                     (basic_band.iloc[i] - basic_band.iloc[0]) / i)
                    final_band.iloc[i] = final_band.iloc[i] + buffer_val
            else:  # Lower band
                if ((basic_band.iloc[i] > final_band.iloc[i-1]) or 
                    (close.iloc[i-1] < final_band.iloc[i-1])):
                    final_band.iloc[i] = basic_band.iloc[i]
                else:
                    final_band.iloc[i] = final_band.iloc[i-1]
                    
                # Apply buffer when switching trends
                if buffer_multiplier > 0 and close.iloc[i-1] < final_band.iloc[i-1]:
                    buffer_val = buffer_multiplier * (basic_band.iloc[i] - basic_band.iloc[i] + 
                                                     (basic_band.iloc[0] - basic_band.iloc[i]) / i)
                    final_band.iloc[i] = final_band.iloc[i] - buffer_val
        
        return final_band
    
    def _calculate_trend(self, close: pd.Series, 
                       upper_band: pd.Series, 
                       lower_band: pd.Series) -> pd.Series:
        """
        Calculate trend direction based on price and bands
        """
        trend = pd.Series(0, index=close.index)
        
        # First value initialization based on first candle
        if close.iloc[0] > upper_band.iloc[0]:
            trend.iloc[0] = 1
        elif close.iloc[0] < lower_band.iloc[0]:
            trend.iloc[0] = -1
        
        # Process remaining values
        for i in range(1, len(close)):
            if close.iloc[i] > upper_band.iloc[i-1]:
                trend.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i-1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1]
        
        return trend
    
    def _calculate_supertrend_signals(self, df: pd.DataFrame) -> None:
        """
        Calculate SuperTrend, trend changes and signals
        """
        # Calculate SuperTrend line
        df['supertrend'] = 0.0
        for i in range(len(df)):
            if df['trend'].iloc[i] == 1:
                df.loc[df.index[i], 'supertrend'] = df['final_lower_band'].iloc[i]
            else:
                df.loc[df.index[i], 'supertrend'] = df['final_upper_band'].iloc[i]
        
        # Detect trend changes
        df['trend_changed'] = df['trend'] != df['trend'].shift(1)
        
        # Generate signals
        df['supertrend_signal'] = 0
        df.loc[df['trend_changed'] & (df['trend'] == 1), 'supertrend_signal'] = 1     # Buy signals
        df.loc[df['trend_changed'] & (df['trend'] == -1), 'supertrend_signal'] = -1   # Sell signals
        
        # Store the upper and lower bands
        df['supertrend_upper'] = df['final_upper_band']
        df['supertrend_lower'] = df['final_lower_band']
    
    def _ensure_gpu_context(self):
        """
        Create or reuse GPU context for calculations
        """
        if self._gpu_context is None and HAS_CUDF:
            try:
                import cudf
                # Simple test to initialize context
                test_df = cudf.DataFrame({'a': [1, 2, 3]})
                test_result = test_df['a'].sum()
                self._gpu_context = True
                self._log('debug', "GPU context initialized successfully")
            except Exception as e:
                self._log('error', f"Failed to initialize GPU context: {str(e)}")
                self._gpu_context = False
    
    def cleanup_resources(self):
        """
        Clean up GPU resources if used
        """
        if HAS_CUDF and self._gpu_context:
            try:
                import cudf
                import rmm
                
                # Clear any cached allocations
                if hasattr(rmm, 'reinitialize'):
                    self._log('debug', "Cleaning up GPU memory with RMM reinitialize")
                    rmm.reinitialize()
                    
                self._log('info', "GPU resources cleaned up")
            except Exception as e:
                self._log('warning', f"Error cleaning up GPU resources: {str(e)}")   



# ==============================================================================
# BACKTESTING ENGINE
# ==============================================================================

class Trade:
    """Represents a single trade with relevant metrics and information"""
    
    def __init__(self, 
                 trade_id: int,
                 entry_time: pd.Timestamp,
                 entry_price: float,
                 position_type: str,  # 'long' or 'short'
                 entry_signal: str,   # 'buy_signal', 'sell_signal', etc.
                 size: float = 1.0):
        
        self.trade_id = trade_id
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.position_type = position_type.lower()
        self.entry_signal = entry_signal
        self.size = size
        
        # Exit information (to be filled later)
        self.exit_time = None
        self.exit_price = None
        self.exit_signal = None
        self.is_open = True
        self.is_stopped_out = False
        self.is_time_stopped = False
        
        # Performance metrics (to be calculated on exit)
        self.profit = 0.0
        self.profit_pct = 0.0
        self.duration_hours = 0.0
        self.candle_count = 0
        self.risk_reward_ratio = 0.0
        
        # Risk management
        self.stop_price = None
        self.target_price = None
    
    def set_stop_loss(self, stop_price: float):
        """Set stop loss price for the trade"""
        self.stop_price = stop_price
    
    def set_take_profit(self, target_price: float):
        """Set take profit target for the trade"""
        self.target_price = target_price
    
    def close_trade(self, exit_time: pd.Timestamp, exit_price: float, exit_signal: str):
        """Close the trade and calculate performance metrics"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_signal = exit_signal
        self.is_open = False
        
        # Calculate profit
        if self.position_type == 'long':
            raw_profit = exit_price - self.entry_price
        else:  # short
            raw_profit = self.entry_price - exit_price
            
        self.profit = raw_profit * self.size
        self.profit_pct = (raw_profit / self.entry_price) * 100.0
        
        # Calculate duration
        self.duration_hours = (exit_time - self.entry_time).total_seconds() / 3600.0
        self.candle_count = int((exit_time - self.entry_time).total_seconds() / 300)  # Assuming 5-min candles
        
        # Calculate R:R ratio if stop was set
        if self.stop_price is not None:
            if self.position_type == 'long':
                risk = self.entry_price - self.stop_price
                reward = exit_price - self.entry_price
            else:  # short
                risk = self.stop_price - self.entry_price
                reward = self.entry_price - exit_price
                
            self.risk_reward_ratio = abs(reward / risk) if risk != 0 else 0.0
    
    def to_dict(self) -> dict:
        """Convert trade to dictionary for serialization"""
        return {
            'trade_id': self.trade_id,
            'entry_time': self.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
            'entry_price': self.entry_price,
            'exit_time': self.exit_time.strftime('%Y-%m-%d %H:%M:%S') if self.exit_time else None,
            'exit_price': self.exit_price,
            'position_type': self.position_type,
            'entry_signal': self.entry_signal,
            'exit_signal': self.exit_signal,
            'is_open': self.is_open,
            'is_stopped_out': self.is_stopped_out,
            'is_time_stopped': self.is_time_stopped,
            'profit': self.profit,
            'profit_pct': self.profit_pct,
            'duration_hours': self.duration_hours,
            'candle_count': self.candle_count,
            'risk_reward_ratio': self.risk_reward_ratio,
        }


class TradeProcessor:
    """
    Processes trades based on signals generated by indicators
    """
    
    def __init__(self, log_manager: LogManager = None):
        self.log = log_manager
        self.current_utc = "2025-06-24 12:59:23"  # Updated timestamp
        self.current_user = "arullr001"           # Updated username
    
    def _log(self, level: str, message: str):
        """Log message if log_manager is available"""
        if self.log:
            if level == 'info':
                self.log.info(message)
            elif level == 'debug':
                self.log.debug(message)
            elif level == 'warning':
                self.log.warning(message)
            elif level == 'error':
                self.log.error(message)
    
    def process_supertrend_signals(self, df: pd.DataFrame, hard_stop_distance: float = None,
                                  time_exit_hours: float = 48.0) -> List[Trade]:
        """
        Process SuperTrend signals to generate trades
        
        Args:
            df: DataFrame with SuperTrend signals
            hard_stop_distance: Fixed stop loss distance (optional)
            time_exit_hours: Exit trade after this many hours (default 48 hours)
            
        Returns:
            List of Trade objects
        """
        self._log('info', f"Processing SuperTrend signals with stop={hard_stop_distance}, time_exit={time_exit_hours}h")
        
        # Ensure required columns exist
        required_cols = ['buy_signal', 'sell_signal', 'supertrend', 'direction']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            self._log('error', f"Missing required columns for signal processing: {missing}")
            raise ValueError(f"Missing required columns: {missing}")
        
        trades = []
        trade_id = 0
        in_position = False
        current_trade = None
        
        # Process each candle
        for i in range(1, len(df)):
            current_time = df.index[i]
            current_price = df['close'].iloc[i]
            
            # Check for trade exit if in position
            if in_position:
                exit_signal = None
                
                # Check for trend reversal
                if current_trade.position_type == 'long' and df['direction'].iloc[i] > 0:
                    exit_signal = "trend_reversal"
                elif current_trade.position_type == 'short' and df['direction'].iloc[i] < 0:
                    exit_signal = "trend_reversal"
                
                # Check for stop loss hit
                if hard_stop_distance is not None:
                    if current_trade.position_type == 'long' and current_price <= (current_trade.entry_price - hard_stop_distance):
                        exit_signal = "stop_loss"
                        current_trade.is_stopped_out = True
                    elif current_trade.position_type == 'short' and current_price >= (current_trade.entry_price + hard_stop_distance):
                        exit_signal = "stop_loss"
                        current_trade.is_stopped_out = True
                
                # Check for time-based exit
                hours_in_trade = (current_time - current_trade.entry_time).total_seconds() / 3600
                if hours_in_trade >= time_exit_hours:
                    exit_signal = "time_exit"
                    current_trade.is_time_stopped = True
                
                # Exit the trade if any exit condition is met
                if exit_signal:
                    current_trade.close_trade(current_time, current_price, exit_signal)
                    self._log('debug', f"Closed {current_trade.position_type} trade #{current_trade.trade_id} with signal {exit_signal}, profit: {current_trade.profit_pct:.2f}%")
                    in_position = False
                    current_trade = None
            
            # Check for new trade entry if not in position
            if not in_position:
                if df['buy_signal'].iloc[i]:
                    trade_id += 1
                    current_trade = Trade(
                        trade_id=trade_id,
                        entry_time=current_time,
                        entry_price=current_price,
                        position_type='long',
                        entry_signal='buy_signal'
                    )
                    
                    # Set stop loss if specified
                    if hard_stop_distance is not None:
                        current_trade.set_stop_loss(current_price - hard_stop_distance)
                    
                    trades.append(current_trade)
                    in_position = True
                    self._log('debug', f"Opened long trade #{trade_id} at {current_price}")
                    
                elif df['sell_signal'].iloc[i]:
                    trade_id += 1
                    current_trade = Trade(
                        trade_id=trade_id,
                        entry_time=current_time,
                        entry_price=current_price,
                        position_type='short',
                        entry_signal='sell_signal'
                    )
                    
                    # Set stop loss if specified
                    if hard_stop_distance is not None:
                        current_trade.set_stop_loss(current_price + hard_stop_distance)
                    
                    trades.append(current_trade)
                    in_position = True
                    self._log('debug', f"Opened short trade #{trade_id} at {current_price}")
        
        # Close any open trade at the end of the data
        if in_position and current_trade is not None:
            current_trade.close_trade(df.index[-1], df['close'].iloc[-1], "end_of_data")
            self._log('debug', f"Closed {current_trade.position_type} trade #{current_trade.trade_id} at end of data, profit: {current_trade.profit_pct:.2f}%")
        
        # Log summary
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.profit > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_profit = sum(t.profit_pct for t in trades)
        
        self._log('info', f"Processed {total_trades} trades with {winning_trades} winners ({win_rate:.2%})")
        self._log('info', f"Total profit: {total_profit:.2f}%, Average per trade: {total_profit/total_trades:.2f}% if {total_trades} > 0 else 0")
        
        return trades


class PerformanceCalculator:
    """
    Calculates performance metrics for a backtest
    """
    
    def __init__(self, log_manager: LogManager = None):
        self.log = log_manager
    
    def _log(self, level: str, message: str):
        """Log message if log_manager is available"""
        if self.log:
            if level == 'info':
                self.log.info(message)
            elif level == 'debug':
                self.log.debug(message)
            elif level == 'warning':
                self.log.warning(message)
            elif level == 'error':
                self.log.error(message)
    
    def calculate_performance(self, trades: List[Trade], initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Calculate performance metrics from a list of trades
        
        Args:
            trades: List of Trade objects
            initial_capital: Initial capital for calculating returns
            
        Returns:
            Dictionary of performance metrics
        """
        if not trades:
            self._log('warning', "No trades to calculate performance")
            return {
                'trade_count': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_profit': 0,
                'total_profit_pct': 0,
                'equity_curve': pd.DataFrame(),
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'risk_adjusted_return': 0
            }
        
        self._log('info', f"Calculating performance for {len(trades)} trades")
        
        # Extract basic statistics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.profit > 0)
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit metrics
        gross_profit = sum(t.profit for t in trades if t.profit > 0)
        gross_loss = abs(sum(t.profit for t in trades if t.profit < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        total_profit = sum(t.profit for t in trades)
        total_profit_pct = sum(t.profit_pct for t in trades)
        avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
        avg_profit_pct_per_trade = total_profit_pct / total_trades if total_trades > 0 else 0
        
        # Calculate win/loss streak metrics
        results = [1 if t.profit > 0 else 0 for t in trades]
        streak_metrics = self._calculate_streaks(results)
        
        # Calculate drawdown and time-based metrics
        equity_curve = self._calculate_equity_curve(trades, initial_capital)
        max_drawdown, max_drawdown_duration = self._calculate_drawdown(equity_curve)
        
        # Calculate risk-adjusted return metrics
        if len(equity_curve) > 1:
            returns = equity_curve['return_pct'].values
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(total_profit_pct, max_drawdown)
            risk_adjusted_return = (total_profit_pct / 100) / (max_drawdown / 100) if max_drawdown > 0 else float('inf')
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            calmar_ratio = 0
            risk_adjusted_return = 0
        
        # Calculate time metrics
        avg_trade_duration = np.mean([t.duration_hours for t in trades])
        total_market_time_hours = sum(t.duration_hours for t in trades)
        
        # Calculate R:R ratio metrics
        risk_reward_values = [t.risk_reward_ratio for t in trades if t.risk_reward_ratio > 0]
        avg_risk_reward = np.mean(risk_reward_values) if risk_reward_values else 0
        
        # Calculate trade type statistics
        long_trades = sum(1 for t in trades if t.position_type == 'long')
        short_trades = total_trades - long_trades
        
        long_wins = sum(1 for t in trades if t.position_type == 'long' and t.profit > 0)
        short_wins = sum(1 for t in trades if t.position_type == 'short' and t.profit > 0)
        
        long_win_rate = long_wins / long_trades if long_trades > 0 else 0
        short_win_rate = short_wins / short_trades if short_trades > 0 else 0
        
        # Calculate exit reasons statistics
        exit_types = Counter(t.exit_signal for t in trades)
        
        # Generate performance report
        performance = {
            'trade_count': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_profit': total_profit,
            'total_profit_pct': total_profit_pct,
            'average_profit': avg_profit_per_trade,
            'average_profit_pct': avg_profit_pct_per_trade,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'max_consecutive_wins': streak_metrics['max_consecutive_wins'],
            'max_consecutive_losses': streak_metrics['max_consecutive_losses'],
            'largest_winning_trade': max(t.profit for t in trades) if trades else 0,
            'largest_losing_trade': min(t.profit for t in trades) if trades else 0,
            'equity_curve': equity_curve,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'risk_adjusted_return': risk_adjusted_return,
            'avg_trade_duration': avg_trade_duration,
            'total_market_time_hours': total_market_time_hours,
            'avg_risk_reward': avg_risk_reward,
            'long_trades': long_trades,
            'short_trades': short_trades,
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
            'exit_types': dict(exit_types)
        }
        
        self._log('debug', f"Performance metrics: Win rate={win_rate:.2%}, Profit Factor={profit_factor:.2f}, Max DD={max_drawdown:.2f}%")
        
        return performance
    
    def _calculate_streaks(self, results: List[int]) -> Dict[str, int]:
        """Calculate longest winning and losing streaks"""
        # Initialize variables
        current_win_streak = 0
        current_loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        # Iterate through results
        for result in results:
            if result == 1:  # Win
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:  # Loss
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        return {
            'max_consecutive_wins': max_win_streak,
            'max_consecutive_losses': max_loss_streak
        }
    
    def _calculate_equity_curve(self, trades: List[Trade], initial_capital: float) -> pd.DataFrame:
        """Calculate equity curve from trades"""
        # Initialize equity curve
        points = [(pd.Timestamp.min, initial_capital)]  # Start point
        
        # Add each trade to the equity curve
        for trade in trades:
            # Skip open trades
            if trade.is_open:
                continue
                
            # Add entry point
            points.append((trade.entry_time, points[-1][1]))  # Equity at entry
            
            # Add exit point
            equity_after_trade = points[-1][1] + trade.profit
            points.append((trade.exit_time, equity_after_trade))
        
        if len(points) <= 1:
            # No completed trades, return empty dataframe with correct columns
            return pd.DataFrame(columns=['equity', 'drawdown', 'drawdown_pct', 'return_pct'])
        
        # Convert to DataFrame
        equity_curve = pd.DataFrame(points, columns=['timestamp', 'equity'])
        equity_curve.set_index('timestamp', inplace=True)
        
        # Sort by time
        equity_curve = equity_curve.sort_index()
        
        # Calculate returns
        equity_curve['return'] = equity_curve['equity'].diff()
        equity_curve['return_pct'] = equity_curve['equity'].pct_change() * 100
        
        # Calculate drawdown
        equity_curve['peak'] = equity_curve['equity'].cummax()
        equity_curve['drawdown'] = equity_curve['peak'] - equity_curve['equity']
        equity_curve['drawdown_pct'] = (equity_curve['drawdown'] / equity_curve['peak']) * 100
        
        return equity_curve
    
    def _calculate_drawdown(self, equity_curve: pd.DataFrame) -> Tuple[float, float]:
        """Calculate maximum drawdown and its duration"""
        if equity_curve.empty:
            return 0.0, 0.0
        
        # Get max drawdown percentage
        max_dd_pct = equity_curve['drawdown_pct'].max()
        
        # Calculate drawdown duration
        is_in_drawdown = equity_curve['drawdown'] > 0
        drawdown_periods = []
        start_idx = None
        
        for i, in_dd in enumerate(is_in_drawdown):
            if in_dd and start_idx is None:
                start_idx = i
            elif not in_dd and start_idx is not None:
                # Drawdown ended
                duration = (equity_curve.index[i] - equity_curve.index[start_idx]).total_seconds() / 3600  # in hours
                dd_pct = equity_curve['drawdown_pct'].iloc[start_idx:i].max()
                drawdown_periods.append((dd_pct, duration))
                start_idx = None
        
        # Handle if still in drawdown at end
        if start_idx is not None:
            duration = (equity_curve.index[-1] - equity_curve.index[start_idx]).total_seconds() / 3600
            dd_pct = equity_curve['drawdown_pct'].iloc[start_idx:].max()
            drawdown_periods.append((dd_pct, duration))
        
        # Find maximum drawdown duration
        if drawdown_periods:
            # Find the period with the maximum drawdown percentage
            max_dd_period = max(drawdown_periods, key=lambda x: x[0])
            max_dd_duration = max_dd_period[1]
        else:
            max_dd_duration = 0.0
        
        return max_dd_pct, max_dd_duration
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from returns"""
        if len(returns) < 2:
            return 0.0
            
        mean_return = np.mean(returns)
        std_dev = np.std(returns, ddof=1)
        
        if std_dev == 0:
            return 0.0
            
        return (mean_return - risk_free_rate) / std_dev
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio from returns (using only downside deviation)"""
        if len(returns) < 2:
            return 0.0
            
        mean_return = np.mean(returns)
        
        # Calculate downside deviation (only negative returns)
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')  # No negative returns
            
        downside_deviation = np.std(negative_returns, ddof=1)
        
        if downside_deviation == 0:
            return 0.0
            
        return (mean_return - risk_free_rate) / downside_deviation
    
    def _calculate_calmar_ratio(self, total_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio (return / max drawdown)"""
        if max_drawdown == 0:
            return 0.0
            
        return total_return / max_drawdown


class MonteCarloSimulation:
    """
    Performs Monte Carlo simulation on a set of trades to assess strategy robustness
    """
    
    def __init__(self, log_manager: LogManager = None):
        self.log = log_manager
        self.current_utc = "2025-06-24 12:59:23"  # Updated timestamp
        self.current_user = "arullr001"           # Updated username
        self.performance_calculator = PerformanceCalculator(log_manager)
    
    def _log(self, level: str, message: str):
        """Log message if log_manager is available"""
        if self.log:
            if level == 'info':
                self.log.info(message)
            elif level == 'debug':
                self.log.debug(message)
            elif level == 'warning':
                self.log.warning(message)
            elif level == 'error':
                self.log.error(message)
    
    def run_simulation(self, trades: List[Trade], num_simulations: int = 1000, 
                      initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation by randomly shuffling trades
        
        Args:
            trades: List of Trade objects
            num_simulations: Number of simulations to run
            initial_capital: Initial capital for each simulation
            
        Returns:
            Dictionary with simulation results
        """
        if not trades:
            self._log('warning', "No trades provided for Monte Carlo simulation")
            return {}
        
        self._log('info', f"Running Monte Carlo simulation with {num_simulations} iterations")
        
        # Create copies of the trades for simulation
        trade_copies = [deepcopy(trades) for _ in range(num_simulations)]
        
        # Shuffle each copy
        for trade_list in trade_copies:
            random.shuffle(trade_list)
        
        # Run simulations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._run_single_simulation, trade_list, i, initial_capital)
                for i, trade_list in enumerate(trade_copies)
            ]
            
            simulation_results = []
            for future in concurrent.futures.as_completed(futures):
                simulation_results.append(future.result())
        
        # Compile results
        final_equities = [result['final_equity'] for result in simulation_results]
        max_drawdowns = [result['max_drawdown'] for result in simulation_results]
        win_rates = [result['win_rate'] for result in simulation_results]
        profit_factors = [result['profit_factor'] for result in simulation_results]
        
        # Calculate statistics
        results = {
            'equity': {
                'mean': np.mean(final_equities),
                'median': np.median(final_equities),
                'min': np.min(final_equities),
                'max': np.max(final_equities),
                'std': np.std(final_equities),
                'percentiles': {
                    '5': np.percentile(final_equities, 5),
                    '25': np.percentile(final_equities, 25),
                    '75': np.percentile(final_equities, 75),
                    '95': np.percentile(final_equities, 95)
                }
            },
            'drawdown': {
                'mean': np.mean(max_drawdowns),
                'median': np.median(max_drawdowns),
                'min': np.min(max_drawdowns),
                'max': np.max(max_drawdowns),
                'std': np.std(max_drawdowns),
                'percentiles': {
                    '5': np.percentile(max_drawdowns, 5),
                    '25': np.percentile(max_drawdowns, 25),
                    '75': np.percentile(max_drawdowns, 75),
                    '95': np.percentile(max_drawdowns, 95)
                }
            },
            'win_rate': {
                'mean': np.mean(win_rates),
                'median': np.median(win_rates),
                'min': np.min(win_rates),
                'max': np.max(win_rates),
                'std': np.std(win_rates)
            },
            'profit_factor': {
                'mean': np.mean(profit_factors),
                'median': np.median(profit_factors),
                'min': np.min(profit_factors),
                'max': np.max(profit_factors),
                'std': np.std(profit_factors)
            },
            'simulation_data': simulation_results
        }
        
        # Calculate probability of profit/loss
        results['probability'] = {
            'profit': sum(1 for e in final_equities if e > initial_capital) / num_simulations,
            'loss': sum(1 for e in final_equities if e < initial_capital) / num_simulations,
            'break_even': sum(1 for e in final_equities if e == initial_capital) / num_simulations
        }
        
        # Success ratio (average profit / average loss)
        results['success_ratio'] = abs(np.mean([r['mean_win'] for r in simulation_results]) / 
                                    np.mean([r['mean_loss'] for r in simulation_results])) if np.mean([r['mean_loss'] for r in simulation_results]) != 0 else float('inf')
        
        self._log('info', f"Monte Carlo simulation completed: Avg Equity={results['equity']['mean']:.2f}, Avg DD={results['drawdown']['mean']:.2f}%")
        
        return results
    
    def _run_single_simulation(self, trade_list: List[Trade], sim_id: int, 
                             initial_capital: float) -> Dict[str, Any]:
        """Run a single simulation"""
        equity = initial_capital
        equity_curve = [equity]
        current_peak = equity
        current_drawdown = 0
        max_drawdown = 0
        
        wins = []
        losses = []
        
        # Process each trade
        for trade in trade_list:
            # Apply trade P&L
            equity += trade.profit
            equity_curve.append(equity)
            
            # Track for statistics
            if trade.profit > 0:
                wins.append(trade.profit)
            else:
                losses.append(trade.profit)
            
            # Update drawdown
            if equity > current_peak:
                current_peak = equity
                current_drawdown = 0
            else:
                current_drawdown = (current_peak - equity) / current_peak * 100
                max_drawdown = max(max_drawdown, current_drawdown)
        
        # Calculate statistics
        win_count = len(wins)
        loss_count = len(losses)
        total_trades = win_count + loss_count
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        mean_win = np.mean(wins) if wins else 0
        mean_loss = np.mean(losses) if losses else 0
        
        return {
            'simulation_id': sim_id,
            'final_equity': equity,
            'return_pct': ((equity / initial_capital) - 1) * 100,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'mean_win': mean_win,
            'mean_loss': mean_loss,
            'equity_curve': equity_curve
        }


class WalkForwardTesting:
    """
    Perform walk-forward testing to evaluate parameter stability over time
    """
    
    def __init__(self, log_manager: LogManager = None):
        self.log = log_manager
        self.current_utc = "2025-06-24 12:59:23"  # Updated timestamp
        self.current_user = "arullr001"           # Updated username
        self.performance_calculator = PerformanceCalculator(log_manager)
    
    def _log(self, level: str, message: str):
        """Log message if log_manager is available"""
        if self.log:
            if level == 'info':
                self.log.info(message)
            elif level == 'debug':
                self.log.debug(message)
            elif level == 'warning':
                self.log.warning(message)
            elif level == 'error':
                self.log.error(message)
    
    def run_walk_forward_test(self, df: pd.DataFrame, 
                            in_sample_size: int, 
                            out_of_sample_size: int,
                            param_ranges: Dict[str, List],
                            optimization_metric: str = 'profit_factor',
                            num_windows: int = None) -> Dict[str, Any]:
        """
        Run walk-forward optimization and testing
        
        Args:
            df: DataFrame with OHLC data
            in_sample_size: Number of candles for in-sample optimization
            out_of_sample_size: Number of candles for out-of-sample testing
            param_ranges: Dictionary with parameter ranges to test
            optimization_metric: Metric to optimize ('profit_factor', 'win_rate', etc.)
            num_windows: Number of walk-forward windows to test (None for all possible)
            
        Returns:
            Dictionary with walk-forward results
        """
        self._log('info', f"Running walk-forward test with {in_sample_size} in-sample and {out_of_sample_size} out-of-sample candles")
        
        # Check if data is sufficient
        if len(df) < (in_sample_size + out_of_sample_size):
            self._log('error', f"Insufficient data for walk-forward testing: {len(df)} candles, need at least {in_sample_size + out_of_sample_size}")
            raise ValueError(f"Insufficient data for walk-forward testing. Need at least {in_sample_size + out_of_sample_size} candles.")
        
        # Calculate number of possible windows
        max_windows = (len(df) - in_sample_size) // out_of_sample_size
        if num_windows is None or num_windows > max_windows:
            num_windows = max_windows
        
        self._log('debug', f"Will run {num_windows} walk-forward windows")
        
        # Create SuperTrend and TradeProcessor instances
        supertrend = SuperTrend(self.log)
        trade_processor = TradeProcessor(self.log)
        
        # Store results for each window
        window_results = []
        
        # Out-of-sample combined results
        all_oos_trades = []
        
        # Run walk-forward windows
        for i in range(num_windows):
            # Calculate window indices
            is_start = i * out_of_sample_size
            is_end = is_start + in_sample_size
            oos_start = is_end
            oos_end = min(oos_start + out_of_sample_size, len(df))
            
            if is_end >= len(df) or oos_start >= len(df) or oos_end <= oos_start:
                self._log('warning', f"Window {i+1}: Invalid indices, skipping")
                continue
            
            # Extract in-sample and out-of-sample data
            is_data = df.iloc[is_start:is_end].copy()
            oos_data = df.iloc[oos_start:oos_end].copy()
            
            self._log('info', f"Window {i+1}: Optimizing on {len(is_data)} candles, testing on {len(oos_data)} candles")
            
            # Optimize parameters on in-sample data
            best_params, is_performance = self._optimize_parameters(
                supertrend,
                trade_processor,
                is_data,
                param_ranges,
                optimization_metric
            )
            
            if not best_params:
                self._log('warning', f"Window {i+1}: No valid parameters found in optimization")
                continue
            
            # Test optimized parameters on out-of-sample data
            oos_result = self._test_parameters(
                supertrend,
                trade_processor,
                oos_data,
                best_params
            )
            
            # Store window results
            window_result = {
                'window': i+1,
                'in_sample': {
                    'start_date': is_data.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                    'end_date': is_data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                    'candle_count': len(is_data),
                    'best_params': best_params,
                    'performance': is_performance
                },
                'out_of_sample': {
                    'start_date': oos_data.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                    'end_date': oos_data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                    'candle_count': len(oos_data),
                    'performance': oos_result['performance'],
                    'trade_count': len(oos_result['trades'])
                }
            }
            
            # Add trades to combined results
            all_oos_trades.extend(oos_result['trades'])
            
            window_results.append(window_result)
            
            self._log('info', f"Window {i+1} completed: In-sample {is_performance[optimization_metric]:.2f}, Out-of-sample {oos_result['performance'][optimization_metric]:.2f}")
        
        # Calculate overall out-of-sample performance
        overall_oos_performance = self.performance_calculator.calculate_performance(all_oos_trades) if all_oos_trades else {}
        
        # Calculate parameter stability
        param_stability = self._analyze_parameter_stability([w['in_sample']['best_params'] for w in window_results])
        
        # Calculate statistical significance
        statistical_significance = self._calculate_statistical_significance(overall_oos_performance, len(all_oos_trades))
        
        # Return combined results
        results = {
            'window_count': len(window_results),
            'windows': window_results,
            'overall_oos_performance': overall_oos_performance,
            'parameter_stability': param_stability,
            'statistical_significance': statistical_significance,
            'optimization_metric': optimization_metric,
            'in_sample_size': in_sample_size,
            'out_of_sample_size': out_of_sample_size,
            'is_robust': self._evaluate_robustness(window_results, overall_oos_performance, param_stability)
        }
        
        self._log('info', f"Walk-forward test completed with {len(window_results)} windows")
        
        return results
    
    def _optimize_parameters(self, supertrend, trade_processor, data: pd.DataFrame, 
                           param_ranges: Dict[str, List], optimization_metric: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Find optimal parameters on in-sample data"""
        self._log('debug', f"Optimizing parameters using {optimization_metric} as target")
        
        # Create parameter combinations
        atr_lengths = param_ranges.get('atr_lengths', [14])
        factors = param_ranges.get('factors', [3.0])
        buffers = param_ranges.get('buffers', [0.3])
        stops = param_ranges.get('stops', [50])
        
        param_combinations = list(itertools.product(atr_lengths, factors, buffers, stops))
        self._log('debug', f"Testing {len(param_combinations)} parameter combinations")
        
        best_score = -float('inf')
        best_params = None
        best_performance = None
        
        # Test each parameter combination
        for atr_length, factor, buffer_multiplier, hard_stop_distance in param_combinations:
            # Calculate SuperTrend
            st_result = supertrend.calculate(data, atr_length, factor, buffer_multiplier)
            
            # Process signals to get trades
            trades = trade_processor.process_supertrend_signals(st_result, hard_stop_distance, 48.0)
            
            if not trades:
                continue
                
            # Calculate performance
            performance = self.performance_calculator.calculate_performance(trades)
            
            # Check if this is the best so far
            score = performance.get(optimization_metric, -float('inf'))
            
            if score > best_score:
                best_score = score
                best_params = {
                    'atr_length': atr_length,
                    'factor': factor,
                    'buffer_multiplier': buffer_multiplier,
                    'hard_stop_distance': hard_stop_distance
                }
                best_performance = performance
        
        return best_params, best_performance
    
    def _test_parameters(self, supertrend, trade_processor, data: pd.DataFrame, 
                       params: Dict[str, float]) -> Dict[str, Any]:
        """Test parameters on data"""
        # Calculate SuperTrend
        st_result = supertrend.calculate(
            data, 
            params['atr_length'], 
            params['factor'], 
            params['buffer_multiplier']
        )
        
        # Process signals to get trades
        trades = trade_processor.process_supertrend_signals(
            st_result, 
            params['hard_stop_distance'], 
            48.0
        )
        
        # Calculate performance
        performance = self.performance_calculator.calculate_performance(trades)
        
        return {'trades': trades, 'performance': performance}
    
    def _analyze_parameter_stability(self, param_sets: List[Dict[str, float]]) -> Dict[str, Any]:
        """Analyze stability of parameters across windows"""
        if not param_sets:
            return {}
            
        # Extract parameter values across windows
        atr_lengths = [p['atr_length'] for p in param_sets]
        factors = [p['factor'] for p in param_sets]
        buffers = [p['buffer_multiplier'] for p in param_sets]
        stops = [p['hard_stop_distance'] for p in param_sets]
        
        # Calculate statistics
        result = {
            'atr_length': {
                'mean': np.mean(atr_lengths),
                'std': np.std(atr_lengths),
                'median': np.median(atr_lengths),
                'min': np.min(atr_lengths),
                'max': np.max(atr_lengths),
                'cv': np.std(atr_lengths) / np.mean(atr_lengths) if np.mean(atr_lengths) > 0 else 0
            },
            'factor': {
                'mean': np.mean(factors),
                'std': np.std(factors),
                'median': np.median(factors),
                'min': np.min(factors),
                'max': np.max(factors),
                'cv': np.std(factors) / np.mean(factors) if np.mean(factors) > 0 else 0
            },
            'buffer_multiplier': {
                'mean': np.mean(buffers),
                'std': np.std(buffers),
                'median': np.median(buffers),
                'min': np.min(buffers),
                'max': np.max(buffers),
                'cv': np.std(buffers) / np.mean(buffers) if np.mean(buffers) > 0 else 0
            },
            'hard_stop_distance': {
                'mean': np.mean(stops),
                'std': np.std(stops),
                'median': np.median(stops),
                'min': np.min(stops),
                'max': np.max(stops),
                'cv': np.std(stops) / np.mean(stops) if np.mean(stops) > 0 else 0
            }
        }
        
        # Calculate overall stability score (using coefficient of variation)
        cv_values = [
            result['atr_length']['cv'],
            result['factor']['cv'],
            result['buffer_multiplier']['cv'],
            result['hard_stop_distance']['cv']
        ]
        
        result['overall_stability'] = 1.0 - (sum(cv_values) / len(cv_values))
        
        return result
    
    def _calculate_statistical_significance(self, performance: Dict[str, Any], 
                                         trade_count: int) -> Dict[str, float]:
        """Calculate statistical significance of trading results"""
        if not performance or trade_count == 0:
            return {}
            
        # Get win rate and number of winning trades
        win_rate = performance.get('win_rate', 0)
        winning_trades = int(win_rate * trade_count)
        
        # Calculate p-value using binomial test (null hypothesis: win_rate = 0.5)
        p_value = 1.0
        try:
            import scipy.stats as stats
            p_value = stats.binom_test(winning_trades, trade_count, p=0.5)
        except ImportError:
            # Approximate p-value using normal approximation to binomial
            if trade_count >= 20:  # Apply only for large enough samples
                z_score = (winning_trades - (trade_count * 0.5)) / np.sqrt(trade_count * 0.5 * 0.5)
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
        
        return {
            'p_value': p_value,
            'significant_95': p_value < 0.05,
            'significant_99': p_value < 0.01,
            'confidence_level': (1.0 - p_value) * 100
        }
    
    def _evaluate_robustness(self, window_results: List[Dict[str, Any]], 
                          overall_performance: Dict[str, Any],
                          param_stability: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate overall robustness of the strategy"""
        if not window_results or not overall_performance:
            return {'is_robust': False, 'reasons': ['Insufficient data']}
            
        # Parameters to consider
        key_metrics = ['profit_factor', 'win_rate', 'max_drawdown', 'sharpe_ratio']
        
        # Count number of profitable out-of-sample windows
        total_windows = len(window_results)
        profitable_windows = sum(1 for w in window_results 
                             if w['out_of_sample']['performance'].get('total_profit', 0) > 0)
        profitable_pct = profitable_windows / total_windows if total_windows > 0 else 0
        
        # Calculate consistency ratio (out-of-sample / in-sample performance)
        consistency_ratios = []
        for w in window_results:
            is_perf = w['in_sample']['performance'].get('profit_factor', 1.0)
            oos_perf = w['out_of_sample']['performance'].get('profit_factor', 0.0)
            
            if is_perf > 0:
                consistency_ratios.append(oos_perf / is_perf)
        
        avg_consistency = np.mean(consistency_ratios) if consistency_ratios else 0
        
        # Get parameter stability
        param_stability_score = param_stability.get('overall_stability', 0)
        
        # Define thresholds for robustness
        robust_thresholds = {
            'profitable_pct': 0.6,  # >60% of windows should be profitable
            'consistency_ratio': 0.7,  # OOS performance should be >70% of IS
            'param_stability': 0.7,  # Parameter stability should be >70%
            'profit_factor': 1.3,  # Overall profit factor should be >1.3
            'win_rate': 0.5,  # Win rate should be >50%
            'trade_count': 20  # Need enough trades for statistical validity
        }
        
        # Check if thresholds are met
        meets_threshold = {
            'profitable_pct': profitable_pct >= robust_thresholds['profitable_pct'],
            'consistency_ratio': avg_consistency >= robust_thresholds['consistency_ratio'],
            'param_stability': param_stability_score >= robust_thresholds['param_stability'],
            'profit_factor': overall_performance.get('profit_factor', 0) >= robust_thresholds['profit_factor'],
            'win_rate': overall_performance.get('win_rate', 0) >= robust_thresholds['win_rate'],
            'trade_count': overall_performance.get('trade_count', 0) >= robust_thresholds['trade_count']
        }
        
        # Get reasons for non-robustness
        reasons = [k for k, v in meets_threshold.items() if not v]
        
        # Compute overall robustness score
        robustness_score = sum(1 for v in meets_threshold.values() if v) / len(meets_threshold)
        
        # Determine if strategy is robust
        is_robust = robustness_score >= 0.8  # At least 80% of criteria should be met
        
        return {
            'is_robust': is_robust,
            'robustness_score': robustness_score,
            'reasons': reasons,
            'profitable_windows_pct': profitable_pct,
            'avg_consistency_ratio': avg_consistency,
            'thresholds': robust_thresholds,
            'meets_threshold': meets_threshold
        }


class Backtester:
    """
    Main backtesting engine for SuperTrend strategy
    """
    
    def __init__(self, log_manager: LogManager = None):
        self.log = log_manager
        self.current_utc = "2025-06-24 12:59:23"  # Updated timestamp
        self.current_user = "arullr001"           # Updated username
        
        # Initialize components
        self.supertrend = SuperTrend(log_manager)
        self.trade_processor = TradeProcessor(log_manager)
        self.performance_calculator = PerformanceCalculator(log_manager)
        self.monte_carlo = MonteCarloSimulation(log_manager)
        self.walk_forward = WalkForwardTesting(log_manager)
    
    def _log(self, level: str, message: str):
        """Log message if log_manager is available"""
        if self.log:
            if level == 'info':
                self.log.info(message)
            elif level == 'debug':
                self.log.debug(message)
            elif level == 'warning':
                self.log.warning(message)
            elif level == 'error':
                self.log.error(message)
    
    def run_backtest(self, df: pd.DataFrame, atr_length: int, factor: float, 
                    buffer_multiplier: float, hard_stop_distance: float,
                    time_exit_hours: float = 48.0) -> Dict[str, Any]:
        """
        Run a complete backtest of the SuperTrend strategy
        
        Args:
            df: DataFrame with OHLC data
            atr_length: Period for ATR calculation
            factor: Multiplier for ATR to set band distance
            buffer_multiplier: Multiplier for setting buffer zones
            hard_stop_distance: Fixed stop loss distance
            time_exit_hours: Exit trade after this many hours
            
        Returns:
            Dictionary with backtest results
        """
        self._log('info', f"Running backtest with ATR={atr_length}, Factor={factor}, Buffer={buffer_multiplier}, Stop={hard_stop_distance}")
        
        start_time = time.time()
        
        # Calculate SuperTrend
        df_st = self.supertrend.calculate(df, atr_length, factor, buffer_multiplier)
        
        # Process signals to get trades
        trades = self.trade_processor.process_supertrend_signals(df_st, hard_stop_distance, time_exit_hours)
        
        # Calculate performance metrics
        performance = self.performance_calculator.calculate_performance(trades)
        
        # Run Monte Carlo simulation if we have enough trades
        monte_carlo_results = {}
        if len(trades) >= 20:
            monte_carlo_results = self.monte_carlo.run_simulation(trades, num_simulations=500)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Compile results
        results = {
            'parameters': {
                'atr_length': atr_length,
                'factor': factor,
                'buffer_multiplier': buffer_multiplier,
                'hard_stop_distance': hard_stop_distance,
                'time_exit_hours': time_exit_hours
            },
            'performance': performance,
            'trades': trades,
            'monte_carlo': monte_carlo_results,
            'metadata': {
                'run_date': self.current_utc,
                'run_by': self.current_user,
                'execution_time_seconds': execution_time,
                'data_period': {
                    'start': df.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                    'end': df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                    'candle_count': len(df)
                }
            }
        }
        
        self._log('info', f"Backtest completed with {len(trades)} trades in {execution_time:.2f} seconds")
        
        return results
    
    def run_optimization(self, df: pd.DataFrame, 
                       param_ranges: Dict[str, List],
                       optimization_metric: str = 'profit_factor',
                       min_trades: int = 10,
                       time_exit_hours: float = 48.0) -> Dict[str, Any]:
        """
        Run parameter optimization for the SuperTrend strategy
        
        Args:
            df: DataFrame with OHLC data
            param_ranges: Dictionary with parameter ranges to test
            optimization_metric: Metric to optimize ('profit_factor', 'win_rate', etc.)
            min_trades: Minimum number of trades required for a valid result
            time_exit_hours: Exit trade after this many hours
            
        Returns:
            Dictionary with optimization results
        """
        self._log('info', f"Running parameter optimization using {optimization_metric} as target metric")
        
        start_time = time.time()
        
        # Extract parameter ranges
        atr_lengths = param_ranges.get('atr_lengths', [14])
        factors = param_ranges.get('factors', [3.0])
        buffers = param_ranges.get('buffers', [0.3])
        stops = param_ranges.get('stops', [50])
        
        # Generate all parameter combinations
        param_combinations = list(itertools.product(atr_lengths, factors, buffers, stops))
        total_combinations = len(param_combinations)
        
        self._log('info', f"Testing {total_combinations} parameter combinations")
        
        # Set up progress tracking
        completed = 0
        valid_results = []
        
        # Process combinations
        for atr_length, factor, buffer_multiplier, hard_stop_distance in param_combinations:
            self._log('debug', f"Testing ATR={atr_length}, Factor={factor}, Buffer={buffer_multiplier}, Stop={hard_stop_distance}")
            
            try:
                # Calculate SuperTrend
                df_st = self.supertrend.calculate(df, atr_length, factor, buffer_multiplier)
                
                # Process signals to get trades
                trades = self.trade_processor.process_supertrend_signals(df_st, hard_stop_distance, time_exit_hours)
                
                # Skip if not enough trades
                if len(trades) < min_trades:
                    self._log('debug', f"Skipping combination - only {len(trades)} trades (minimum {min_trades})")
                    completed += 1
                    continue
                
                # Calculate performance metrics
                performance = self.performance_calculator.calculate_performance(trades)
                
                # Store result
                result = {
                    'parameters': {
                        'atr_length': atr_length,
                        'factor': factor,
                        'buffer_multiplier': buffer_multiplier,
                        'hard_stop_distance': hard_stop_distance
                    },
                    'performance': performance,
                    'trade_count': len(trades)
                }
                
                valid_results.append(result)
                
                # Log progress periodically
                completed += 1
                if completed % max(1, total_combinations // 20) == 0:
                    progress_pct = (completed / total_combinations) * 100
                    elapsed = time.time() - start_time
                    remaining = (elapsed / completed) * (total_combinations - completed) if completed > 0 else 0
                    self._log('info', f"Progress: {completed}/{total_combinations} ({progress_pct:.1f}%), ETA: {format_time_delta(remaining)}")
                
            except Exception as e:
                self._log('error', f"Error processing combination: {str(e)}")
                completed += 1
        
        # Sort results by optimization metric
        if valid_results:
            valid_results.sort(key=lambda x: x['performance'].get(optimization_metric, 0), reverse=True)
        
        execution_time = time.time() - start_time
        
        # Compile optimization results
        results = {
            'best_parameters': valid_results[0]['parameters'] if valid_results else None,
            'best_performance': valid_results[0]['performance'] if valid_results else None,
            'all_results': valid_results,
            'metadata': {
                'run_date': self.current_utc,
                'run_by': self.current_user,
                'execution_time_seconds': execution_time,
                'optimization_metric': optimization_metric,
                'min_trades': min_trades,
                'total_combinations': total_combinations,
                'valid_combinations': len(valid_results),
                'data_period': {
                    'start': df.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                    'end': df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                    'candle_count': len(df)
                }
            }
        }
        
        # Generate parameter distribution analysis
        if valid_results:
            results['parameter_analysis'] = self._analyze_parameter_distribution(valid_results, optimization_metric)
        
        self._log('info', f"Optimization completed: {len(valid_results)} valid combinations out of {total_combinations} in {execution_time:.2f} seconds")
        
        return results
    
    def _analyze_parameter_distribution(self, results: List[Dict], 
                                      optimization_metric: str) -> Dict[str, Any]:
        """Analyze parameter distribution among top-performing combinations"""
        if not results:
            return {}
        
        # Sort by optimization metric
        sorted_results = sorted(results, key=lambda x: x['performance'].get(optimization_metric, 0), reverse=True)
        
        # Take top 10% or at least 5 results
        top_n = max(5, len(sorted_results) // 10)
        top_results = sorted_results[:top_n]
        
        # Extract parameters
        atr_lengths = [r['parameters']['atr_length'] for r in top_results]
        factors = [r['parameters']['factor'] for r in top_results]
        buffers = [r['parameters']['buffer_multiplier'] for r in top_results]
        stops = [r['parameters']['hard_stop_distance'] for r in top_results]
        
        # Calculate statistics
        analysis = {
            'atr_length': {
                'mean': np.mean(atr_lengths),
                'median': np.median(atr_lengths),
                'min': np.min(atr_lengths),
                'max': np.max(atr_lengths),
                'std': np.std(atr_lengths),
                'most_common': max(set(atr_lengths), key=atr_lengths.count)
            },
            'factor': {
                'mean': np.mean(factors),
                'median': np.median(factors),
                'min': np.min(factors),
                'max': np.max(factors),
                'std': np.std(factors),
                'most_common': max(set(factors), key=factors.count)
            },
            'buffer_multiplier': {
                'mean': np.mean(buffers),
                'median': np.median(buffers),
                'min': np.min(buffers),
                'max': np.max(buffers),
                'std': np.std(buffers),
                'most_common': max(set(buffers), key=buffers.count)
            },
            'hard_stop_distance': {
                'mean': np.mean(stops),
                'median': np.median(stops),
                'min': np.min(stops),
                'max': np.max(stops),
                'std': np.std(stops),
                'most_common': max(set(stops), key=stops.count)
            }
        }
        
        # Calculate parameter importance using variance
        variances = {
            'atr_length': np.var(atr_lengths),
            'factor': np.var(factors),
            'buffer_multiplier': np.var(buffers),
            'hard_stop_distance': np.var(stops)
        }
        
        # Normalize to sum to 1
        total_variance = sum(variances.values())
        importance = {k: v / total_variance if total_variance > 0 else 0.25 for k, v in variances.items()}
        
        analysis['parameter_importance'] = importance
        
        # Recommended parameter ranges based on top performers
        analysis['recommended_ranges'] = {
            'atr_length': (int(np.floor(np.min(atr_lengths))), int(np.ceil(np.max(atr_lengths)))),
            'factor': (np.min(factors), np.max(factors)),
            'buffer_multiplier': (np.min(buffers), np.max(buffers)),
            'hard_stop_distance': (int(np.floor(np.min(stops))), int(np.ceil(np.max(stops))))
        }
        
        return analysis
		
		
# ==============================================================================
# OPTIMIZATION METHODS
# ==============================================================================



class GridSearchOptimizer:
    """
    Grid search optimization for parameter tuning
    """
    
    def __init__(self, log_manager: LogManager = None, progress_callback=None):
        self.log = log_manager
        self.current_utc = "2025-06-27 18:59:01"  # Updated timestamp
        self.current_user = "arullr001"           # Updated username
        self.progress_callback = progress_callback  # Add progress callback
        
        # Initialize components
        self.backtester = Backtester(log_manager)
    
    def _log(self, level: str, message: str):
        """Log message if log_manager is available"""
        if self.log:
            if level == 'info':
                self.log.info(message)
            elif level == 'debug':
                self.log.debug(message)
            elif level == 'warning':
                self.log.warning(message)
            elif level == 'error':
                self.log.error(message)
    
    def optimize(self, df: pd.DataFrame, 
                param_ranges: Dict[str, List],
                optimization_metric: str = 'profit_factor',
                min_trades: int = 10,
                time_exit_hours: float = 48.0,
                parallelism: str = 'thread',
                force_implementation: str = None,
                precomputed_patterns: Dict = None) -> Dict[str, Any]:
        """
        Run exhaustive grid search optimization for SuperTrend parameters
        
        Args:
            df: DataFrame with OHLC data
            param_ranges: Dictionary with parameter ranges to test
            optimization_metric: Metric to optimize ('profit_factor', 'win_rate', etc.)
            min_trades: Minimum number of trades required for a valid result
            time_exit_hours: Exit trade after this many hours
            parallelism: Type of parallelism ('thread', 'process', 'dask', 'none')
            force_implementation: Force specific SuperTrend implementation
            precomputed_patterns: Precomputed patterns for GPU optimization
            
        Returns:
            Dictionary with optimization results
        """
        self._log('info', f"Running grid search optimization using {optimization_metric} as target metric")
        
        start_time = time.time()
        
        # Extract parameter ranges
        atr_lengths = param_ranges.get('atr_lengths', [14])
        factors = param_ranges.get('factors', [3.0])
        buffers = param_ranges.get('buffers', [0.3])
        stops = param_ranges.get('stops', [50])
        
        # Generate all parameter combinations
        param_combinations = list(itertools.product(atr_lengths, factors, buffers, stops))
        total_combinations = len(param_combinations)
        
        self._log('info', f"Testing {total_combinations} parameter combinations using {parallelism} parallelism")
        
        # Always force thread parallelism for better GPU utilization
        parallelism = "thread"
        
        # Send initial progress update
        if self.progress_callback:
            self.progress_callback({
                'completed': 0,
                'total': total_combinations,
                'progress_pct': 0,
                'elapsed': 0,
                'remaining': 0,
                'valid_results': 0,
                'top_combinations': []
            })
        
        
        # Choose execution method
        if parallelism == 'thread':
            results = self._run_threaded(
                df, param_combinations, optimization_metric, 
                min_trades, time_exit_hours, force_implementation, precomputed_patterns
            )
        elif parallelism == 'process':
            results = self._run_multiprocess(df, param_combinations, optimization_metric, min_trades, time_exit_hours)
        elif parallelism == 'dask' and HAS_DASK:
            results = self._run_dask(df, param_combinations, optimization_metric, min_trades, time_exit_hours)
        else:
            if parallelism == 'dask' and not HAS_DASK:
                self._log('warning', "Dask not available, falling back to sequential execution")
            results = self._run_sequential(df, param_combinations, optimization_metric, min_trades, time_exit_hours)
        
        # Compile results
        valid_results = [r for r in results if r]  # Filter out None results
        
        execution_time = time.time() - start_time
        
        # Sort results by optimization metric
        if valid_results:
            valid_results.sort(key=lambda x: x['performance'].get(optimization_metric, 0), reverse=True)
            best_result = valid_results[0]
        else:
            best_result = None
        
        # Compile optimization results
        optimization_results = {
            'best_parameters': best_result['parameters'] if best_result else None,
            'best_performance': best_result['performance'] if best_result else None,
            'all_results': valid_results,
            'metadata': {
                'run_date': self.current_utc,
                'run_by': self.current_user,
                'execution_time_seconds': execution_time,
                'optimization_metric': optimization_metric,
                'min_trades': min_trades,
                'total_combinations': total_combinations,
                'valid_combinations': len(valid_results),
                'parallelism': parallelism,
                'data_period': {
                    'start': df.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                    'end': df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                    'candle_count': len(df)
                }
            }
        }
        
        # Generate parameter distribution analysis
        if valid_results:
            parameter_analyzer = ParameterAnalyzer(self.log)
            optimization_results['parameter_analysis'] = parameter_analyzer.analyze_parameter_distribution(
                valid_results, optimization_metric
            )
        
        # Final progress update
        if self.progress_callback:
            self.progress_callback({
                'completed': total_combinations,
                'total': total_combinations,
                'progress_pct': 100,
                'elapsed': execution_time,
                'remaining': 0,
                'valid_results': len(valid_results),
                'top_combinations': self._get_top_combinations(valid_results, optimization_metric)
            })
        
        self._log('info', f"Grid search optimization completed: {len(valid_results)} valid combinations out of {total_combinations} in {execution_time:.2f} seconds")
        
        # Create a compatible result object with the expected .x attribute
        from types import SimpleNamespace
        result_obj = SimpleNamespace()
    
        # Set the .x attribute to the best parameters found
        if valid_results and best_result:
            result_obj.x = [
                best_result['parameters'].get('atr_length', 14),
                best_result['parameters'].get('factor', 3.0),
                best_result['parameters'].get('buffer_multiplier', 0.3),
                best_result['parameters'].get('hard_stop_distance', 50)
            ]
            result_obj.fun = -best_result['performance'].get(optimization_metric, 0)
        else:
            # Default values if no valid results
            result_obj.x = [14, 3.0, 0.3, 50]  # Default parameters
            result_obj.fun = 0
        
        return result_obj  # Return the compatible object instead of the results dictionary
        
        
        return optimization_results
        
        
        
    
    def _get_top_combinations(self, results, optimization_metric, top_n=5):
        """Extract top combinations from results for progress updates"""
        if not results:
            return []
            
        # Sort by optimization metric
        sorted_results = sorted(
            results, 
            key=lambda x: x['performance'].get(optimization_metric, 0), 
            reverse=True
        )[:top_n]
        
        # Format for display
        top_combos = []
        for result in sorted_results:
            params = result['parameters']
            perf = result['performance']
            
            top_combos.append({
                'parameters': params,
                'performance': {
                    'win_rate': perf.get('win_rate', 0),
                    'profit_factor': perf.get('profit_factor', 0),
                    'total_profit_pct': perf.get('total_profit_pct', 0),
                    optimization_metric: perf.get(optimization_metric, 0)
                }
            })
        
        return top_combos
    
    def _run_sequential(self, df: pd.DataFrame, param_combinations: List[Tuple], 
                      optimization_metric: str, min_trades: int, time_exit_hours: float) -> List[Dict]:
        """Run optimization sequentially"""
        self._log('info', "Running optimization sequentially")
        
        results = []
        total = len(param_combinations)
        valid_count = 0
        start_time = time.time()
        
        for i, (atr_length, factor, buffer, stop) in enumerate(param_combinations):
            if i % 10 == 0:  # Log progress every 10 combinations
                self._log('debug', f"Progress: {i}/{total} ({i/total*100:.1f}%)")
            
            result = self._evaluate_combination(df, atr_length, factor, buffer, stop, min_trades, time_exit_hours)
            if result:
                results.append(result)
                valid_count += 1
            
            # Send progress update every 5% or every 10 combinations, whichever is more frequent
            if i % max(10, total // 20) == 0 and self.progress_callback:
                elapsed = time.time() - start_time
                progress_pct = (i+1) / total * 100
                remaining = (elapsed / (i+1)) * (total - (i+1)) if i > 0 else 0
                
                self.progress_callback({
                    'completed': i+1,
                    'total': total,
                    'progress_pct': progress_pct,
                    'elapsed': elapsed,
                    'remaining': remaining,
                    'valid_results': valid_count,
                    'top_combinations': self._get_top_combinations(results, optimization_metric)
                })
        
        return results

    def determine_batch_size(self, total_combinations: int) -> int:
        """Dynamically determine optimal batch size based on system characteristics"""
        # Base sizing on available cores and total workload
        cores = max(1, CPU_THREADS)
        
        # For small workloads, use smaller batches for more responsive feedback
        if total_combinations <= 100:
            return max(1, total_combinations // 20)
            
        # For medium workloads, create approximately 3-4x as many batches as cores
        if total_combinations <= 1000:
            optimal_batches = cores * 3
            return max(5, total_combinations // optimal_batches)
            
        # For larger workloads, create approximately 2-3x as many batches as cores
        optimal_batches = cores * 2
        batch_size = max(10, total_combinations // optimal_batches)
        
        # Cap batch size to ensure responsiveness and prevent memory issues
        return min(batch_size, 50)
    
    def _run_threaded(self, df: pd.DataFrame, param_combinations: List[Tuple], 
                    optimization_metric: str, min_trades: int, time_exit_hours: float,
                    force_implementation: str = None, precomputed_patterns: Dict = None) -> List[Dict]:
        """Run optimization using thread parallelism with optimized GPU usage"""
        self._log('info', f"Running optimization with ThreadPoolExecutor using {CPU_THREADS} workers")

        # Create shared SuperTrend calculator once for all threads
        shared_supertrend = SuperTrend(self.log)
        
        results = []
        total = len(param_combinations)
        completed = 0
        valid_count = 0
        start_time = time.time()

        # Thread-safe counter for progress updates
        counter_lock = threading.Lock()
        
        # Determine optimal batch size
        batch_size = self.determine_batch_size(total)
        self._log('info', f"Using dynamic batch size of {batch_size} for {total} combinations")
        print(f"Dynamic batch size: {batch_size} (UTC: {self.current_utc}, User: {self.current_user})")
        
        # Batch param combinations
        batched_params = [param_combinations[i:i+batch_size] for i in range(0, len(param_combinations), batch_size)]
        
        def process_batch(batch_idx, batch):
            nonlocal completed, valid_count
            batch_results = []
            
            # Process each combination individually with progress updates
            for params in batch:
                atr_length, factor, buffer, stop = params
                
                # Process with shared SuperTrend instance and GPU implementation
                result = self._evaluate_combination(
                    df, atr_length, factor, buffer, stop, 
                    min_trades, time_exit_hours, 
                    shared_supertrend, force_implementation, precomputed_patterns
                )
                
                # Update progress after each combination
                with counter_lock:
                    completed += 1
                    if result:
                        batch_results.append(result)
                        valid_count += 1
                    
                    # Update progress every few combinations to reduce overhead
                    if completed % 5 == 0 and self.progress_callback:
                        elapsed = time.time() - start_time
                        progress_pct = completed / total * 100
                        remaining = (elapsed / completed) * (total - completed) if completed > 0 else 0
                        
                        # Only get top combinations occasionally to reduce overhead
                        show_top_combos = (completed % 20 == 0)
                        
                        top_combos = []
                        if show_top_combos:
                            # Get current top combinations
                            all_results_snapshot = list(results)
                            all_results_snapshot.extend(batch_results)
                            top_combos = self._get_top_combinations(all_results_snapshot, optimization_metric)
                        
                        self.progress_callback({
                            'completed': completed,
                            'total': total,
                            'progress_pct': progress_pct,
                            'elapsed': elapsed,
                            'remaining': remaining,
                            'valid_results': valid_count,
                            'top_combinations': top_combos
                        })
            
            return batch_results

        # Create a thread pool with optimal thread count for your system
        max_workers = min(16, CPU_THREADS)  # Limit to reasonable number

        # Send initial progress update
        if self.progress_callback:
            self.progress_callback({
                'completed': 0,
                'total': total,
                'progress_pct': 0,
                'elapsed': 0,
                'remaining': 0,
                'valid_results': 0,
                'top_combinations': []
            })

        # Process batches in the thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches to be processed
            futures = {executor.submit(process_batch, i, batch): i for i, batch in enumerate(batched_params)}
        
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                batch_results = future.result()
                results.extend(batch_results)

        return results
    
    def _run_multiprocess(self, df: pd.DataFrame, param_combinations: List[Tuple], 
                        optimization_metric: str, min_trades: int, time_exit_hours: float) -> List[Dict]:
        """Run optimization using process parallelism"""
        self._log('info', f"Running optimization with ProcessPoolExecutor using {CPU_THREADS} workers")
        
        # Change to use thread parallelism instead if having issues with process parallelism
        self._log('info', "Falling back to thread parallelism due to potential multiprocessing issues")
        return self._run_threaded(df, param_combinations, optimization_metric, min_trades, time_exit_hours)
    
    def _run_dask(self, df: pd.DataFrame, param_combinations: List[Tuple], 
                optimization_metric: str, min_trades: int, time_exit_hours: float) -> List[Dict]:
        """Run optimization using Dask for distributed computing"""
        if not HAS_DASK:
            self._log('error', "Dask not available for distributed computing")
            return self._run_sequential(df, param_combinations, optimization_metric, min_trades, time_exit_hours)
        
        # Fallback to threaded implementation as it's more reliable with GPU
        self._log('info', "Using threaded implementation instead of Dask for better GPU coordination")
        return self._run_threaded(df, param_combinations, optimization_metric, min_trades, time_exit_hours)
    
    def _evaluate_combination(self, df: pd.DataFrame, atr_length: int, factor: float, 
                           buffer_multiplier: float, hard_stop_distance: float,
                           min_trades: int, time_exit_hours: float,
                           shared_supertrend=None, force_implementation=None,
                           precomputed_patterns=None) -> Dict[str, Any]:
        """
        Evaluate a single parameter combination with GPU optimization
        
        Returns a dictionary with parameters and performance metrics, or None if invalid
        """
        start_time = time.time()
        MAX_EVAL_TIME = 300  # 5 minutes max per evaluation
        
        try:
            # Use shared SuperTrend instance if provided
            supertrend_instance = shared_supertrend or SuperTrend(self.log)
            
            # Calculate SuperTrend with forced GPU implementation
            df_st = supertrend_instance.calculate(
                df,
                atr_length=atr_length,
                factor=factor,
                buffer_multiplier=buffer_multiplier,
                force_implementation=force_implementation or 'gpu',
                precomputed_patterns=precomputed_patterns
            )
            
            # Process signals to get trades
            trade_processor = TradeProcessor(self.log)
            trades = trade_processor.process_supertrend_signals(
                df_st, 
                hard_stop_distance, 
                time_exit_hours
            )
            
            # Check if the result meets minimum trade criteria
            if len(trades) < min_trades:
                return None
            
            # Calculate performance metrics
            performance_calculator = PerformanceCalculator(self.log)
            performance = performance_calculator.calculate_performance(trades)
            
            # Return formatted result
            return {
                'parameters': {
                    'atr_length': atr_length,
                    'factor': factor,
                    'buffer_multiplier': buffer_multiplier,
                    'hard_stop_distance': hard_stop_distance
                },
                'performance': performance,
                'trade_count': len(trades)
            }
            
        except Exception as e:
            self._log('error', f"Error evaluating combination ({atr_length}, {factor}, {buffer_multiplier}, {hard_stop_distance}): {str(e)}")
            return None
    
    @staticmethod
    def _evaluate_combination_mp(df_path: str, atr_length: int, factor: float, 
                               buffer_multiplier: float, hard_stop_distance: float,
                               min_trades: int, time_exit_hours: float) -> Dict[str, Any]:
        """
        Multiprocessing-safe version of evaluate_combination that loads data from file
        """
        try:
            # Import locally to avoid import issues in child processes
            import pandas as pd
            from supertrend_backtester.models import SuperTrend, TradeProcessor, PerformanceCalculator
            
            # Load dataframe from path
            df = pd.read_parquet(df_path)
            
            # Create components
            supertrend = SuperTrend()
            trade_processor = TradeProcessor()
            performance_calculator = PerformanceCalculator()
            
            # Calculate supertrend with GPU
            df_st = supertrend.calculate(df, atr_length, factor, buffer_multiplier, force_implementation='gpu')
            
            # Process signals to get trades
            trades = trade_processor.process_supertrend_signals(df_st, hard_stop_distance, time_exit_hours)
            
            # Check if meets minimum trades requirement
            if len(trades) < min_trades:
                return None
                
            # Calculate performance metrics
            performance = performance_calculator.calculate_performance(trades)
            
            # Return formatted result
            return {
                'parameters': {
                    'atr_length': atr_length,
                    'factor': factor,
                    'buffer_multiplier': buffer_multiplier,
                    'hard_stop_distance': hard_stop_distance
                },
                'performance': performance,
                'trade_count': len(trades)
            }
            
        except Exception as e:
            print(f"Error in MP evaluation: {str(e)}")
            return None


class BayesianOptimizer:
    """
    Bayesian optimization for efficient parameter tuning using scikit-optimize
    """
    
    def __init__(self, log_manager: LogManager = None, progress_callback=None):
        self.log = log_manager
        self.current_utc = "2025-06-27 19:02:03"  # Updated timestamp
        self.current_user = "arullr001"           # Updated username
        self.progress_callback = progress_callback  # Add progress callback
        
        # Initialize components
        self.supertrend = None  # Will initialize once for all evaluations
        self.trade_processor = None
        self.performance_calculator = None
        
        # Check if skopt is available
        if not HAS_SKOPT:
            self._log('warning', "scikit-optimize (skopt) not available, Bayesian optimization will not work")
    
    def _log(self, level: str, message: str):
        """Log message if log_manager is available"""
        if self.log:
            if level == 'info':
                self.log.info(message)
            elif level == 'debug':
                self.log.debug(message)
            elif level == 'warning':
                self.log.warning(message)
            elif level == 'error':
                self.log.error(message)
    
    def _evaluate_batch(self, params_batch):
        """Evaluate a batch of parameters using thread parallelism"""
        results = []
    
        # Use thread pooling with reasonable number of workers
        max_workers = min(16, CPU_THREADS)
    
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map the evaluation function to all parameters in the batch
            future_to_params = {executor.submit(self._evaluate_parameters, params): params 
                               for params in params_batch}
        
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_params):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    self._log('error', f"Error in batch evaluation: {str(e)}")
                    
        return results
    
    def optimize(self, df: pd.DataFrame, 
                param_ranges: Dict[str, Tuple[float, float]],
                optimization_metric: str = 'profit_factor',
                min_trades: int = 10,
                time_exit_hours: float = 48.0,
                n_calls: int = 50,
                n_random_starts: int = 10,
                force_implementation: str = 'gpu') -> Dict[str, Any]:
        """
        Run Bayesian optimization for SuperTrend parameters
        
        Args:
            df: DataFrame with OHLC data
            param_ranges: Dictionary with parameter ranges as (min, max) tuples
            optimization_metric: Metric to optimize ('profit_factor', 'win_rate', etc.)
            min_trades: Minimum number of trades required for a valid result
            time_exit_hours: Exit trade after this many hours
            n_calls: Number of total optimization steps
            n_random_starts: Number of random initial points
            force_implementation: Force specific ST implementation (default: 'gpu')
            
        Returns:
            Dictionary with optimization results
        """
        if not HAS_SKOPT:
            self._log('error', "scikit-optimize (skopt) not available for Bayesian optimization")
            return {'error': 'scikit-optimize not available'}
        
        self._log('info', f"Running Bayesian optimization using {optimization_metric} as target metric")
        
        start_time = time.time()
        
        # Define the search space
        space = []
        param_names = []
        
        # ATR Length (integer)
        if 'atr_length' in param_ranges:
            min_val, max_val = param_ranges['atr_length']
            space.append(Integer(int(min_val), int(max_val), name='atr_length'))
            param_names.append('atr_length')
        else:
            self._log('warning', "No ATR length range provided, using default")
            space.append(Integer(10, 20, name='atr_length'))
            param_names.append('atr_length')
        
        # Factor (continuous)
        if 'factor' in param_ranges:
            min_val, max_val = param_ranges['factor']
            space.append(Real(float(min_val), float(max_val), name='factor'))
            param_names.append('factor')
        else:
            self._log('warning', "No factor range provided, using default")
            space.append(Real(1.0, 5.0, name='factor'))
            param_names.append('factor')
        
        # Buffer (continuous)
        if 'buffer' in param_ranges:
            min_val, max_val = param_ranges['buffer']
            space.append(Real(float(min_val), float(max_val), name='buffer'))
            param_names.append('buffer_multiplier')
        else:
            self._log('warning', "No buffer range provided, using default")
            space.append(Real(0.1, 0.5, name='buffer'))
            param_names.append('buffer_multiplier')
        
        # Stop (integer)
        if 'stop' in param_ranges:
            min_val, max_val = param_ranges['stop']
            space.append(Integer(int(min_val), int(max_val), name='stop'))
            param_names.append('hard_stop_distance')
        else:
            self._log('warning', "No stop range provided, using default")
            space.append(Integer(10, 50, name='stop'))
            param_names.append('hard_stop_distance')
        
        # Initialize components once for all evaluations
        self.supertrend = SuperTrend(self.log)
        self.trade_processor = TradeProcessor(self.log)
        self.performance_calculator = PerformanceCalculator(self.log)
        
        
        
        
        # Prepare GPU for optimization by running a dummy calculation
        self._log('info', "Precomputing GPU patterns for optimization...")
        try:
            # Compute reusable patterns once for optimization
            precomputed_patterns = self.supertrend.precompute_patterns(df)
            self._log('info', "GPU patterns precomputed successfully")
        except Exception as e:
            self._log('warning', f"Could not precompute GPU patterns: {str(e)}")
            precomputed_patterns = None
        
        # Store copy of data
        self.df = df
        self.min_trades = min_trades
        self.time_exit_hours = time_exit_hours
        self.optimization_metric = optimization_metric
        self.force_implementation = force_implementation
        self.precomputed_patterns = precomputed_patterns
        
        # Storage for evaluated results to track progress
        self.evaluated_results = []
        self.valid_results_count = 0
        
        # Send initial progress update
        if self.progress_callback:
            self.progress_callback({
                'completed': 0,
                'total': n_calls,
                'progress_pct': 0,
                'elapsed': 0,
                'remaining': 0,
                'valid_results': 0,
                'top_combinations': []
            })
        
        
        
        
        # Define the objective function to minimize (negative of our metric to maximize)
        def objective(params):
            # Create a dictionary of parameters
            param_dict = {name: value for name, value in zip(param_names, params)}
            
            # Extract parameters
            atr_length = int(param_dict.get('atr_length', 14))
            factor = float(param_dict.get('factor', 3.0))
            buffer = float(param_dict.get('buffer_multiplier', 0.3))
            stop = float(param_dict.get('hard_stop_distance', 50))
            
            # Calculate evaluation count for progress tracking
            evaluation_count = len(self.evaluated_results) + 1
            
            try:
                # Calculate SuperTrend using shared instance and GPU
                df_st = self.supertrend.calculate(
                    self.df, atr_length, factor, buffer,
                    force_implementation=self.force_implementation,
                    precomputed_patterns=self.precomputed_patterns
                )
                
                # Process signals to get trades
                trades = self.trade_processor.process_supertrend_signals(df_st, stop, time_exit_hours)
                
                # Check if meets minimum trades requirement
                if len(trades) < min_trades:
                    result = {
                        'parameters': param_dict,
                        'performance': {'trade_count': len(trades), optimization_metric: 0.0},
                        'trade_count': len(trades),
                        'valid': False
                    }
                    self.evaluated_results.append(result)
                    
                    # Send progress update
                    if self.progress_callback:
                        elapsed = time.time() - start_time
                        progress_pct = (evaluation_count / n_calls) * 100
                        remaining = (elapsed / evaluation_count) * (n_calls - evaluation_count) if evaluation_count > 0 else 0
                        
                        self.progress_callback({
                            'completed': evaluation_count,
                            'total': n_calls,
                            'progress_pct': progress_pct,
                            'elapsed': elapsed,
                            'remaining': remaining,
                            'valid_results': self.valid_results_count,
                            'top_combinations': self._get_top_combinations()
                        })
                    
                    return 0.0  # Penalty for too few trades
                
                # Calculate performance metrics
                performance = self.performance_calculator.calculate_performance(trades)
                
                # Get the optimization metric
                metric_value = performance.get(optimization_metric, 0.0)
                
                # Store result
                result = {
                    'parameters': param_dict,
                    'performance': performance,
                    'trade_count': len(trades),
                    'valid': True
                }
                
                self.evaluated_results.append(result)
                self.valid_results_count += 1
                
                # Send progress update
                if self.progress_callback:
                    elapsed = time.time() - start_time
                    progress_pct = (evaluation_count / n_calls) * 100
                    remaining = (elapsed / evaluation_count) * (n_calls - evaluation_count) if evaluation_count > 0 else 0
                    
                    self.progress_callback({
                        'completed': evaluation_count,
                        'total': n_calls,
                        'progress_pct': progress_pct,
                        'elapsed': elapsed,
                        'remaining': remaining,
                        'valid_results': self.valid_results_count,
                        'top_combinations': self._get_top_combinations()
                    })
                
                # We want to maximize, so return negative for minimization
                return -metric_value
                
            except Exception as e:
                self._log('error', f"Error in objective function: {str(e)}")
                # Store failed result
                failed_result = {
                    'parameters': param_dict,
                    'error': str(e),
                    'valid': False
                }
                self.evaluated_results.append(failed_result)
                
                # Send progress update
                if self.progress_callback:
                    elapsed = time.time() - start_time
                    progress_pct = (evaluation_count / n_calls) * 100
                    remaining = (elapsed / evaluation_count) * (n_calls - evaluation_count) if evaluation_count > 0 else 0
                    
                    self.progress_callback({
                        'completed': evaluation_count,
                        'total': n_calls,
                        'progress_pct': progress_pct,
                        'elapsed': elapsed,
                        'remaining': remaining,
                        'valid_results': self.valid_results_count,
                        'top_combinations': self._get_top_combinations()
                    })
                
                return 0.0
        
        # Run the optimization
        self._log('info', f"Starting optimization with {n_calls} total calls, {n_random_starts} random starts")
        
        opt_results = None
        try:
            opt_results = Optimizer(
                dimensions=space,
                random_state=42,
                n_initial_points=n_random_starts
            )
            
            opt_results.run(
                func=objective,
                n_iter=n_calls - n_random_starts
            )
        except Exception as e:
            self._log('error', f"Error during optimization: {str(e)}")
            return {'error': str(e)}
        
        execution_time = time.time() - start_time
        
        # Process results
        if opt_results is None:
            self._log('error', "Optimization failed to run")
            return {'error': 'Optimization failed'}
        
        # Get the best parameters
        best_params = opt_results.x
        best_param_dict = {name: value for name, value in zip(param_names, best_params)}
        
        # Convert types to correct format
        best_param_dict['atr_length'] = int(best_param_dict.get('atr_length', 14))
        best_param_dict['factor'] = float(best_param_dict.get('factor', 3.0))
        best_param_dict['buffer_multiplier'] = float(best_param_dict.get('buffer_multiplier', 0.3))
        best_param_dict['hard_stop_distance'] = float(best_param_dict.get('hard_stop_distance', 50))
        
        # Run final evaluation with best parameters
        final_result = self._evaluate_parameters(best_param_dict)
        
        # Prepare results
        all_evaluations = []
        for i, res in enumerate(opt_results.func_vals):
            params = opt_results.x_iters[i]
            param_dict = {name: value for name, value in zip(param_names, params)}
            
            # Convert types
            param_dict['atr_length'] = int(param_dict.get('atr_length', 14))
            param_dict['factor'] = float(param_dict.get('factor', 3.0))
            param_dict['buffer_multiplier'] = float(param_dict.get('buffer_multiplier', 0.3))
            param_dict['hard_stop_distance'] = float(param_dict.get('hard_stop_distance', 50))
            
            all_evaluations.append({
                'parameters': param_dict,
                'objective_value': -res,  # Convert back to positive
                'iteration': i
            })
        
        # Sort by objective value (descending)
        all_evaluations.sort(key=lambda x: x['objective_value'], reverse=True)
        
        # Compile results
        results = {
            'best_parameters': best_param_dict,
            'best_performance': final_result['performance'] if final_result else None,
            'all_evaluations': all_evaluations,
            'trade_count': final_result['trade_count'] if final_result else 0,
            'convergence': list(opt_results.func_vals),
            'metadata': {
                'run_date': self.current_utc,
                'run_by': self.current_user,
                'execution_time_seconds': execution_time,
                'optimization_metric': optimization_metric,
                'min_trades': min_trades,
                'n_calls': n_calls,
                'n_random_starts': n_random_starts,
                'implementation': self.force_implementation,
                'data_period': {
                    'start': df.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                    'end': df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                    'candle_count': len(df)
                }
            }
        }
        
        # Generate parameter analysis
        parameter_analyzer = ParameterAnalyzer(self.log)
        valid_results = [r for r in self.evaluated_results if r.get('valid', False)]
        if valid_results:
            results['parameter_analysis'] = parameter_analyzer.analyze_parameter_distribution(
                valid_results, optimization_metric
            )
        
        # Final progress update
        if self.progress_callback:
            self.progress_callback({
                'completed': n_calls,
                'total': n_calls,
                'progress_pct': 100,
                'elapsed': execution_time,
                'remaining': 0,
                'valid_results': self.valid_results_count,
                'top_combinations': self._get_top_combinations()
            })
        
        # Explicitly clean up resources
        if hasattr(self.supertrend, 'cleanup_resources'):
            self.supertrend.cleanup_resources()
        
        self._log('info', f"Bayesian optimization completed in {execution_time:.2f} seconds")
        
        return results
    
    def _get_top_combinations(self, top_n=5):
        """Extract top combinations from evaluated results for progress updates"""
        valid_results = [r for r in self.evaluated_results if r.get('valid', False)]
        if not valid_results:
            return []
        
        # Sort by optimization metric
        sorted_results = sorted(
            valid_results,
            key=lambda x: x['performance'].get(self.optimization_metric, 0),
            reverse=True
        )[:top_n]
        
        # Format for display
        top_combos = []
        for result in sorted_results:
            params = result['parameters']
            perf = result['performance']
            
            top_combos.append({
                'parameters': params,
                'performance': {
                    'win_rate': perf.get('win_rate', 0),
                    'profit_factor': perf.get('profit_factor', 0),
                    'total_profit_pct': perf.get('total_profit_pct', 0),
                    self.optimization_metric: perf.get(self.optimization_metric, 0)
                }
            })
        
        return top_combos
    
    def _evaluate_parameters(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate a set of parameters using GPU optimization"""
        try:
            # Extract parameters
            atr_length = int(params.get('atr_length', 14))
            factor = float(params.get('factor', 3.0))
            buffer = float(params.get('buffer_multiplier', 0.3))
            stop = float(params.get('hard_stop_distance', 50))
            
            # Calculate SuperTrend with GPU
            df_st = self.supertrend.calculate(
                self.df, atr_length, factor, buffer, 
                force_implementation=self.force_implementation,
                precomputed_patterns=self.precomputed_patterns
            )
            
            # Process signals to get trades
            trades = self.trade_processor.process_supertrend_signals(df_st, stop, self.time_exit_hours)
            
            # Calculate performance metrics
            performance = self.performance_calculator.calculate_performance(trades)
            
            return {
                'parameters': params,
                'performance': performance,
                'trade_count': len(trades)
            }
            
        except Exception as e:
            self._log('error', f"Error evaluating parameters: {str(e)}")
            return None


class GeneticOptimizer:
    """
    Genetic Algorithm-based optimization for SuperTrend parameters
    Uses DEAP (Distributed Evolutionary Algorithms in Python) for implementation
    """
    
    def __init__(self, log_manager: LogManager = None, progress_callback=None):
        self.log = log_manager
        self.current_utc = "2025-06-27 19:08:30"  # Updated timestamp
        self.current_user = "arullr001"           # Updated username
        self.progress_callback = progress_callback  # Add progress callback
        
        # Initialize components - will be set during optimization
        self.supertrend = None
        self.trade_processor = None
        self.performance_calculator = None
        
        # Check if DEAP is available
        if not HAS_DEAP:
            self._log('warning', "DEAP library not available, genetic optimization will not work")
    
    def _log(self, level: str, message: str):
        """Log message if log_manager is available"""
        if self.log:
            if level == 'info':
                self.log.info(message)
            elif level == 'debug':
                self.log.debug(message)
            elif level == 'warning':
                self.log.warning(message)
            elif level == 'error':
                self.log.error(message)
    
    def optimize(self, df: pd.DataFrame, 
                param_ranges: Dict[str, Tuple[float, float]],
                optimization_metric: str = 'profit_factor',
                min_trades: int = 10,
                time_exit_hours: float = 48.0,
                population_size: int = 30,
                generations: int = 10,
                mutation_prob: float = 0.2,
                crossover_prob: float = 0.7,
                tournament_size: int = 3,
                force_implementation: str = 'gpu') -> Dict[str, Any]:
        """
        Run Genetic Algorithm optimization for SuperTrend parameters
        
        Args:
            df: DataFrame with OHLC data
            param_ranges: Dictionary with parameter ranges as (min, max) tuples
            optimization_metric: Metric to optimize ('profit_factor', 'win_rate', etc.)
            min_trades: Minimum number of trades required for a valid result
            time_exit_hours: Exit trade after this many hours
            population_size: Size of the population in each generation
            generations: Number of generations to evolve
            mutation_prob: Probability of mutation
            crossover_prob: Probability of crossover 
            tournament_size: Size of tournament selection
            force_implementation: Force specific SuperTrend implementation
            
        Returns:
            Dictionary with optimization results
        """
        if not HAS_DEAP:
            self._log('error', "DEAP library not available for genetic optimization")
            return {'error': 'DEAP library not available'}
        
        self._log('info', f"Running Genetic Algorithm optimization using {optimization_metric} as target metric")
        
        start_time = time.time()
        
        # Initialize components for shared use
        self.supertrend = SuperTrend(self.log)
        self.trade_processor = TradeProcessor(self.log)
        self.performance_calculator = PerformanceCalculator(self.log)
        
        # Store optimization settings as instance variables
        self.df = df
        self.min_trades = min_trades
        self.time_exit_hours = time_exit_hours
        self.optimization_metric = optimization_metric
        self.population_size = population_size
        self.generations = generations
        
        # Define parameter ranges
        self.param_ranges = param_ranges
        
        # Prepare GPU implementation
        self._log('info', "Precomputing GPU patterns for optimization...")
        try:
            # Compute reusable patterns once for optimization
            self.precomputed_patterns = self.supertrend.precompute_patterns(df)
            self._log('info', "GPU patterns precomputed successfully")
        except Exception as e:
            self._log('warning', f"Could not precompute GPU patterns: {str(e)}")
            self.precomputed_patterns = None
        
        self.force_implementation = force_implementation
        
        # Storage for tracking progress
        self.evaluated_individuals = 0
        self.total_expected_evaluations = population_size * (generations + 1) # +1 for initial pop
        self.valid_results_count = 0
        self.all_individuals = []
        
        # Register DEAP types
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Define the genome (chromosome) structure
        self.param_names = []
        genome = []
        
        # ATR Length (integer)
        if 'atr_length' in param_ranges:
            min_val, max_val = param_ranges['atr_length']
            genome.append((int(min_val), int(max_val), True))  # True for integer
            self.param_names.append('atr_length')
        else:
            genome.append((10, 20, True))  # Default ATR Length
            self.param_names.append('atr_length')
        
        # Factor (continuous)
        if 'factor' in param_ranges:
            min_val, max_val = param_ranges['factor']
            genome.append((float(min_val), float(max_val), False))  # False for float
            self.param_names.append('factor')
        else:
            genome.append((1.0, 5.0, False))  # Default Factor
            self.param_names.append('factor')
        
        # Buffer (continuous)
        if 'buffer' in param_ranges:
            min_val, max_val = param_ranges['buffer']
            genome.append((float(min_val), float(max_val), False))  # False for float
            self.param_names.append('buffer_multiplier')
        else:
            genome.append((0.1, 0.5, False))  # Default Buffer
            self.param_names.append('buffer_multiplier')
        
        # Stop (integer)
        if 'stop' in param_ranges:
            min_val, max_val = param_ranges['stop']
            genome.append((int(min_val), int(max_val), True))  # True for integer
            self.param_names.append('hard_stop_distance')
        else:
            genome.append((10, 50, True))  # Default Stop
            self.param_names.append('hard_stop_distance')
        
        # Create toolbox
        toolbox = base.Toolbox()
        
        # Create a compatible result object
        from types import SimpleNamespace
        opt_results = SimpleNamespace()
        opt_results.x = best_params  # Your best parameters array
        opt_results.fun = best_score  # The optimal score
    
        return opt_results  # Now has the .x attribute
        
        
        # Register attribute generators
        # Generate a random value within range, respecting int/float type
        def generate_param(min_val, max_val, is_int):
            if is_int:
                return random.randint(min_val, max_val)
            else:
                return random.uniform(min_val, max_val)
        
        # Register the toolbox operations
        for i, (min_val, max_val, is_int) in enumerate(genome):
            # Attribute generators - one for each parameter
            toolbox.register(f"attr_{i}", generate_param, min_val, max_val, is_int)
        
        # Structure initializers - create individual and population
        # Initialize with all the attributes
        attrs = [getattr(toolbox, f"attr_{i}") for i in range(len(genome))]
        toolbox.register("individual", tools.initCycle, creator.Individual, tuple(attrs), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Define evaluation function
        def evaluate_individual(individual):
            params = {}
            
            # Extract parameters with proper type conversion
            for i, param_name in enumerate(self.param_names):
                if genome[i][2]:  # If integer
                    params[param_name] = int(individual[i])
                else:
                    params[param_name] = float(individual[i])
            
            # Update evaluated count
            self.evaluated_individuals += 1
            
            # Calculate fitness
            result = self._evaluate_parameters(params)
            
            if not result:
                # If invalid, penalize the fitness
                self.all_individuals.append({
                    'parameters': params,
                    'generation': 'initial' if self.evaluated_individuals <= population_size else 'evolved',
                    'fitness': 0.0,
                    'valid': False
                })
                
                # Report progress
                if self.progress_callback:
                    elapsed = time.time() - start_time
                    progress_pct = min(100, (self.evaluated_individuals / self.total_expected_evaluations) * 100)
                    remaining = (elapsed / self.evaluated_individuals) * (self.total_expected_evaluations - self.evaluated_individuals) if self.evaluated_individuals > 0 else 0
                    
                    if self.evaluated_individuals % 5 == 0:  # Update every 5 evaluations
                        self.progress_callback({
                            'completed': self.evaluated_individuals,
                            'total': self.total_expected_evaluations,
                            'progress_pct': progress_pct,
                            'elapsed': elapsed,
                            'remaining': remaining,
                            'valid_results': self.valid_results_count,
                            'top_combinations': self._get_top_combinations()
                        })
                
                return (0.0,)  # Return tuple for DEAP
            
            # Get fitness value
            fitness_value = result['performance'].get(self.optimization_metric, 0.0)
            
            # Track valid result
            self.valid_results_count += 1
            self.all_individuals.append({
                'parameters': params,
                'generation': 'initial' if self.evaluated_individuals <= population_size else 'evolved',
                'fitness': fitness_value,
                'valid': True,
                'performance': result['performance']
            })
            
            # Report progress
            if self.progress_callback:
                elapsed = time.time() - start_time
                progress_pct = min(100, (self.evaluated_individuals / self.total_expected_evaluations) * 100)
                remaining = (elapsed / self.evaluated_individuals) * (self.total_expected_evaluations - self.evaluated_individuals) if self.evaluated_individuals > 0 else 0
                
                if self.evaluated_individuals % 5 == 0:  # Update every 5 evaluations
                    self.progress_callback({
                        'completed': self.evaluated_individuals,
                        'total': self.total_expected_evaluations,
                        'progress_pct': progress_pct,
                        'elapsed': elapsed,
                        'remaining': remaining,
                        'valid_results': self.valid_results_count,
                        'top_combinations': self._get_top_combinations()
                    })
            
            return (fitness_value,)  # Return tuple for DEAP
        
        # Register evaluation function
        toolbox.register("evaluate", evaluate_individual)
        
        # Genetic operators
        # Tournament selection with tournament_size
        toolbox.register("select", tools.selTournament, tournsize=tournament_size)
        
        # Multi-point crossover with crossover_prob probability
        def custom_crossover(ind1, ind2):
            for i in range(len(ind1)):
                if random.random() < crossover_prob:
                    ind1[i], ind2[i] = ind2[i], ind1[i]
            return ind1, ind2
        
        toolbox.register("mate", custom_crossover)
        
        # Mutation with mutation_prob probability
        def custom_mutate(individual):
            for i, (min_val, max_val, is_int) in enumerate(genome):
                if random.random() < mutation_prob:
                    if is_int:
                        individual[i] = random.randint(min_val, max_val)
                    else:
                        individual[i] = random.uniform(min_val, max_val)
            return individual,
            
        toolbox.register("mutate", custom_mutate)
        
        # Initialize population
        pop = toolbox.population(n=population_size)
        
        # Record initial population statistics
        self._log('info', f"Initial population size: {len(pop)}")
        
        # Evaluate initial population in parallel batches
        self._log('info', "Evaluating initial population using GPU optimization...")
        
        # Use thread pool to evaluate initial population
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, CPU_THREADS)) as executor:
            fitnesses = list(executor.map(toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
        
        # Signal start of evolution
        self._log('info', f"Starting evolution for {generations} generations")
        
        # Begin the evolution
        for g in range(generations):
            # Select parents
            offspring = toolbox.select(pop, len(pop))
            
            # Clone selected individuals
            offspring = list(map(toolbox.clone, offspring))
            
            # Apply crossover and mutation
            for i in range(1, len(offspring), 2):
                # Crossover
                if i < len(offspring) - 1:  # Check to avoid index error
                    toolbox.mate(offspring[i-1], offspring[i])
                    
                    # Clear fitness values of modified individuals
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values
            
            # Mutation with a controlled rate
            for i in range(len(offspring)):
                toolbox.mutate(offspring[i])
                if not offspring[i].fitness.valid:
                    del offspring[i].fitness.values
            
            # Evaluate offspring in parallel batches
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            
            if invalid_ind:
                # Use thread pool for batch evaluation
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, CPU_THREADS)) as executor:
                    fitnesses = list(executor.map(toolbox.evaluate, invalid_ind))
                    for ind, fit in zip(invalid_ind, fitnesses):
                        ind.fitness.values = fit
            
            # Replace population with offspring
            pop[:] = offspring
            
            # Log progress
            best_ind = tools.selBest(pop, 1)[0]
            best_fitness = best_ind.fitness.values[0]
            
            self._log('info', f"Generation {g+1}/{generations}: Best fitness = {best_fitness}")
        
        # Retrieve best individual
        best_ind = tools.selBest(pop, 1)[0]
        
        # Convert best individual to parameters
        best_params = {}
        for i, param_name in enumerate(self.param_names):
            if genome[i][2]:  # If integer
                best_params[param_name] = int(best_ind[i])
            else:
                best_params[param_name] = float(best_ind[i])
                
        # Do final evaluation
        final_result = self._evaluate_parameters(best_params)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Compile results
        all_evaluations = sorted(
            self.all_individuals,
            key=lambda x: x.get('fitness', 0),
            reverse=True
        )
        
        # Compile analysis
        valid_results = [r for r in self.all_individuals if r.get('valid', False)]
        
        # Generate parameter analysis
        parameter_analysis = None
        if valid_results:
            parameter_analyzer = ParameterAnalyzer(self.log)
            parameter_analysis = parameter_analyzer.analyze_parameter_distribution(
                valid_results, optimization_metric
            )
        
        # Final result set
        results = {
            'best_parameters': best_params,
            'best_performance': final_result['performance'] if final_result else None,
            'all_evaluations': all_evaluations,
            'parameter_analysis': parameter_analysis,
            'metadata': {
                'run_date': self.current_utc,
                'run_by': self.current_user,
                'execution_time_seconds': execution_time,
                'optimization_metric': optimization_metric,
                'min_trades': min_trades,
                'population_size': population_size,
                'generations': generations,
                'mutation_prob': mutation_prob,
                'crossover_prob': crossover_prob,
                'tournament_size': tournament_size,
                'implementation': force_implementation,
                'total_evaluations': self.evaluated_individuals,
                'valid_evaluations': self.valid_results_count,
                'data_period': {
                    'start': df.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                    'end': df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                    'candle_count': len(df)
                }
            }
        }
        
        # Final progress update
        if self.progress_callback:
            self.progress_callback({
                'completed': self.total_expected_evaluations,
                'total': self.total_expected_evaluations,
                'progress_pct': 100,
                'elapsed': execution_time,
                'remaining': 0,
                'valid_results': self.valid_results_count,
                'top_combinations': self._get_top_combinations()
            })
        
        # Clean up creator classes to avoid warning when running multiple optimizations
        if hasattr(creator, 'FitnessMax'):
            del creator.FitnessMax
        if hasattr(creator, 'Individual'):
            del creator.Individual
            
        # Explicitly clean up resources
        if hasattr(self.supertrend, 'cleanup_resources'):
            self.supertrend.cleanup_resources()
        
        self._log('info', f"Genetic optimization completed in {execution_time:.2f} seconds")
        
        return results
    
    def _get_top_combinations(self, top_n=5):
        """Extract top combinations for progress updates"""
        valid_results = [r for r in self.all_individuals if r.get('valid', False)]
        if not valid_results:
            return []
        
        # Sort by fitness (which is the optimization metric)
        sorted_results = sorted(
            valid_results,
            key=lambda x: x.get('fitness', 0),
            reverse=True
        )[:top_n]
        
        # Format for display
        top_combos = []
        for result in sorted_results:
            params = result['parameters']
            perf = result.get('performance', {})
            
            top_combos.append({
                'parameters': params,
                'performance': {
                    'win_rate': perf.get('win_rate', 0),
                    'profit_factor': perf.get('profit_factor', 0),
                    'total_profit_pct': perf.get('total_profit_pct', 0),
                    self.optimization_metric: perf.get(self.optimization_metric, 0)
                }
            })
        
        return top_combos
    
    def _evaluate_parameters(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate a set of parameters with GPU optimization"""
        try:
            # Extract parameters
            atr_length = int(params.get('atr_length', 14))
            factor = float(params.get('factor', 3.0))
            buffer = float(params.get('buffer_multiplier', 0.3))
            stop = float(params.get('hard_stop_distance', 50))
            
            # Calculate SuperTrend with GPU
            df_st = self.supertrend.calculate(
                self.df, atr_length, factor, buffer, 
                force_implementation=self.force_implementation,
                precomputed_patterns=self.precomputed_patterns
            )
            
            # Process signals to get trades
            trades = self.trade_processor.process_supertrend_signals(df_st, stop, self.time_exit_hours)
            
            # Check if meets minimum trades requirement
            if len(trades) < self.min_trades:
                return None
            
            # Calculate performance metrics
            performance = self.performance_calculator.calculate_performance(trades)
            
            return {
                'parameters': params,
                'performance': performance,
                'trade_count': len(trades)
            }
            
        except Exception as e:
            self._log('error', f"Error evaluating parameters: {str(e)}")
            return None


class ParameterAnalyzer:
    """
    Analyzes parameter optimization results to extract insights and recommendations
    """
    
    def __init__(self, log_manager: LogManager = None, progress_callback=None):
        self.log = log_manager
        self.current_utc = "2025-06-24 13:08:53"  # Updated timestamp
        self.current_user = "arullr001"           # Updated username
    
    def _log(self, level: str, message: str):
        """Log message if log_manager is available"""
        if self.log:
            if level == 'info':
                self.log.info(message)
            elif level == 'debug':
                self.log.debug(message)
            elif level == 'warning':
                self.log.warning(message)
            elif level == 'error':
                self.log.error(message)
    
    def analyze_parameter_distribution(self, results: List[Dict], 
                                    optimization_metric: str) -> Dict[str, Any]:
        """
        Analyze parameter distribution among top-performing combinations
        
        Args:
            results: List of optimization results
            optimization_metric: Metric used for optimization
            
        Returns:
            Dictionary with parameter analysis
        """
        if not results:
            self._log('warning', "No results to analyze parameter distribution")
            return {}
        
        # Sort by optimization metric
        sorted_results = sorted(results, key=lambda x: x['performance'].get(optimization_metric, 0), reverse=True)
        
        # Take top 10% or at least 5 results
        top_n = max(5, len(sorted_results) // 10)
        top_results = sorted_results[:top_n]
        
        self._log('info', f"Analyzing parameter distribution among top {top_n} results")
        
        # Extract parameters
        atr_lengths = [r['parameters']['atr_length'] for r in top_results]
        factors = [r['parameters']['factor'] for r in top_results]
        buffers = [r['parameters']['buffer_multiplier'] for r in top_results]
        stops = [r['parameters']['hard_stop_distance'] for r in top_results]
        
        # Calculate statistics
        analysis = {
            'atr_length': {
                'mean': np.mean(atr_lengths),
                'median': np.median(atr_lengths),
                'min': np.min(atr_lengths),
                'max': np.max(atr_lengths),
                'std': np.std(atr_lengths),
                'most_common': max(set(atr_lengths), key=atr_lengths.count)
            },
            'factor': {
                'mean': np.mean(factors),
                'median': np.median(factors),
                'min': np.min(factors),
                'max': np.max(factors),
                'std': np.std(factors),
                'most_common': max(set(factors), key=factors.count)
            },
            'buffer_multiplier': {
                'mean': np.mean(buffers),
                'median': np.median(buffers),
                'min': np.min(buffers),
                'max': np.max(buffers),
                'std': np.std(buffers),
                'most_common': max(set(buffers), key=buffers.count)
            },
            'hard_stop_distance': {
                'mean': np.mean(stops),
                'median': np.median(stops),
                'min': np.min(stops),
                'max': np.max(stops),
                'std': np.std(stops),
                'most_common': max(set(stops), key=stops.count)
            }
        }
        
        # Calculate parameter importance using variance
        variances = {
            'atr_length': np.var(atr_lengths),
            'factor': np.var(factors),
            'buffer_multiplier': np.var(buffers),
            'hard_stop_distance': np.var(stops)
        }
        
        # Normalize to sum to 1
        total_variance = sum(variances.values())
        importance = {k: v / total_variance if total_variance > 0 else 0.25 for k, v in variances.items()}
        
        analysis['parameter_importance'] = importance
        
        # Compute recommended parameter ranges based on top performers
        analysis['recommended_ranges'] = {
            'atr_length': (int(np.floor(np.min(atr_lengths))), int(np.ceil(np.max(atr_lengths)))),
            'factor': (np.min(factors), np.max(factors)),
            'buffer_multiplier': (np.min(buffers), np.max(buffers)),
            'hard_stop_distance': (int(np.floor(np.min(stops))), int(np.ceil(np.max(stops))))
        }
        
        # Compute optimal parameter combination
        optimal = {
            'atr_length': int(round(analysis['atr_length']['median'])),
            'factor': float(analysis['factor']['median']),
            'buffer_multiplier': float(analysis['buffer_multiplier']['median']),
            'hard_stop_distance': float(analysis['hard_stop_distance']['median'])
        }
        
        analysis['optimal_combination'] = optimal
        
        # Find correlations between parameters and performance
        correlations = self._calculate_correlations(sorted_results, optimization_metric)
        analysis['correlations'] = correlations
        
        # Generate insights based on correlations
        analysis['insights'] = self._generate_insights(correlations, analysis)
        
        self._log('debug', f"Parameter analysis completed with {len(top_results)} top results")
        
        return analysis
    
    def _calculate_correlations(self, results: List[Dict], metric: str) -> Dict[str, float]:
        """Calculate correlations between parameters and performance"""
        if not results:
            return {}
            
        # Extract data
        atr_lengths = [r['parameters']['atr_length'] for r in results]
        factors = [r['parameters']['factor'] for r in results]
        buffers = [r['parameters']['buffer_multiplier'] for r in results]
        stops = [r['parameters']['hard_stop_distance'] for r in results]
        metrics = [r['performance'].get(metric, 0) for r in results]
        
        # Calculate correlations
        corr_atr = np.corrcoef(atr_lengths, metrics)[0, 1] if len(set(atr_lengths)) > 1 else 0
        corr_factor = np.corrcoef(factors, metrics)[0, 1] if len(set(factors)) > 1 else 0
        corr_buffer = np.corrcoef(buffers, metrics)[0, 1] if len(set(buffers)) > 1 else 0
        corr_stop = np.corrcoef(stops, metrics)[0, 1] if len(set(stops)) > 1 else 0
        
        return {
            'atr_length': corr_atr,
            'factor': corr_factor,
            'buffer_multiplier': corr_buffer,
            'hard_stop_distance': corr_stop
        }
    
    def _generate_insights(self, correlations: Dict[str, float], analysis: Dict[str, Any]) -> List[str]:
        """Generate insights based on parameter analysis"""
        insights = []
        
        # Check importance
        importance = analysis['parameter_importance']
        most_important = max(importance, key=importance.get)
        least_important = min(importance, key=importance.get)
        
        insights.append(f"The most influential parameter is '{most_important}' (importance: {importance[most_important]:.2f}).")
        insights.append(f"The least influential parameter is '{least_important}' (importance: {importance[least_important]:.2f}).")
        
        # Check correlations
        for param, corr in correlations.items():
            if abs(corr) > 0.7:
                direction = "positively" if corr > 0 else "negatively"
                insights.append(f"'{param}' is strongly {direction} correlated with performance (correlation: {corr:.2f}).")
            elif abs(corr) > 0.3:
                direction = "positively" if corr > 0 else "negatively"
                insights.append(f"'{param}' is moderately {direction} correlated with performance (correlation: {corr:.2f}).")
        
        # Check parameter ranges
        ranges = analysis['recommended_ranges']
        for param, (min_val, max_val) in ranges.items():
            range_size = max_val - min_val
            
            if range_size == 0:
                insights.append(f"All top results use the exact same value for '{param}': {min_val}.")
            elif range_size < 0.1 * (max_val + min_val) / 2:
                insights.append(f"Top results use a very narrow range for '{param}': {min_val} to {max_val}.")
        
        # Add suggestion for optimal parameter combination
        insights.append("Based on the analysis, the optimal parameter combination is:")
        for param, value in analysis['optimal_combination'].items():
            insights.append(f"- {param}: {value}")
        
        return insights


# ==============================================================================
# RESULTS MANAGEMENT
# ==============================================================================

class ResultsStorage:
    """
    Manages storage and retrieval of backtest and optimization results
    """
    
    def __init__(self, base_dir: str = None, log_manager: LogManager = None):
        self.log = log_manager
        self.current_utc = "2025-06-24 17:20:39"  # Updated with current UTC time
        self.current_user = "arullr001"           # Updated with current username
    
        # Set base directory for results storage
        if base_dir is None:
            # Get script directory instead of home directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Use the same run directory as logs if available
            if hasattr(log_manager, 'base_dir') and log_manager.base_dir:
                self.base_dir = log_manager.base_dir
            else:
                # Create a new results directory at script location
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                run_id = generate_unique_id()
                run_dir = f"SuperTrend_Run_{timestamp}_{run_id}"
                self.base_dir = os.path.join(script_dir, run_dir)
        else:
            self.base_dir = base_dir
    
        # Create directory structure if it doesn't exist
        self.backtest_dir = os.path.join(self.base_dir, "backtests")
        self.optimization_dir = os.path.join(self.base_dir, "optimizations")
        self.reports_dir = os.path.join(self.base_dir, "reports")
        self.charts_dir = os.path.join(self.base_dir, "charts")
    
        os.makedirs(self.backtest_dir, exist_ok=True)
        os.makedirs(self.optimization_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
    
        self._log('debug', f"ResultsStorage initialized with base directory: {self.base_dir}")    
        
        
    def _log(self, level: str, message: str):
        """Log message if log_manager is available"""
        if self.log:
            if level == 'info':
                self.log.info(message)
            elif level == 'debug':
                self.log.debug(message)
            elif level == 'warning':
                self.log.warning(message)
            elif level == 'error':
                self.log.error(message)


    def save_backtest_results(self, results: Dict[str, Any], name: str = None) -> str:
        """
        Save backtest results to disk
        
        Args:
            results: Backtest results dictionary
            name: Name for the backtest (optional)
            
        Returns:
            Path to saved file
        """
        # Generate filename
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        uid = generate_unique_id()
        
        if name:
            safe_name = "".join([c if c.isalnum() else "_" for c in name])
            filename = f"backtest_{safe_name}_{timestamp}_{uid}.json"
        else:
            filename = f"backtest_{timestamp}_{uid}.json"
        
        filepath = os.path.join(self.backtest_dir, filename)
        
        # Prepare results for serialization
        serializable_results = self._prepare_for_serialization(results)
        
        # Add metadata
        serializable_results['_metadata'] = {
            'saved_at': "2025-06-24 15:00:41",  # Updated with current UTC time
            'saved_by': "arullr001",  # Updated with current username
            'filename': filename
        }
        
        # Save to file
        try:
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self._log('info', f"Backtest results saved to {filepath}")
            return filepath
        except Exception as e:
            self._log('error', f"Error saving backtest results: {str(e)}")
            return None
    
    def save_optimization_results(self, results: Dict[str, Any], name: str = None) -> str:
        """
        Save optimization results to disk
        
        Args:
            results: Optimization results dictionary
            name: Name for the optimization (optional)
            
        Returns:
            Path to saved file
        """
        # Generate filename
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        uid = generate_unique_id()
        
        if name:
            safe_name = "".join([c if c.isalnum() else "_" for c in name])
            filename = f"optimization_{safe_name}_{timestamp}_{uid}.json"
        else:
            filename = f"optimization_{timestamp}_{uid}.json"
        
        filepath = os.path.join(self.optimization_dir, filename)
        
        # Prepare results for serialization
        serializable_results = self._prepare_for_serialization(results)
        
        # Add metadata
        serializable_results['_metadata'] = {
            'saved_at': "2025-06-24 15:00:41",  # Updated with current UTC time
            'saved_by': "arullr001",  # Updated with current username
            'filename': filename
        }
        
        # Save to file
        try:
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self._log('info', f"Optimization results saved to {filepath}")
            return filepath
        except Exception as e:
            self._log('error', f"Error saving optimization results: {str(e)}")
            return None

    def save_report(self, report_content: str, report_type: str, name: str = None) -> str:
        """
        Save a report to disk
        
        Args:
            report_content: Report content as string (HTML, markdown, etc.)
            report_type: Type of report ('html', 'md', 'txt')
            name: Name for the report (optional)
            
        Returns:
            Path to saved file
        """
        # Generate filename
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        uid = generate_unique_id()
        
        if name:
            safe_name = "".join([c if c.isalnum() else "_" for c in name])
            filename = f"report_{safe_name}_{timestamp}_{uid}.{report_type}"
        else:
            filename = f"report_{timestamp}_{uid}.{report_type}"
        
        filepath = os.path.join(self.reports_dir, filename)
        
        # Save to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self._log('info', f"Report saved to {filepath}")
            return filepath
        except Exception as e:
            self._log('error', f"Error saving report: {str(e)}")
            return None
    
    def save_chart(self, figure: plt.Figure, name: str = None, dpi: int = 300) -> str:
        """
        Save a matplotlib figure to disk
        
        Args:
            figure: Matplotlib figure to save
            name: Name for the chart (optional)
            dpi: Resolution for the saved image
            
        Returns:
            Path to saved file
        """
        # Generate filename
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        uid = generate_unique_id()
        
        if name:
            safe_name = "".join([c if c.isalnum() else "_" for c in name])
            filename = f"chart_{safe_name}_{timestamp}_{uid}.png"
        else:
            filename = f"chart_{timestamp}_{uid}.png"
        
        filepath = os.path.join(self.charts_dir, filename)
        
        # Save to file
        try:
            figure.savefig(filepath, dpi=dpi, bbox_inches='tight')
            self._log('info', f"Chart saved to {filepath}")
            return filepath
        except Exception as e:
            self._log('error', f"Error saving chart: {str(e)}")
            return None    
    
    
    
    
    def load_backtest_results(self, filename: str = None) -> Dict[str, Any]:
        """
        Load backtest results from disk
        
        Args:
            filename: Name of the file to load (if None, loads the most recent)
            
        Returns:
            Dictionary of backtest results
        """
        filepath = filename
        
        # If filename not provided, get latest
        if filename is None:
            files = [os.path.join(self.backtest_dir, f) for f in os.listdir(self.backtest_dir)
                    if f.startswith('backtest_') and f.endswith('.json')]
            
            if not files:
                self._log('warning', "No backtest results found")
                return None
                
            filepath = max(files, key=os.path.getmtime)
        else:
            if not os.path.isabs(filename):
                filepath = os.path.join(self.backtest_dir, filename)
        
        # Load from file
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
                
            self._log('info', f"Loaded backtest results from {filepath}")
            return results
        except Exception as e:
            self._log('error', f"Error loading backtest results: {str(e)}")
            return None
    
    def load_optimization_results(self, filename: str = None) -> Dict[str, Any]:
        """
        Load optimization results from disk
        
        Args:
            filename: Name of the file to load (if None, loads the most recent)
            
        Returns:
            Dictionary of optimization results
        """
        filepath = filename
        
        # If filename not provided, get latest
        if filename is None:
            files = [os.path.join(self.optimization_dir, f) for f in os.listdir(self.optimization_dir)
                    if f.startswith('optimization_') and f.endswith('.json')]
            
            if not files:
                self._log('warning', "No optimization results found")
                return None
                
            filepath = max(files, key=os.path.getmtime)
        else:
            if not os.path.isabs(filename):
                filepath = os.path.join(self.optimization_dir, filename)
        
        # Load from file
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
                
            self._log('info', f"Loaded optimization results from {filepath}")
            return results
        except Exception as e:
            self._log('error', f"Error loading optimization results: {str(e)}")
            return None


    
    def list_backtest_results(self) -> List[Dict[str, str]]:
        """
        List all available backtest results
        
        Returns:
            List of dictionaries with backtest metadata
        """
        files = [f for f in os.listdir(self.backtest_dir) if f.endswith('.json')]
        results = []
        
        for filename in sorted(files, reverse=True):
            filepath = os.path.join(self.backtest_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Extract basic metadata
                results.append({
                    'filename': filename,
                    'filepath': filepath,
                    'date': data.get('_metadata', {}).get('saved_at', 'Unknown'),
                    'name': self._extract_name_from_filename(filename),
                    'parameters': data.get('parameters', {})
                })
            except Exception as e:
                self._log('warning', f"Error reading metadata from {filename}: {str(e)}")
        
        return results
    
    def list_optimization_results(self) -> List[Dict[str, str]]:
        """
        List all available optimization results
        
        Returns:
            List of dictionaries with optimization metadata
        """
        files = [f for f in os.listdir(self.optimization_dir) if f.endswith('.json')]
        results = []
        
        for filename in sorted(files, reverse=True):
            filepath = os.path.join(self.optimization_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Extract basic metadata
                results.append({
                    'filename': filename,
                    'filepath': filepath,
                    'date': data.get('_metadata', {}).get('saved_at', '2025-06-24 15:05:44'),
                    'name': self._extract_name_from_filename(filename),
                    'optimizer': data.get('metadata', {}).get('optimizer_type', 'Unknown'),
                    'best_params': data.get('best_parameters', {})
                })
            except Exception as e:
                self._log('warning', f"Error reading metadata from {filename}: {str(e)}")
        
        return results
    
    def delete_result(self, filepath: str) -> bool:
        """
        Delete a result file
        
        Args:
            filepath: Path to the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                self._log('info', f"Deleted result file: {filepath}")
                return True
            else:
                self._log('warning', f"File not found: {filepath}")
                return False
        except Exception as e:
            self._log('error', f"Error deleting file: {str(e)}")
            return False


    def _prepare_for_serialization(self, data: Any) -> Any:
        """
        Prepare data structure for serialization to JSON
        
        Args:
            data: Any data structure
            
        Returns:
            JSON-serializable version of the data
        """
        if isinstance(data, dict):
            return {k: self._prepare_for_serialization(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_serialization(item) for item in data]
        elif isinstance(data, tuple):
            return [self._prepare_for_serialization(item) for item in data]
        elif isinstance(data, (np.int64, np.int32)):
            return int(data)
        elif isinstance(data, np.float64):
            return float(data)
        elif isinstance(data, np.ndarray):
            return self._prepare_for_serialization(data.tolist())
        elif isinstance(data, pd.DataFrame):
            return {
                '_type': 'DataFrame',
                'data': data.to_dict('records'),
                'index': data.index.tolist() if not isinstance(data.index, pd.DatetimeIndex) else [str(idx) for idx in data.index]
            }
        elif isinstance(data, pd.Series):
            return {
                '_type': 'Series',
                'data': data.tolist(),
                'index': data.index.tolist() if not isinstance(data.index, pd.DatetimeIndex) else [str(idx) for idx in data.index]
            }
        elif isinstance(data, datetime):
            return data.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return data
    
    def _extract_name_from_filename(self, filename: str) -> str:
        """Extract the name part from a result filename"""
        parts = filename.split('_')
        if len(parts) < 3:
            return "Unknown"
            
        # Skip first part (backtest/optimization) and last parts (timestamp_uid.json)
        return '_'.join(parts[1:-2])		



class ReportGenerator:
    """
    Generates reports for backtest and optimization results
    """
    
    def __init__(self, log_manager: LogManager = None):
        self.log = log_manager
        self.current_utc = "2025-06-24 15:07:24"  # Updated with current UTC time
        self.current_user = "arullr001"           # Updated with current username
    
    def _log(self, level: str, message: str):
        """Log message if log_manager is available"""
        if self.log:
            if level == 'info':
                self.log.info(message)
            elif level == 'debug':
                self.log.debug(message)
            elif level == 'warning':
                self.log.warning(message)
            elif level == 'error':
                self.log.error(message)


    def generate_backtest_report(self, results: Dict[str, Any], format_type: str = 'html') -> str:
        """
        Generate a comprehensive report for backtest results
        
        Args:
            results: Backtest results dictionary
            format_type: Output format ('html', 'md', 'txt')
            
        Returns:
            Report content as string
        """
        self._log('info', f"Generating backtest report in {format_type} format")
        
        if format_type == 'html':
            return self._generate_backtest_html_report(results)
        elif format_type == 'md':
            return self._generate_backtest_markdown_report(results)
        elif format_type == 'txt':
            return self._generate_backtest_text_report(results)
        else:
            self._log('error', f"Unsupported report format: {format_type}")
            return f"Error: Unsupported report format '{format_type}'"
    
    def generate_optimization_report(self, results: Dict[str, Any], format_type: str = 'html') -> str:
        """
        Generate a comprehensive report for optimization results
        
        Args:
            results: Optimization results dictionary
            format_type: Output format ('html', 'md', 'txt')
            
        Returns:
            Report content as string
        """
        self._log('info', f"Generating optimization report in {format_type} format")
        
        if format_type == 'html':
            return self._generate_optimization_html_report(results)
        elif format_type == 'md':
            return self._generate_optimization_markdown_report(results)
        elif format_type == 'txt':
            return self._generate_optimization_text_report(results)
        else:
            self._log('error', f"Unsupported report format: {format_type}")
            return f"Error: Unsupported report format '{format_type}'"
    
    def generate_chart_backtest(self, df: pd.DataFrame, results: Dict[str, Any]) -> plt.Figure:
        """
        Generate a comprehensive chart for backtest results
        
        Args:
            df: DataFrame with OHLC data
            results: Backtest results dictionary
            
        Returns:
            Matplotlib figure
        """
        self._log('info', "Generating backtest chart")
        
        # Set style
        set_plot_style()
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        
        # Define subplots
        gs = plt.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.2)
        
        # Price chart with SuperTrend
        ax1 = fig.add_subplot(gs[0])
        
        # Extract parameters
        params = results.get('parameters', {})
        atr_length = params.get('atr_length', 0)
        factor = params.get('factor', 0)
        buffer = params.get('buffer_multiplier', 0)
        
        # Extract trades
        trades = results.get('trades', [])
        if isinstance(trades, list) and len(trades) > 0 and isinstance(trades[0], dict):
            trade_entries = [(t['entry_time'], t['entry_price'], t['position_type']) for t in trades]
            trade_exits = [(t['exit_time'], t['exit_price'], t['position_type']) for t in trades if t['exit_time']]
        else:
            trade_entries = []
            trade_exits = []
        
        # Plot price
        ax1.plot(df.index, df['close'], color='black', linewidth=1.0, label='Close')
        
        # Plot SuperTrend if available
        if 'supertrend' in df.columns:
            uptrend_mask = df['direction'] < 0
            downtrend_mask = df['direction'] > 0
            
            # Plot supertrend line
            ax1.plot(df.index[uptrend_mask], df['supertrend'][uptrend_mask], color='green', linewidth=1.0)
            ax1.plot(df.index[downtrend_mask], df['supertrend'][downtrend_mask], color='red', linewidth=1.0)
            
            # Plot buffer zones
            if 'up_trend_buffer' in df.columns:
                ax1.plot(df.index, df['up_trend_buffer'], color='green', alpha=0.5, linewidth=0.7, linestyle='--')
            
            if 'down_trend_buffer' in df.columns:
                ax1.plot(df.index, df['down_trend_buffer'], color='red', alpha=0.5, linewidth=0.7, linestyle='--')
        
        # Plot trade entries
        for time_str, price, pos_type in trade_entries:
            try:
                time_val = pd.to_datetime(time_str)
                if pos_type.lower() == 'long':
                    ax1.scatter(time_val, price, color='green', marker='^', s=80, zorder=5)
                else:
                    ax1.scatter(time_val, price, color='red', marker='v', s=80, zorder=5)
            except Exception as e:
                self._log('warning', f"Error plotting trade entry: {str(e)}")
        
        # Plot trade exits
        for time_str, price, pos_type in trade_exits:
            try:
                time_val = pd.to_datetime(time_str)
                ax1.scatter(time_val, price, color='blue', marker='x', s=60, zorder=5)
            except Exception as e:
                self._log('warning', f"Error plotting trade exit: {str(e)}")
        
        # Set title and labels
        ax1.set_title(f"SuperTrend Backtest (ATR={atr_length}, Factor={factor}, Buffer={buffer})")
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Equity curve
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        equity_curve = results.get('performance', {}).get('equity_curve', pd.DataFrame())
        if not equity_curve.empty and isinstance(equity_curve, dict) and '_type' in equity_curve and equity_curve['_type'] == 'DataFrame':
            # If equity curve is serialized DataFrame representation
            try:
                # Recreate DataFrame
                equity_data = pd.DataFrame(equity_curve['data'])
                equity_data.index = pd.to_datetime(equity_curve['index'])
                ax2.plot(equity_data.index, equity_data['equity'], color='blue', linewidth=1.2)
                
                # Add drawdown shading
                if 'drawdown' in equity_data.columns:
                    for i in range(len(equity_data)-1):
                        if equity_data['drawdown'].iloc[i] > 0:
                            ax2.axvspan(equity_data.index[i], equity_data.index[i+1], color='red', alpha=0.1)
            except Exception as e:
                self._log('warning', f"Error plotting equity curve: {str(e)}")
                ax2.text(0.5, 0.5, "Error plotting equity curve", ha='center', va='center', transform=ax2.transAxes)
        else:
            ax2.text(0.5, 0.5, "No equity curve data available", ha='center', va='center', transform=ax2.transAxes)
        
        ax2.set_ylabel('Equity')
        ax2.grid(True, alpha=0.3)
        
        # Trade performance
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        
        if trades:
            # Extract trade profits
            trade_profits = []
            trade_times = []
            colors = []
            
            for t in trades:
                if 'exit_time' in t and t['exit_time']:
                    try:
                        exit_time = pd.to_datetime(t['exit_time'])
                        profit = t.get('profit', 0)
                        trade_times.append(exit_time)
                        trade_profits.append(profit)
                        colors.append('green' if profit > 0 else 'red')
                    except Exception as e:
                        self._log('warning', f"Error processing trade for plot: {str(e)}")
            
            # Plot trade profits
            if trade_profits:
                ax3.bar(trade_times, trade_profits, color=colors, alpha=0.7)
        else:
            ax3.text(0.5, 0.5, "No trade data available", ha='center', va='center', transform=ax3.transAxes)
        
        ax3.set_ylabel('Trade Profit')
        ax3.grid(True, alpha=0.3)
        
        # Format dates on x-axis
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        # Add performance summary
        performance = results.get('performance', {})
        if performance:
            summary_text = (
                f"Total trades: {performance.get('trade_count', 0)}  |  "
                f"Win rate: {performance.get('win_rate', 0)*100:.1f}%  |  "
                f"Profit factor: {performance.get('profit_factor', 0):.2f}  |  "
                f"Total profit: {performance.get('total_profit_pct', 0):.2f}%  |  "
                f"Max DD: {performance.get('max_drawdown', 0):.2f}%"
            )
            fig.text(0.5, 0.01, summary_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        
        # Adjust layout
        fig.tight_layout()
        
        return fig


		
    def generate_chart_optimization(self, results: Dict[str, Any]) -> plt.Figure:
        """
        Generate a comprehensive chart for optimization results
        
        Args:
            results: Optimization results dictionary
            
        Returns:
            Matplotlib figure
        """
        self._log('info', "Generating optimization chart")
        
        # Set style
        set_plot_style()
        
        # Check if we have parameter analysis
        if 'parameter_analysis' not in results:
            self._log('warning', "No parameter analysis in results, cannot generate chart")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No parameter analysis data available", ha='center', va='center', transform=ax.transAxes)
            return fig
        
        param_analysis = results['parameter_analysis']
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        
        # Define subplots
        gs = plt.GridSpec(2, 2, hspace=0.3, wspace=0.3)
        
        # Parameter importance chart
        ax1 = fig.add_subplot(gs[0, 0])
        
        if 'parameter_importance' in param_analysis:
            importance = param_analysis['parameter_importance']
            labels = list(importance.keys())
            values = list(importance.values())
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
            ax1.bar(labels, values, color=colors)
            ax1.set_title('Parameter Importance')
            ax1.set_ylabel('Importance Score')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Rotate labels if necessary
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        else:
            ax1.text(0.5, 0.5, "Parameter importance data not available", ha='center', va='center', transform=ax1.transAxes)
        
        # Parameter correlation chart
        ax2 = fig.add_subplot(gs[0, 1])
        
        if 'correlations' in param_analysis:
            correlations = param_analysis['correlations']
            labels = list(correlations.keys())
            values = list(correlations.values())
            
            colors = ['green' if v >= 0 else 'red' for v in values]
            ax2.bar(labels, values, color=colors)
            ax2.set_title('Parameter Correlations with Performance')
            ax2.set_ylabel('Correlation Coefficient')
            ax2.set_ylim(min(min(values) * 1.2, -1), max(max(values) * 1.2, 1))
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Rotate labels if necessary
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        else:
            ax2.text(0.5, 0.5, "Correlation data not available", ha='center', va='center', transform=ax2.transAxes)

        # Parameter heatmaps for top 2 parameters
        if 'parameter_heatmaps' in param_analysis and len(param_analysis['parameter_heatmaps']) >= 1:
            heatmaps = param_analysis['parameter_heatmaps']
            
            # First heatmap (most important parameters)
            ax3 = fig.add_subplot(gs[1, 0])
            if len(heatmaps) > 0:
                hm1 = heatmaps[0]
                param1 = hm1['param_x']
                param2 = hm1['param_y']
                values = hm1['values']
                
                # Create heatmap matrix
                x_labels = sorted(list(set([point[0] for point in values])))
                y_labels = sorted(list(set([point[1] for point in values])))
                z_matrix = np.zeros((len(y_labels), len(x_labels)))
                
                for x, y, z in values:
                    x_idx = x_labels.index(x)
                    y_idx = y_labels.index(y)
                    z_matrix[y_idx, x_idx] = z
                
                # Plot heatmap
                im = ax3.imshow(z_matrix, cmap='viridis', interpolation='nearest', aspect='auto')
                ax3.set_xticks(np.arange(len(x_labels)))
                ax3.set_yticks(np.arange(len(y_labels)))
                ax3.set_xticklabels(x_labels)
                ax3.set_yticklabels(y_labels)
                ax3.set_xlabel(param1)
                ax3.set_ylabel(param2)
                ax3.set_title(f'Performance Heatmap: {param1} vs {param2}')
                plt.colorbar(im, ax=ax3, label='Performance Metric')
                
                # Rotate x-axis labels if necessary
                plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            else:
                ax3.text(0.5, 0.5, "Heatmap data not available", ha='center', va='center', transform=ax3.transAxes)
            
            # Second heatmap (if available)
            ax4 = fig.add_subplot(gs[1, 1])
            if len(heatmaps) > 1:
                hm2 = heatmaps[1]
                param1 = hm2['param_x']
                param2 = hm2['param_y']
                values = hm2['values']
                
                # Create heatmap matrix
                x_labels = sorted(list(set([point[0] for point in values])))
                y_labels = sorted(list(set([point[1] for point in values])))
                z_matrix = np.zeros((len(y_labels), len(x_labels)))
                
                for x, y, z in values:
                    x_idx = x_labels.index(x)
                    y_idx = y_labels.index(y)
                    z_matrix[y_idx, x_idx] = z
                
                # Plot heatmap
                im = ax4.imshow(z_matrix, cmap='viridis', interpolation='nearest', aspect='auto')
                ax4.set_xticks(np.arange(len(x_labels)))
                ax4.set_yticks(np.arange(len(y_labels)))
                ax4.set_xticklabels(x_labels)
                ax4.set_yticklabels(y_labels)
                ax4.set_xlabel(param1)
                ax4.set_ylabel(param2)
                ax4.set_title(f'Performance Heatmap: {param1} vs {param2}')
                plt.colorbar(im, ax=ax4, label='Performance Metric')
                
                # Rotate x-axis labels if necessary
                plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
            else:
                ax4.text(0.5, 0.5, "Second heatmap not available", ha='center', va='center', transform=ax4.transAxes)
        else:
            # No heatmaps available
            ax3 = fig.add_subplot(gs[1, :])
            ax3.text(0.5, 0.5, "Parameter heatmap data not available", ha='center', va='center', transform=ax3.transAxes)
        
        # Add optimization summary
        best_params = results.get('best_parameters', {})
        best_metrics = results.get('best_metrics', {})
        
        if best_params and best_metrics:
            param_text = ", ".join([f"{k}={v}" for k, v in best_params.items()])
            metric_value = list(best_metrics.values())[0] if best_metrics else "N/A"
            metric_name = list(best_metrics.keys())[0] if best_metrics else "metric"
            
            summary_text = f"Best parameters: {param_text} | {metric_name}: {metric_value:.4f}"
            fig.text(0.5, 0.01, summary_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        
        # Adjust layout
        fig.tight_layout()
        
        return fig

 
        
    def _generate_backtest_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report for backtest results"""
        self._log('debug', "Generating HTML backtest report")
        
        # Extract metadata
        metadata = results.get('_metadata', {})
        saved_at = metadata.get('saved_at', self.current_utc)
        saved_by = metadata.get('saved_by', self.current_user)
        
        # Extract parameters
        params = results.get('parameters', {})
        atr_length = params.get('atr_length', 0)
        factor = params.get('factor', 0)
        buffer_multiplier = params.get('buffer_multiplier', 0)
        
        # Extract performance metrics
        performance = results.get('performance', {})
        trade_count = performance.get('trade_count', 0)
        win_rate = performance.get('win_rate', 0) * 100  # Convert to percentage
        profit_factor = performance.get('profit_factor', 0)
        total_profit = performance.get('total_profit_pct', 0)
        max_drawdown = performance.get('max_drawdown', 0)
        sharpe_ratio = performance.get('sharpe_ratio', 0)
        sortino_ratio = performance.get('sortino_ratio', 0)
        
        # Extract trades
        trades = results.get('trades', [])
        
        # Start building HTML
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SuperTrend Backtest Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .header {{
                    margin-bottom: 30px;
                }}
                .summary {{
                    display: flex;
                    flex-wrap: wrap;
                    margin: 20px 0;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                }}
                .metric {{
                    flex: 1;
                    min-width: 200px;
                    margin: 10px;
                }}
                .metric h3 {{
                    margin-bottom: 5px;
                    font-size: 16px;
                }}
                .metric p {{
                    font-size: 20px;
                    font-weight: bold;
                    margin: 5px 0;
                }}
                .positive {{
                    color: green;
                }}
                .negative {{
                    color: red;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .footer {{
                    margin-top: 30px;
                    font-size: 12px;
                    color: #666;
                    text-align: center;
                }}
                .chart-placeholder {{
                    background-color: #f8f9fa;
                    border: 1px dashed #ccc;
                    height: 400px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>SuperTrend Backtest Report</h1>
                <p>Report generated on {self.current_utc} by {self.current_user}</p>
                <p>Backtest saved on {saved_at} by {saved_by}</p>
            </div>
            
            <h2>Strategy Configuration</h2>
            <div class="summary">
                <div class="metric">
                    <h3>ATR Length</h3>
                    <p>{atr_length}</p>
                </div>
                <div class="metric">
                    <h3>Multiplier</h3>
                    <p>{factor}</p>
                </div>
                <div class="metric">
                    <h3>Buffer Multiplier</h3>
                    <p>{buffer_multiplier}</p>
                </div>
            </div>
            
            <h2>Performance Metrics</h2>
            <div class="summary">
                <div class="metric">
                    <h3>Total Trades</h3>
                    <p>{trade_count}</p>
                </div>
                <div class="metric">
                    <h3>Win Rate</h3>
                    <p>{win_rate:.2f}%</p>
                </div>
                <div class="metric">
                    <h3>Profit Factor</h3>
                    <p class="{'' if profit_factor <= 1 else 'positive'}">{profit_factor:.2f}</p>
                </div>
                <div class="metric">
                    <h3>Total Return</h3>
                    <p class="{'negative' if total_profit < 0 else 'positive'}">{total_profit:.2f}%</p>
                </div>
                <div class="metric">
                    <h3>Max Drawdown</h3>
                    <p class="negative">{max_drawdown:.2f}%</p>
                </div>
                <div class="metric">
                    <h3>Sharpe Ratio</h3>
                    <p class="{'' if sharpe_ratio < 1 else 'positive'}">{sharpe_ratio:.2f}</p>
                </div>
                <div class="metric">
                    <h3>Sortino Ratio</h3>
                    <p class="{'' if sortino_ratio < 1 else 'positive'}">{sortino_ratio:.2f}</p>
                </div>
            </div>
        """
        
        # Chart section (placeholder)
        html += """
            <h2>Backtest Analysis Charts</h2>
            <div class="chart-placeholder">
                <p>Chart images would be embedded here when viewing in the application</p>
            </div>
        """
        
        # Trade list
        html += """
            <h2>Trade List</h2>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Type</th>
                        <th>Entry Time</th>
                        <th>Entry Price</th>
                        <th>Exit Time</th>
                        <th>Exit Price</th>
                        <th>Profit/Loss</th>
                        <th>Profit %</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add trade rows
        for i, trade in enumerate(trades[:100]):  # Limit to first 100 trades
            entry_time = trade.get('entry_time', '')
            entry_price = trade.get('entry_price', 0)
            exit_time = trade.get('exit_time', '')
            exit_price = trade.get('exit_price', 0)
            profit = trade.get('profit', 0)
            profit_pct = trade.get('profit_pct', 0) * 100  # Convert to percentage
            position_type = trade.get('position_type', '')
            
            profit_class = 'positive' if profit > 0 else 'negative' if profit < 0 else ''
            
            html += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{position_type}</td>
                    <td>{entry_time}</td>
                    <td>{entry_price:.4f}</td>
                    <td>{exit_time if exit_time else 'Open'}</td>
                    <td>{exit_price:.4f if exit_price else '-'}</td>
                    <td class="{profit_class}">{profit:.4f}</td>
                    <td class="{profit_class}">{profit_pct:.2f}%</td>
                </tr>
            """
        
        # Close table and add footer
        html += """
                </tbody>
            </table>
        """
        
        # Add additional notes if trade count > 100
        if len(trades) > 100:
            html += f"<p>Showing 100 of {len(trades)} trades. Export the full results for complete trade list.</p>"
        
        # Footer
        html += f"""
            <div class="footer">
                <p>Generated by SuperTrendAnalyzer v1.0 | {self.current_utc}</p>
            </div>
        </body>
        </html>
        """
        
        return html


    def _generate_optimization_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report for optimization results"""
        self._log('debug', "Generating HTML optimization report")
        
        # Extract metadata
        metadata = results.get('_metadata', {})
        saved_at = metadata.get('saved_at', '2025-06-24 15:28:21')  # Updated with current time
        saved_by = metadata.get('saved_by', 'arullr001')  # Updated with current username
        
        # Extract optimization settings
        opt_settings = results.get('optimization_settings', {})
        optimizer_type = opt_settings.get('optimizer_type', 'Unknown')
        iterations = opt_settings.get('iterations', 0)
        fitness_function = opt_settings.get('fitness_function', 'Unknown')
        parameter_space = opt_settings.get('parameter_space', {})
        
        # Extract best results
        best_parameters = results.get('best_parameters', {})
        best_metrics = results.get('best_metrics', {})
        
        # Extract top results
        top_results = results.get('top_results', [])
        
        # Start building HTML
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SuperTrend Optimization Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .header {{
                    margin-bottom: 30px;
                }}
                .summary {{
                    display: flex;
                    flex-wrap: wrap;
                    margin: 20px 0;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                }}
                .metric {{
                    flex: 1;
                    min-width: 200px;
                    margin: 10px;
                }}
                .metric h3 {{
                    margin-bottom: 5px;
                    font-size: 16px;
                }}
                .metric p {{
                    font-size: 20px;
                    font-weight: bold;
                    margin: 5px 0;
                }}
                .parameter-space {{
                    margin: 20px 0;
                }}
                .parameter {{
                    margin-bottom: 10px;
                }}
                .positive {{
                    color: green;
                }}
                .negative {{
                    color: red;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .footer {{
                    margin-top: 30px;
                    font-size: 12px;
                    color: #666;
                    text-align: center;
                }}
                .chart-placeholder {{
                    background-color: #f8f9fa;
                    border: 1px dashed #ccc;
                    height: 400px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 20px 0;
                }}
                .insights {{
                    background-color: #e9f7ef;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .insights ul {{
                    margin-top: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>SuperTrend Optimization Report</h1>
                <p>Report generated on {self.current_utc} by {self.current_user}</p>
                <p>Optimization saved on {saved_at} by {saved_by}</p>
            </div>
            
            <h2>Optimization Configuration</h2>
            <div class="summary">
                <div class="metric">
                    <h3>Optimizer</h3>
                    <p>{optimizer_type}</p>
                </div>
                <div class="metric">
                    <h3>Iterations</h3>
                    <p>{iterations}</p>
                </div>
                <div class="metric">
                    <h3>Fitness Function</h3>
                    <p>{fitness_function}</p>
                </div>
            </div>
            
            <h3>Parameter Space</h3>
            <div class="parameter-space">
        """
        
        
        
        # Add parameter space
        for param_name, param_range in parameter_space.items():
            if isinstance(param_range, list):
                if len(param_range) == 3 and param_range[2] != 1:  # min, max, step
                    html += f"""
                    <div class="parameter">
                        <strong>{param_name}</strong>: Range [{param_range[0]} to {param_range[1]}] with step {param_range[2]}
                    </div>
                    """
                else:
                    html += f"""
                    <div class="parameter">
                        <strong>{param_name}</strong>: {param_range}
                    </div>
                    """
            else:
                html += f"""
                <div class="parameter">
                    <strong>{param_name}</strong>: {param_range}
                </div>
                """
        
        # Best results
        html += """
            </div>
            
            <h2>Optimization Results</h2>
            <h3>Best Parameters</h3>
            <div class="summary">
        """
        
        # Add best parameters
        for param_name, param_value in best_parameters.items():
            html += f"""
            <div class="metric">
                <h3>{param_name}</h3>
                <p>{param_value}</p>
            </div>
            """
        
        # Best metrics
        html += """
            </div>
            
            <h3>Performance Metrics</h3>
            <div class="summary">
        """
        
        # Add best metrics
        for metric_name, metric_value in best_metrics.items():
            # Determine if metric is positive or negative
            css_class = ''
            if 'profit' in metric_name.lower() or 'return' in metric_name.lower():
                css_class = 'positive' if metric_value > 0 else 'negative'
            elif 'drawdown' in metric_name.lower():
                css_class = 'negative'
            elif 'sharpe' in metric_name.lower() or 'sortino' in metric_name.lower():
                css_class = 'positive' if metric_value > 1 else ''
                
            html += f"""
            <div class="metric">
                <h3>{metric_name}</h3>
                <p class="{css_class}">{metric_value:.4f}</p>
            </div>
            """
        
        # Chart section (placeholder)
        html += """
            </div>
            
            <h2>Optimization Analysis</h2>
            <div class="chart-placeholder">
                <p>Chart images would be embedded here when viewing in the application</p>
            </div>
        """
                # Add insights section
        param_analysis = results.get('parameter_analysis', {})
        if param_analysis:
            html += """
            <h3>Parameter Insights</h3>
            <div class="insights">
                <ul>
            """
            
            # Add insights based on parameter analysis
            if 'parameter_importance' in param_analysis:
                importance = param_analysis['parameter_importance']
                most_important = max(importance.items(), key=lambda x: x[1])
                html += f"""
                <li>Parameter <strong>{most_important[0]}</strong> has the highest impact on performance.</li>
                """
            
            if 'correlations' in param_analysis:
                correlations = param_analysis['correlations']
                # Find highest positive correlation
                pos_corr = [(k, v) for k, v in correlations.items() if v > 0]
                if pos_corr:
                    highest_pos = max(pos_corr, key=lambda x: x[1])
                    html += f"""
                    <li>Increasing <strong>{highest_pos[0]}</strong> generally leads to better performance (correlation: {highest_pos[1]:.2f}).</li>
                    """
                
                # Find highest negative correlation
                neg_corr = [(k, v) for k, v in correlations.items() if v < 0]
                if neg_corr:
                    highest_neg = min(neg_corr, key=lambda x: x[1])
                    html += f"""
                    <li>Increasing <strong>{highest_neg[0]}</strong> generally leads to worse performance (correlation: {highest_neg[1]:.2f}).</li>
                    """
            
            # Close insights
            html += """
                </ul>
            </div>
            """
        
        # Top results
        html += """
            <h3>Top Parameter Combinations</h3>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
        """
        
        # Add parameter column headers
        if top_results and len(top_results) > 0:
            # Get parameter names from the first result
            param_names = list(top_results[0].get('parameters', {}).keys())
            metric_names = list(top_results[0].get('metrics', {}).keys())
            
            # Add parameter headers
            for param in param_names:
                html += f"<th>{param}</th>"
            
            # Add metric headers
            for metric in metric_names:
                html += f"<th>{metric}</th>"
            
            html += """
                    </tr>
                </thead>
                <tbody>
            """
            
            # Add rows for top results
            for i, result in enumerate(top_results[:10]):  # Top 10 results
                params = result.get('parameters', {})
                metrics = result.get('metrics', {})
                
                html += f"""
                    <tr>
                        <td>{i+1}</td>
                """
                
                # Add parameter values
                for param in param_names:
                    html += f"<td>{params.get(param, '')}</td>"
                
                # Add metric values with formatting
                for metric_name in metric_names:
                    metric_value = metrics.get(metric_name, 0)
                    
                    # Determine CSS class
                    css_class = ''
                    if 'profit' in metric_name.lower() or 'return' in metric_name.lower():
                        css_class = 'positive' if metric_value > 0 else 'negative'
                    elif 'drawdown' in metric_name.lower():
                        css_class = 'negative'
                    elif 'sharpe' in metric_name.lower() or 'sortino' in metric_name.lower():
                        css_class = 'positive' if metric_value > 1 else ''
                    
                    html += f"<td class='{css_class}'>{metric_value:.4f}</td>"
                
                html += "</tr>"
        else:
            html += """
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td colspan="99">No top results available</td>
                    </tr>
            """
        
        # Close table
        html += """
                </tbody>
            </table>
        """
        
        # Footer
        html += f"""
            <div class="footer">
                <p>Generated by SuperTrendAnalyzer v1.0 | {self.current_utc}</p>
            </div>
        </body>
        </html>
        """
        
        return html

    
    def _generate_backtest_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate Markdown report for backtest results"""
        self._log('debug', "Generating Markdown backtest report")
        
        # Extract metadata
        metadata = results.get('_metadata', {})
        saved_at = metadata.get('saved_at', self.current_utc)
        saved_by = metadata.get('saved_by', self.current_user)
        
        # Extract parameters
        params = results.get('parameters', {})
        atr_length = params.get('atr_length', 0)
        factor = params.get('factor', 0)
        buffer_multiplier = params.get('buffer_multiplier', 0)
        
        # Extract performance metrics
        performance = results.get('performance', {})
        trade_count = performance.get('trade_count', 0)
        win_rate = performance.get('win_rate', 0) * 100  # Convert to percentage
        profit_factor = performance.get('profit_factor', 0)
        total_profit = performance.get('total_profit_pct', 0)
        max_drawdown = performance.get('max_drawdown', 0)
        sharpe_ratio = performance.get('sharpe_ratio', 0)
        sortino_ratio = performance.get('sortino_ratio', 0)
        
        # Extract trades
        trades = results.get('trades', [])
        
        # Start building Markdown
        md = f"""# SuperTrend Backtest Report

Report generated on {self.current_utc} by {self.current_user}  
Backtest saved on {saved_at} by {saved_by}

## Strategy Configuration

- **ATR Length:** {atr_length}
- **Multiplier:** {factor}
- **Buffer Multiplier:** {buffer_multiplier}

## Performance Metrics

- **Total Trades:** {trade_count}
- **Win Rate:** {win_rate:.2f}%
- **Profit Factor:** {profit_factor:.2f}
- **Total Return:** {total_profit:.2f}%
- **Max Drawdown:** {max_drawdown:.2f}%
- **Sharpe Ratio:** {sharpe_ratio:.2f}
- **Sortino Ratio:** {sortino_ratio:.2f}

## Trade List

| # | Type | Entry Time | Entry Price | Exit Time | Exit Price | Profit/Loss | Profit % |
|---|------|------------|------------|-----------|-----------|------------|----------|
"""
        
        # Add trade rows
        for i, trade in enumerate(trades[:50]):  # Limit to first 50 trades in markdown
            entry_time = trade.get('entry_time', '')
            entry_price = trade.get('entry_price', 0)
            exit_time = trade.get('exit_time', '')
            exit_price = trade.get('exit_price', 0)
            profit = trade.get('profit', 0)
            profit_pct = trade.get('profit_pct', 0) * 100  # Convert to percentage
            position_type = trade.get('position_type', '')
            
            md += f"| {i+1} | {position_type} | {entry_time} | {entry_price:.4f} | {exit_time if exit_time else 'Open'} | {exit_price:.4f if exit_price else '-'} | {profit:.4f} | {profit_pct:.2f}% |\n"
        
        # Add notes if trade count > 50
        if len(trades) > 50:
            md += f"\n*Showing 50 of {len(trades)} trades. Export the full results for complete trade list.*\n"
        
        # Add footer
        md += f"\n\n---\nGenerated by SuperTrendAnalyzer v1.0 | {self.current_utc}"
        
        return md


    def _generate_backtest_text_report(self, results: Dict[str, Any]) -> str:
        """Generate plain text report for backtest results"""
        self._log('debug', "Generating plain text backtest report")
        
        # Extract metadata
        metadata = results.get('_metadata', {})
        saved_at = metadata.get('saved_at', self.current_utc)
        saved_by = metadata.get('saved_by', self.current_user)
        
        # Extract parameters
        params = results.get('parameters', {})
        atr_length = params.get('atr_length', 0)
        factor = params.get('factor', 0)
        buffer_multiplier = params.get('buffer_multiplier', 0)
        
        # Extract performance metrics
        performance = results.get('performance', {})
        trade_count = performance.get('trade_count', 0)
        win_rate = performance.get('win_rate', 0) * 100  # Convert to percentage
        profit_factor = performance.get('profit_factor', 0)
        total_profit = performance.get('total_profit_pct', 0)
        max_drawdown = performance.get('max_drawdown', 0)
        sharpe_ratio = performance.get('sharpe_ratio', 0)
        sortino_ratio = performance.get('sortino_ratio', 0)
        
        # Extract trades
        trades = results.get('trades', [])
        
        # Build text report - header
        text = f"""
=============================================================
                SUPERTREND BACKTEST REPORT
=============================================================
Report generated on {self.current_utc} by {self.current_user}
Backtest saved on {saved_at} by {saved_by}
=============================================================

STRATEGY CONFIGURATION
-------------------------------------------------------------
ATR Length: {atr_length}
Multiplier: {factor}
Buffer Multiplier: {buffer_multiplier}

PERFORMANCE METRICS
-------------------------------------------------------------
Total Trades: {trade_count}
Win Rate: {win_rate:.2f}%
Profit Factor: {profit_factor:.2f}
Total Return: {total_profit:.2f}%
Max Drawdown: {max_drawdown:.2f}%
Sharpe Ratio: {sharpe_ratio:.2f}
Sortino Ratio: {sortino_ratio:.2f}

TRADE LIST
-------------------------------------------------------------
"""
        
        # Add column headers for trades
        text += f"{'#':3s} {'TYPE':6s} {'ENTRY TIME':20s} {'ENTRY PRICE':12s} {'EXIT TIME':20s} {'EXIT PRICE':12s} {'PROFIT':10s} {'PROFIT %':10s}\n"
        text += "-" * 95 + "\n"
        
        # Add trade rows
        for i, trade in enumerate(trades[:30]):  # Limit to first 30 trades in text report
            entry_time = trade.get('entry_time', '')[:19]  # Truncate microseconds if present
            entry_price = trade.get('entry_price', 0)
            exit_time = trade.get('exit_time', '')[:19] if trade.get('exit_time', '') else 'Open'
            exit_price = trade.get('exit_price', 0)
            profit = trade.get('profit', 0)
            profit_pct = trade.get('profit_pct', 0) * 100  # Convert to percentage
            position_type = trade.get('position_type', '')[:4]  # Truncate to 'LONG' or 'SHOR'
            
            text += f"{i+1:3d} {position_type:6s} {entry_time:20s} {entry_price:12.4f} {exit_time:20s} "
            text += f"{exit_price:12.4f if exit_price else 0:12.4f} {profit:10.4f} {profit_pct:10.2f}%\n"
        
        # Add notes if trade count > 30
        if len(trades) > 30:
            text += f"\nShowing 30 of {len(trades)} trades. Export the full results for complete trade list.\n"
        
        # Add footer
        text += f"""
=============================================================
Generated by SuperTrendAnalyzer v1.0 | {self.current_utc}
=============================================================
"""
        
        return text


    def _generate_optimization_text_report(self, results: Dict[str, Any]) -> str:
        """Generate plain text report for optimization results"""
        self._log('debug', "Generating plain text optimization report")
        
        # Extract metadata
        metadata = results.get('_metadata', {})
        saved_at = metadata.get('saved_at', self.current_utc)
        saved_by = metadata.get('saved_by', self.current_user)
        
        # Extract optimization settings
        opt_settings = results.get('optimization_settings', {})
        optimizer_type = opt_settings.get('optimizer_type', 'Unknown')
        iterations = opt_settings.get('iterations', 0)
        fitness_function = opt_settings.get('fitness_function', 'Unknown')
        parameter_space = opt_settings.get('parameter_space', {})
        
        # Extract best results
        best_parameters = results.get('best_parameters', {})
        best_metrics = results.get('best_metrics', {})
        
        # Extract top results
        top_results = results.get('top_results', [])
        
        # Build text report - header
        text = f"""
=============================================================
              SUPERTREND OPTIMIZATION REPORT
=============================================================
Report generated on {self.current_utc} by {self.current_user}
Optimization saved on {saved_at} by {saved_by}
=============================================================

OPTIMIZATION CONFIGURATION
-------------------------------------------------------------
Optimizer: {optimizer_type}
Iterations: {iterations}
Fitness Function: {fitness_function}

PARAMETER SPACE
-------------------------------------------------------------
"""
        
        # Add parameter space
        for param_name, param_range in parameter_space.items():
            if isinstance(param_range, list):
                if len(param_range) == 3 and param_range[2] != 1:  # min, max, step
                    text += f"{param_name}: Range [{param_range[0]} to {param_range[1]}] with step {param_range[2]}\n"
                else:
                    text += f"{param_name}: {param_range}\n"
            else:
                text += f"{param_name}: {param_range}\n"
        
        # Best results
        text += f"""
OPTIMIZATION RESULTS
-------------------------------------------------------------
Best Parameters:
"""
        
        # Add best parameters
        for param_name, param_value in best_parameters.items():
            text += f"{param_name}: {param_value}\n"
        
        # Best metrics
        text += f"\nPerformance Metrics:\n"
        
        # Add best metrics
        for metric_name, metric_value in best_metrics.items():
            text += f"{metric_name}: {metric_value:.4f}\n"
        
        # Parameter analysis
        param_analysis = results.get('parameter_analysis', {})
        if param_analysis:
            text += f"""
PARAMETER ANALYSIS
-------------------------------------------------------------
"""
            
            # Add parameter importance if available
            if 'parameter_importance' in param_analysis:
                text += "Parameter Importance:\n"
                importance = param_analysis['parameter_importance']
                for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                    text += f"{param}: {score:.4f}\n"
            
            # Add correlations if available
            if 'correlations' in param_analysis:
                text += "\nParameter Correlations with Performance:\n"
                correlations = param_analysis['correlations']
                for param, score in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
                    direction = "positive" if score > 0 else "negative"
                    text += f"{param}: {score:.4f} ({direction} correlation)\n"
            
            # Add insights
            text += "\nParameter Insights:\n"
            
            if 'parameter_importance' in param_analysis:
                importance = param_analysis['parameter_importance']
                most_important = max(importance.items(), key=lambda x: x[1])
                text += f"- Parameter {most_important[0]} has the highest impact on performance.\n"
            
            if 'correlations' in param_analysis:
                correlations = param_analysis['correlations']
                # Find highest positive correlation
                pos_corr = [(k, v) for k, v in correlations.items() if v > 0]
                if pos_corr:
                    highest_pos = max(pos_corr, key=lambda x: x[1])
                    text += f"- Increasing {highest_pos[0]} generally leads to better performance (correlation: {highest_pos[1]:.2f}).\n"
                
                # Find highest negative correlation
                neg_corr = [(k, v) for k, v in correlations.items() if v < 0]
                if neg_corr:
                    highest_neg = min(neg_corr, key=lambda x: x[1])
                    text += f"- Increasing {highest_neg[0]} generally leads to worse performance (correlation: {highest_neg[1]:.2f}).\n"
        
        # Top results
        text += f"""
TOP PARAMETER COMBINATIONS
-------------------------------------------------------------
"""
        
        if top_results and len(top_results) > 0:
            # Get parameter names from the first result
            param_names = list(top_results[0].get('parameters', {}).keys())
            metric_names = list(top_results[0].get('metrics', {}).keys())
            
            # Create format string for header and rows
            col_width = 10
            header_format = "{:<5s}" + "".join([f"{{:<{col_width}s}}" for _ in range(len(param_names) + len(metric_names))])
            row_format = "{:<5d}" + "".join([f"{{:<{col_width}}}" for _ in range(len(param_names))]) + "".join([f"{{:<{col_width}.4f}}" for _ in range(len(metric_names))])
            
            # Create header
            header = header_format.format("Rank", *param_names, *metric_names)
            text += header + "\n"
            text += "-" * len(header) + "\n"
            
            # Add rows for top results
            for i, result in enumerate(top_results[:10]):  # Top 10 results
                params = result.get('parameters', {})
                metrics = result.get('metrics', {})
                
                param_values = [params.get(param, '') for param in param_names]
                metric_values = [metrics.get(metric, 0) for metric in metric_names]
                
                try:
                    text += row_format.format(i+1, *param_values, *metric_values) + "\n"
                except Exception as e:
                    # Fallback if formatting fails
                    text += f"Rank {i+1}: {params} -> {metrics}\n"
        else:
            text += "No top results available\n"
        
        # Add footer
        text += f"""
=============================================================
Generated by SuperTrendAnalyzer v1.0 | {self.current_utc}
=============================================================
"""
        
        return text

    def set_plot_style(self):
        """Set consistent style for all plots"""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.grid'] = True
    
    
# ==============================================================================
# GUI FRAMEWORK
# ==============================================================================

class OptimizationWorker(QThread):
    progress_signal = pyqtSignal(dict)  # Signal for progress updates
    result_signal = pyqtSignal(dict)    # Signal for final results
    
    def __init__(self, method, df, param_ranges, optimization_metric, min_trades, time_exit_hours, parallelism):
        super().__init__()
        self.method = method
        self.df = df
        self.param_ranges = param_ranges
        self.optimization_metric = optimization_metric
        self.min_trades = min_trades
        self.time_exit_hours = time_exit_hours
        self.parallelism = parallelism
        self.stop_requested = False
        

    def run(self):
        try:
            # Get the current process name for debugging
            process_name = multiprocessing.current_process().name
            print(f"Starting optimization in process: {process_name}")
            
            # Create a wrapper for the progress_callback to ensure thread safety
            def thread_safe_progress_callback(progress_info):
                if not self.stop_requested:
                    self.progress_signal.emit(progress_info)
            
            # Select optimizer based on method
            if self.method == "Grid Search":
                # Force thread parallelism regardless of what was selected
                self.parallelism = "thread"  # Override with thread parallelism 
                optimizer = GridSearchOptimizer(progress_callback=thread_safe_progress_callback)
                
                # Log the override
                print(f"Forcing thread parallelism for optimization")
                
                results = optimizer.optimize(
                    self.df, self.param_ranges, self.optimization_metric,
                    self.min_trades, self.time_exit_hours, self.parallelism
                )
            elif self.method == "Bayesian Optimization":
                # Also force thread parallelism for Bayesian optimization
                optimizer = BayesianOptimizer(progress_callback=thread_safe_progress_callback)
                
                print(f"Forcing thread parallelism for Bayesian optimization")
                
                # Note: Bayesian optimization doesn't have a parallelism parameter directly,
                # but the internal implementation should use threading
                results = optimizer.optimize(
                    self.df, self.param_ranges, self.optimization_metric,
                    self.min_trades, self.time_exit_hours
                )
            elif self.method == "Genetic Algorithm":
                # Also force thread parallelism for Genetic algorithm
                optimizer = GeneticOptimizer(progress_callback=thread_safe_progress_callback)
                
                print(f"Forcing thread parallelism for Genetic algorithm")
                
                # The genetic algorithm should be modified to use threading internally
                results = optimizer.optimize(
                    self.df, self.param_ranges, self.optimization_metric,
                    self.min_trades, self.time_exit_hours
                )
                
            # Check if stop was requested before emitting final results
            if not self.stop_requested:
                self.result_signal.emit(results)
            
        except Exception as e:
            print(f"ERROR in worker: {str(e)}")
            import traceback
            print(traceback.format_exc())
            self.progress_signal.emit({'error': str(e), 'traceback': traceback.format_exc()})
            # Signal that an error occurred
            self.progress_signal.emit({'error': str(e)})
            print(f"Worker thread exception: {str(e)}\n{traceback.format_exc()}")
    
    def update_progress(self, progress_info):
        """Callback function for optimizer to report progress"""
        if not self.stop_requested:
            # Print to console to confirm progress is being generated
            print(f"Progress update: {progress_info.get('completed', 0)}/{progress_info.get('total', 0)}")
            
            # Make sure we emit the signal
            self.progress_signal.emit(progress_info)
    
    def stop(self):
        """Stop the optimization process"""
        self.stop_requested = True
        

class SuperTrendGUI(QMainWindow):
    """
    PyQt5-based GUI for SuperTrend Backtester
    """
    
    def __init__(self, log_manager: LogManager = None):
        super().__init__()
        
        # Initialize variables
        self.log = log_manager
        self.df = None
        self.df_with_supertrend = None
        self.backtester = Backtester(log_manager)
        self.supertrend = SuperTrend(log_manager)
        self.data_loader = DataLoader(log_manager)
        self.data_analyzer = DataAnalyzer(log_manager)
        self.report_generator = ReportGenerator(log_manager)
        self.results_storage = ResultsStorage(log_manager=log_manager)
        self.backtest_results = None
        self.optimization_results = None
        self.chart_figure = None
        
        # Current time and user
        self.current_utc = "2025-06-25 15:37:58"
        self.current_user = "arullr001"
        
        # Setup GUI
        self.setup_ui()
        
        # Log initialization
        self._log('info', "SuperTrend GUI initialized")
    
    def _log(self, level: str, message: str):
        """Log message if log_manager is available"""
        if self.log:
            if level == 'info':
                self.log.info(message)
            elif level == 'debug':
                self.log.debug(message)
            elif level == 'warning':
                self.log.warning(message)
            elif level == 'error':
                self.log.error(message)
                
                
    def setup_ui(self):
        """Set up the main user interface"""
        # Set window properties
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.on_tab_changed)
        main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.setup_data_tab()
        self.setup_parameters_tab()
        self.setup_backtest_tab()
        self.setup_optimization_tab()
        self.setup_results_tab()
        self.setup_chart_tab()
        
        # Create status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        # Create menu bar
        self.setup_menu()
    
    def setup_menu(self):
        """Set up the menu bar"""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("File")
        
        load_data_action = QAction("Load Data...", self)
        load_data_action.triggered.connect(self.on_load_data)
        load_data_action.setShortcut("Ctrl+O")
        file_menu.addAction(load_data_action)
        
        save_results_action = QAction("Save Results...", self)
        save_results_action.triggered.connect(self.on_save_results)
        save_results_action.setShortcut("Ctrl+S")
        file_menu.addAction(save_results_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut("Ctrl+Q")
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menu_bar.addMenu("Tools")
        
        run_backtest_action = QAction("Run Backtest", self)
        run_backtest_action.triggered.connect(self.on_run_backtest)
        tools_menu.addAction(run_backtest_action)
        
        run_optimization_action = QAction("Run Optimization", self)
        run_optimization_action.triggered.connect(self.on_run_optimization)
        tools_menu.addAction(run_optimization_action)
        
        tools_menu.addSeparator()
        
        export_report_action = QAction("Export Report...", self)
        export_report_action.triggered.connect(self.on_export_report)
        tools_menu.addAction(export_report_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.on_about)
        help_menu.addAction(about_action)
    
    def setup_data_tab(self):
        """Set up the Data tab"""
        data_tab = QWidget()
        layout = QVBoxLayout(data_tab)
        
        # Top controls
        top_controls = QHBoxLayout()
        
        # File selection
        file_group = QGroupBox("Data Source")
        file_layout = QVBoxLayout()
        
        file_input_layout = QHBoxLayout()
        self.file_path_input = QLineEdit()
        self.file_path_input.setReadOnly(True)
        self.file_path_input.setPlaceholderText("Select a data file...")
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.on_browse_file)
        
        file_input_layout.addWidget(self.file_path_input)
        file_input_layout.addWidget(browse_button)
        
        # Add supported formats label
        format_label = QLabel(f"Supported formats: {', '.join(['csv', 'xlsx', 'parquet', 'feather'])}")
        
        file_layout.addLayout(file_input_layout)
        file_layout.addWidget(format_label)
        
        file_group.setLayout(file_layout)
        top_controls.addWidget(file_group)
        
        # Data preview
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout()
        
        self.data_preview_table = QTableWidget()
        self.data_preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        preview_layout.addWidget(self.data_preview_table)
        
        preview_group.setLayout(preview_layout)
        
        # Data stats
        stats_group = QGroupBox("Data Analysis")
        stats_layout = QVBoxLayout()
        
        self.data_stats_text = QTextEdit()
        self.data_stats_text.setReadOnly(True)
        
        stats_layout.addWidget(self.data_stats_text)
        
        stats_group.setLayout(stats_layout)
        
        # Add all elements to layout
        layout.addLayout(top_controls)
        
        # Create a splitter for the preview and stats
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(preview_group)
        splitter.addWidget(stats_group)
        splitter.setSizes([500, 300])  # Set initial sizes
        
        layout.addWidget(splitter)
        
        # Add tab
        self.tabs.addTab(data_tab, "Data")
    
    def setup_parameters_tab(self):
        """Set up the Parameters tab"""
        parameters_tab = QWidget()
        layout = QVBoxLayout(parameters_tab)
        
        # Parameter groups
        params_group = QGroupBox("SuperTrend Parameters")
        params_layout = QGridLayout()
        
        # ATR Length
        params_layout.addWidget(QLabel("ATR Length:"), 0, 0)
        self.atr_length_input = QSpinBox()
        self.atr_length_input.setRange(1, 100)
        self.atr_length_input.setValue(14)
        params_layout.addWidget(self.atr_length_input, 0, 1)
        
        # Factor
        params_layout.addWidget(QLabel("Factor:"), 1, 0)
        self.factor_input = QDoubleSpinBox()
        self.factor_input.setRange(0.1, 10.0)
        self.factor_input.setSingleStep(0.1)
        self.factor_input.setValue(3.0)
        params_layout.addWidget(self.factor_input, 1, 1)
        
        # Buffer Multiplier
        params_layout.addWidget(QLabel("Buffer Multiplier:"), 2, 0)
        self.buffer_input = QDoubleSpinBox()
        self.buffer_input.setRange(0.1, 10.0)
        self.buffer_input.setSingleStep(0.1)
        self.buffer_input.setValue(0.3)
        params_layout.addWidget(self.buffer_input, 2, 1)
        
        params_group.setLayout(params_layout)
        
        # Trade parameters
        trade_group = QGroupBox("Trade Parameters")
        trade_layout = QGridLayout()
        
        # Hard Stop
        trade_layout.addWidget(QLabel("Hard Stop Distance:"), 0, 0)
        self.stop_input = QSpinBox()
        self.stop_input.setRange(0, 1000)
        self.stop_input.setValue(50)
        trade_layout.addWidget(self.stop_input, 0, 1)
        
        # Time Exit
        trade_layout.addWidget(QLabel("Time Exit (hours):"), 1, 0)
        self.time_exit_input = QDoubleSpinBox()
        self.time_exit_input.setRange(0, 240)
        self.time_exit_input.setValue(48.0)
        trade_layout.addWidget(self.time_exit_input, 1, 1)
        
        # Initial Capital
        trade_layout.addWidget(QLabel("Initial Capital:"), 2, 0)
        self.capital_input = QDoubleSpinBox()
        self.capital_input.setRange(100, 1000000)
        self.capital_input.setSingleStep(1000)
        self.capital_input.setValue(10000)
        trade_layout.addWidget(self.capital_input, 2, 1)
        
        trade_group.setLayout(trade_layout)
        
        # Parameter recommendations section
        recommendation_group = QGroupBox("Parameter Recommendations")
        recommendation_layout = QVBoxLayout()
        
        self.recommendation_text = QTextEdit()
        self.recommendation_text.setReadOnly(True)
        
        recommendation_layout.addWidget(self.recommendation_text)
        
        recommendation_group.setLayout(recommendation_layout)
        
        # Calculate button
        calculate_button = QPushButton("Calculate SuperTrend")
        calculate_button.clicked.connect(self.on_calculate_supertrend)
        
        # Add all elements to layout
        param_section = QHBoxLayout()
        param_section.addWidget(params_group)
        param_section.addWidget(trade_group)
        
        layout.addLayout(param_section)
        layout.addWidget(recommendation_group)
        layout.addWidget(calculate_button)
        
        # Add tab
        self.tabs.addTab(parameters_tab, "Parameters")
    
    def setup_backtest_tab(self):
        """Set up the Backtest tab"""
        backtest_tab = QWidget()
        layout = QVBoxLayout(backtest_tab)
        
        # Control panel
        control_panel = QHBoxLayout()
        
        # Run backtest button
        run_backtest_button = QPushButton("Run Backtest")
        run_backtest_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        run_backtest_button.clicked.connect(self.on_run_backtest)
        
        # Include Monte Carlo checkbox
        self.monte_carlo_check = QCheckBox("Include Monte Carlo Analysis")
        self.monte_carlo_check.setChecked(True)
        
        # Implementation selection
        implementation_group = QGroupBox("Implementation")
        impl_layout = QHBoxLayout()
        
        self.implementation_combo = QComboBox()
        self.implementation_combo.addItems(["Auto", "CPU", "GPU", "CuPy"])
        impl_layout.addWidget(self.implementation_combo)
        
        implementation_group.setLayout(impl_layout)
        
        # Add to control panel
        control_panel.addWidget(run_backtest_button)
        control_panel.addWidget(self.monte_carlo_check)
        control_panel.addWidget(implementation_group)
        control_panel.addStretch(1)
        
        # Progress bar
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Progress:"))
        
        self.backtest_progress = QProgressBar()
        progress_layout.addWidget(self.backtest_progress)
        
        # Results display
        results_tabs = QTabWidget()
        
        # Summary tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        
        self.backtest_summary = QTextEdit()
        self.backtest_summary.setReadOnly(True)
        summary_layout.addWidget(self.backtest_summary)
        
        results_tabs.addTab(summary_tab, "Summary")
        
        # Trade list tab
        trades_tab = QWidget()
        trades_layout = QVBoxLayout(trades_tab)
        
        self.trades_table = QTableWidget()
        self.trades_table.setEditTriggers(QTableWidget.NoEditTriggers)
        trades_layout.addWidget(self.trades_table)
        
        results_tabs.addTab(trades_tab, "Trades")
        
        # Metrics tab
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        metrics_layout.addWidget(self.metrics_table)
        
        results_tabs.addTab(metrics_tab, "Metrics")
        
        # Add elements to main layout
        layout.addLayout(control_panel)
        layout.addLayout(progress_layout)
        layout.addWidget(results_tabs)
        
        # Add tab
        self.tabs.addTab(backtest_tab, "Backtest")
    
    def setup_optimization_tab(self):
        """Set up the Optimization tab"""
        optimization_tab = QWidget()
        layout = QVBoxLayout(optimization_tab)
        
        # Parameter ranges section
        ranges_group = QGroupBox("Parameter Ranges")
        ranges_layout = QGridLayout(ranges_group)
        
        # ATR Period
        ranges_layout.addWidget(QLabel("ATR Period:"), 0, 0)
        ranges_layout.addWidget(QLabel("Min:"), 0, 1)
        self.atr_min_input = QSpinBox()
        self.atr_min_input.setRange(1, 100)
        self.atr_min_input.setValue(DEFAULT_ATR_LENGTH_RANGE[0])
        ranges_layout.addWidget(self.atr_min_input, 0, 2)
        
        ranges_layout.addWidget(QLabel("Max:"), 0, 3)
        self.atr_max_input = QSpinBox()
        self.atr_max_input.setRange(1, 100)
        self.atr_max_input.setValue(DEFAULT_ATR_LENGTH_RANGE[1])
        ranges_layout.addWidget(self.atr_max_input, 0, 4)
        
        ranges_layout.addWidget(QLabel("Step:"), 0, 5)
        self.atr_step_input = QSpinBox()
        self.atr_step_input.setRange(1, 20)
        self.atr_step_input.setValue(DEFAULT_ATR_LENGTH_RANGE[2])
        ranges_layout.addWidget(self.atr_step_input, 0, 6)
        
        # Factor
        ranges_layout.addWidget(QLabel("Factor:"), 1, 0)
        ranges_layout.addWidget(QLabel("Min:"), 1, 1)
        self.factor_min_input = QDoubleSpinBox()
        self.factor_min_input.setRange(0.1, 50.0)
        self.factor_min_input.setSingleStep(0.1)
        self.factor_min_input.setValue(DEFAULT_FACTOR_RANGE[0])
        ranges_layout.addWidget(self.factor_min_input, 1, 2)
        
        ranges_layout.addWidget(QLabel("Max:"), 1, 3)
        self.factor_max_input = QDoubleSpinBox()
        self.factor_max_input.setRange(0.1, 50.0)
        self.factor_max_input.setSingleStep(0.1)
        self.factor_max_input.setValue(DEFAULT_FACTOR_RANGE[1])
        ranges_layout.addWidget(self.factor_max_input, 1, 4)
        
        ranges_layout.addWidget(QLabel("Step:"), 1, 5)
        self.factor_step_input = QDoubleSpinBox()
        self.factor_step_input.setRange(0.01, 10.0)
        self.factor_step_input.setSingleStep(0.01)
        self.factor_step_input.setValue(DEFAULT_FACTOR_RANGE[2])
        ranges_layout.addWidget(self.factor_step_input, 1, 6)
        
        # Buffer Multiplier
        ranges_layout.addWidget(QLabel("Buffer:"), 2, 0)
        ranges_layout.addWidget(QLabel("Min:"), 2, 1)
        self.buffer_min_input = QDoubleSpinBox()
        self.buffer_min_input.setRange(0.01, 10.0)
        self.buffer_min_input.setSingleStep(0.05)
        self.buffer_min_input.setValue(DEFAULT_BUFFER_RANGE[0])
        ranges_layout.addWidget(self.buffer_min_input, 2, 2)
        
        ranges_layout.addWidget(QLabel("Max:"), 2, 3)
        self.buffer_max_input = QDoubleSpinBox()
        self.buffer_max_input.setRange(0.01, 10.0)
        self.buffer_max_input.setSingleStep(0.05)
        self.buffer_max_input.setValue(DEFAULT_BUFFER_RANGE[1])
        ranges_layout.addWidget(self.buffer_max_input, 2, 4)
        
        ranges_layout.addWidget(QLabel("Step:"), 2, 5)
        self.buffer_step_input = QDoubleSpinBox()
        self.buffer_step_input.setRange(0.01, 10.0)
        self.buffer_step_input.setSingleStep(0.01)
        self.buffer_step_input.setValue(DEFAULT_BUFFER_RANGE[2])
        ranges_layout.addWidget(self.buffer_step_input, 2, 6)
        
        # Stop Distance
        ranges_layout.addWidget(QLabel("Stop Distance:"), 3, 0)
        ranges_layout.addWidget(QLabel("Min:"), 3, 1)
        self.stop_min_input = QSpinBox()
        self.stop_min_input.setRange(0, 500)
        self.stop_min_input.setValue(DEFAULT_STOP_RANGE[0])
        ranges_layout.addWidget(self.stop_min_input, 3, 2)
        
        ranges_layout.addWidget(QLabel("Max:"), 3, 3)
        self.stop_max_input = QSpinBox()
        self.stop_max_input.setRange(0, 500)
        self.stop_max_input.setValue(DEFAULT_STOP_RANGE[1])
        ranges_layout.addWidget(self.stop_max_input, 3, 4)
        
        ranges_layout.addWidget(QLabel("Step:"), 3, 5)
        self.stop_step_input = QSpinBox()
        self.stop_step_input.setRange(1, 500)
        self.stop_step_input.setValue(DEFAULT_STOP_RANGE[2])
        ranges_layout.addWidget(self.stop_step_input, 3, 6)
        
        # Optimization settings
        settings_group = QGroupBox("Optimization Settings")
        settings_layout = QFormLayout(settings_group)
        
        # Method
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Grid Search", "Bayesian Optimization", "Genetic Algorithm"])
        settings_layout.addRow("Method:", self.method_combo)
        
        # Metric
        self.metric_combo = QComboBox()
        self.metric_combo.addItems([
            "profit_factor", "win_rate", "total_profit_pct", "sharpe_ratio", 
            "sortino_ratio", "calmar_ratio", "avg_profit_per_trade"
        ])
        settings_layout.addRow("Optimization Metric:", self.metric_combo)
        
        # Minimum trades filter
        self.min_trades_input = QSpinBox()
        self.min_trades_input.setRange(5, 500)
        self.min_trades_input.setValue(20)
        settings_layout.addRow("Minimum Trades:", self.min_trades_input)
        
        # Parallelism
        self.parallelism_combo = QComboBox()
        self.parallelism_combo.addItems(["thread", "process", "dask", "none"])
        settings_layout.addRow("Parallelism:", self.parallelism_combo)
        
        # Time exit
        self.time_exit_input = QDoubleSpinBox()
        self.time_exit_input.setRange(0, 720)
        self.time_exit_input.setValue(48.0)
        settings_layout.addRow("Time Exit (hours):", self.time_exit_input)
        
        # Run button
        run_layout = QHBoxLayout()
        run_optimization_button = QPushButton("Run Optimization")
        run_optimization_button.setStyleSheet("background-color: #3498DB; color: white; font-weight: bold; padding: 8px;")
        run_optimization_button.clicked.connect(self.on_run_optimization)
        run_layout.addWidget(run_optimization_button)
        
        # Add progress panel
        progress_panel = self.setup_optimization_progress_panel()
        
        # Results section
        results_group = QGroupBox("Optimization Results")
        results_layout = QVBoxLayout(results_group)
        
        # Best parameters
        best_params_layout = QGridLayout()
        best_params_layout.addWidget(QLabel("Best Parameters:"), 0, 0)
        best_params_layout.addWidget(QLabel("ATR Length:"), 1, 0)
        self.opt_atr_result = QLabel("-")
        best_params_layout.addWidget(self.opt_atr_result, 1, 1)
        
        best_params_layout.addWidget(QLabel("Factor:"), 1, 2)
        self.opt_factor_result = QLabel("-")
        best_params_layout.addWidget(self.opt_factor_result, 1, 3)
        
        best_params_layout.addWidget(QLabel("Buffer:"), 2, 0)
        self.opt_buffer_result = QLabel("-")
        best_params_layout.addWidget(self.opt_buffer_result, 2, 1)
        
        best_params_layout.addWidget(QLabel("Stop:"), 2, 2)
        self.opt_stop_result = QLabel("-")
        best_params_layout.addWidget(self.opt_stop_result, 2, 3)
        
        results_layout.addLayout(best_params_layout)
        
        # Performance metrics
        metrics_layout = QGridLayout()
        metrics_layout.addWidget(QLabel("Performance:"), 0, 0)
        metrics_layout.addWidget(QLabel("Profit Factor:"), 1, 0)
        self.opt_pf_result = QLabel("-")
        metrics_layout.addWidget(self.opt_pf_result, 1, 1)
        
        metrics_layout.addWidget(QLabel("Win Rate:"), 1, 2)
        self.opt_wr_result = QLabel("-")
        metrics_layout.addWidget(self.opt_wr_result, 1, 3)
        
        metrics_layout.addWidget(QLabel("Total Profit:"), 2, 0)
        self.opt_profit_result = QLabel("-")
        metrics_layout.addWidget(self.opt_profit_result, 2, 1)
        
        metrics_layout.addWidget(QLabel("Trade Count:"), 2, 2)
        self.opt_trades_result = QLabel("-")
        metrics_layout.addWidget(self.opt_trades_result, 2, 3)
        
        results_layout.addLayout(metrics_layout)
        
        # Apply best parameters button
        apply_button = QPushButton("Apply Best Parameters")
        apply_button.clicked.connect(self.on_apply_best_parameters)
        results_layout.addWidget(apply_button)
        
        # Add elements to main layout
        top_section = QHBoxLayout()
        top_section.addWidget(ranges_group)
        top_section.addWidget(settings_group)
        
        layout.addLayout(top_section)
        layout.addLayout(run_layout)
        layout.addWidget(progress_panel)  # Add the progress panel
        layout.addWidget(results_group)
        
        # Add tab
        self.tabs.addTab(optimization_tab, "Optimization")

    def setup_optimization_progress_panel(self):
        """Set up detailed progress panel for optimization"""
        progress_panel = QGroupBox("Optimization Progress")
        layout = QVBoxLayout()
        
        # Progress bar with percentage
        progress_layout = QHBoxLayout()
        self.optimization_progress = QProgressBar()
        self.progress_label = QLabel("0%")
        progress_layout.addWidget(self.optimization_progress)
        progress_layout.addWidget(self.progress_label)
        layout.addLayout(progress_layout)
        
        # Time information
        time_layout = QGridLayout()
        time_layout.addWidget(QLabel("Started:"), 0, 0)
        time_layout.addWidget(QLabel("Elapsed:"), 1, 0)
        time_layout.addWidget(QLabel("Remaining:"), 2, 0)
        time_layout.addWidget(QLabel("ETA:"), 3, 0)
        
        self.start_time_label = QLabel("")
        self.elapsed_time_label = QLabel("")
        self.remaining_time_label = QLabel("")
        self.eta_label = QLabel("")
        
        time_layout.addWidget(self.start_time_label, 0, 1)
        time_layout.addWidget(self.elapsed_time_label, 1, 1)
        time_layout.addWidget(self.remaining_time_label, 2, 1)
        time_layout.addWidget(self.eta_label, 3, 1)
        
        layout.addLayout(time_layout)
        
        # Combination counters
        counts_layout = QGridLayout()
        counts_layout.addWidget(QLabel("Total Combinations:"), 0, 0)
        counts_layout.addWidget(QLabel("Completed:"), 1, 0)
        counts_layout.addWidget(QLabel("Remaining:"), 2, 0)
        counts_layout.addWidget(QLabel("Valid Results:"), 3, 0)
        
        self.total_combinations_label = QLabel("0")
        self.completed_combinations_label = QLabel("0")
        self.remaining_combinations_label = QLabel("0")
        self.valid_results_label = QLabel("0")
        
        counts_layout.addWidget(self.total_combinations_label, 0, 1)
        counts_layout.addWidget(self.completed_combinations_label, 1, 1)
        counts_layout.addWidget(self.remaining_combinations_label, 2, 1)
        counts_layout.addWidget(self.valid_results_label, 3, 1)
        
        layout.addLayout(counts_layout)
        
        # Top 5 combinations table
        layout.addWidget(QLabel("<b>Top 5 Combinations (Live):</b>"))
        
        self.top_combinations_table = QTableWidget()
        self.top_combinations_table.setColumnCount(6)
        self.top_combinations_table.setRowCount(5)
        self.top_combinations_table.setHorizontalHeaderLabels(
            ["ATR", "Factor", "Buffer", "Stop", "Win Rate", "Profit Factor"]
        )
        self.top_combinations_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.top_combinations_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        layout.addWidget(self.top_combinations_table)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.stop_optimization_button = QPushButton("Stop Optimization")
        self.stop_optimization_button.setStyleSheet("background-color: #E74C3C; color: white;")
        self.stop_optimization_button.clicked.connect(self.stop_optimization)
        self.stop_optimization_button.setEnabled(False)
        
        button_layout.addWidget(self.stop_optimization_button)
        layout.addLayout(button_layout)
        
        progress_panel.setLayout(layout)
        return progress_panel
    
    def setup_results_tab(self):
        """Set up the Results tab"""
        results_tab = QWidget()
        layout = QVBoxLayout(results_tab)
        
        # Top section with file management
        top_layout = QHBoxLayout()
        
        # Backtest results section
        backtest_group = QGroupBox("Backtest Results")
        backtest_layout = QVBoxLayout()
        
        self.backtest_file_combo = QComboBox()
        self.backtest_file_combo.currentIndexChanged.connect(self.on_backtest_file_selected)
        
        refresh_backtest_button = QPushButton("Refresh")
        refresh_backtest_button.clicked.connect(self.refresh_backtest_files)
        
        backtest_file_layout = QHBoxLayout()
        backtest_file_layout.addWidget(QLabel("Select File:"))
        backtest_file_layout.addWidget(self.backtest_file_combo)
        backtest_file_layout.addWidget(refresh_backtest_button)
        
        backtest_layout.addLayout(backtest_file_layout)
        
        load_backtest_button = QPushButton("Load Selected Backtest")
        load_backtest_button.clicked.connect(self.on_load_backtest)
        backtest_layout.addWidget(load_backtest_button)
        
        backtest_group.setLayout(backtest_layout)
        
        # Optimization results section
        optimization_group = QGroupBox("Optimization Results")
        optimization_layout = QVBoxLayout()
        
        self.optimization_file_combo = QComboBox()
        self.optimization_file_combo.currentIndexChanged.connect(self.on_optimization_file_selected)
        
        refresh_optimization_button = QPushButton("Refresh")
        refresh_optimization_button.clicked.connect(self.refresh_optimization_files)
        
        optimization_file_layout = QHBoxLayout()
        optimization_file_layout.addWidget(QLabel("Select File:"))
        optimization_file_layout.addWidget(self.optimization_file_combo)
        optimization_file_layout.addWidget(refresh_optimization_button)
        
        optimization_layout.addLayout(optimization_file_layout)
        
        load_optimization_button = QPushButton("Load Selected Optimization")
        load_optimization_button.clicked.connect(self.on_load_optimization)
        optimization_layout.addWidget(load_optimization_button)
        
        optimization_group.setLayout(optimization_layout)
        
        top_layout.addWidget(backtest_group)
        top_layout.addWidget(optimization_group)
        
        # Report generation section
        report_group = QGroupBox("Generate Reports")
        report_layout = QVBoxLayout()
        
        # Report type selection
        report_type_layout = QHBoxLayout()
        report_type_layout.addWidget(QLabel("Report Type:"))
        
        self.report_type_combo = QComboBox()
        self.report_type_combo.addItems(["Backtest Report", "Optimization Report"])
        report_type_layout.addWidget(self.report_type_combo)
        
        # Format selection
        report_type_layout.addWidget(QLabel("Format:"))
        
        self.report_format_combo = QComboBox()
        self.report_format_combo.addItems(["HTML", "Markdown", "Text"])
        report_type_layout.addWidget(self.report_format_combo)
        
        report_layout.addLayout(report_type_layout)
        
        # Generate button
        generate_button = QPushButton("Generate Report")
        generate_button.clicked.connect(self.on_generate_report)
        report_layout.addWidget(generate_button)
        
        report_group.setLayout(report_layout)
        
        # Results preview
        preview_group = QGroupBox("Results Preview")
        preview_layout = QVBoxLayout()
        
        self.results_preview = QTextEdit()
        self.results_preview.setReadOnly(True)
        
        preview_layout.addWidget(self.results_preview)
        
        preview_group.setLayout(preview_layout)
        
        # Add elements to main layout
        layout.addLayout(top_layout)
        layout.addWidget(report_group)
        layout.addWidget(preview_group)
        
        # Add tab
        self.tabs.addTab(results_tab, "Results")
    
    def setup_chart_tab(self):
        """Set up the Chart tab"""
        chart_tab = QWidget()
        layout = QVBoxLayout(chart_tab)
        
        # Chart controls
        controls_layout = QHBoxLayout()
        
        # Chart type selection
        controls_layout.addWidget(QLabel("Chart Type:"))
        
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "Price with SuperTrend", "Equity Curve", "Drawdown", 
            "Trade Performance", "Parameter Analysis"
        ])
        self.chart_type_combo.currentIndexChanged.connect(self.on_chart_type_changed)
        
        controls_layout.addWidget(self.chart_type_combo)
        
        # Time period selection
        controls_layout.addWidget(QLabel("Time Period:"))
        
        self.time_period_combo = QComboBox()
        self.time_period_combo.addItems([
            "All Data", "Last Month", "Last Week", "Last Day"
        ])
        self.time_period_combo.currentIndexChanged.connect(self.on_time_period_changed)
        
        controls_layout.addWidget(self.time_period_combo)
        
        # Update button
        update_chart_button = QPushButton("Update Chart")
        update_chart_button.clicked.connect(self.update_chart)
        controls_layout.addWidget(update_chart_button)
        
        # Export button
        export_chart_button = QPushButton("Export Chart")
        export_chart_button.clicked.connect(self.on_export_chart)
        controls_layout.addWidget(export_chart_button)
        
        # Chart display area - using matplotlib
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Add everything to layout
        layout.addLayout(controls_layout)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Add tab
        self.tabs.addTab(chart_tab, "Charts")
    
    def on_tab_changed(self, index):
        """Handle tab change events"""
        tab_name = self.tabs.tabText(index)
        self._log('debug', f"Switched to {tab_name} tab")
        
        # Update UI based on current tab
        if tab_name == "Parameters" and self.df is not None:
            # Update parameter recommendations
            self.update_parameter_recommendations()
        elif tab_name == "Charts" and (self.df_st is not None or self.backtest_results is not None):
            # Update chart
            self.update_chart()
        elif tab_name == "Results":
            # Refresh file lists
            self.refresh_backtest_files()
            self.refresh_optimization_files()
    
    def on_browse_file(self):
        """Open file dialog to select data file"""
        file_filter = "All supported files (*.csv *.xlsx *.xls *.parquet *.feather);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;Parquet Files (*.parquet);;Feather Files (*.feather)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", file_filter)
        
        if file_path:
            self.file_path_input.setText(file_path)
            self.load_data_file(file_path)
    
    def load_data_file(self, file_path: str):
        """Load data from file"""
        try:
            self.status_bar.showMessage("Loading data file...")
            self._log('info', f"Loading data from {file_path}")
            
            # Create progress dialog
            progress = QProgressDialog("Loading data file...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setValue(10)
            QApplication.processEvents()
            
            # Load data
            self.df = self.data_loader.load_file(file_path)
            
            progress.setValue(50)
            QApplication.processEvents()
            
            if self.df is not None and not self.df.empty:
                # Update data preview
                self.update_data_preview()
                
                # Calculate data statistics
                progress.setValue(70)
                progress.setLabelText("Analyzing data...")
                QApplication.processEvents()
                
                analysis = self.data_analyzer.analyze_ohlc_data(self.df)
                
                progress.setValue(90)
                QApplication.processEvents()
                
                # Update data stats display
                self.update_data_stats(analysis)
                
                # Update parameter recommendations
                self.update_parameter_recommendations()
                
                self.status_bar.showMessage(f"Successfully loaded data with {len(self.df)} rows")
                self._log('info', f"Successfully loaded data with {len(self.df)} rows")
            else:
                self.status_bar.showMessage("Error: Empty dataset")
                self._log('error', "Loaded data is empty")
            
            progress.setValue(100)
            
        except Exception as e:
            self.status_bar.showMessage(f"Error loading data: {str(e)}")
            self._log('error', f"Error loading data: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error loading data file:\n{str(e)}")
    
    def update_data_preview(self):
        """Update the data preview table"""
        if self.df is None or self.df.empty:
            return
        
        # Limit to first 100 rows for preview
        preview_df = self.df.head(100).reset_index()
        
        # Set up table
        self.data_preview_table.setRowCount(len(preview_df))
        self.data_preview_table.setColumnCount(len(preview_df.columns))
        
        # Set headers
        self.data_preview_table.setHorizontalHeaderLabels(preview_df.columns)
        
        # Fill data
        for row in range(len(preview_df)):
            for col in range(len(preview_df.columns)):
                val = str(preview_df.iloc[row, col])
                item = QTableWidgetItem(val)
                self.data_preview_table.setItem(row, col, item)
        
        # Resize columns to content
        self.data_preview_table.resizeColumnsToContents()
    
    def update_data_stats(self, analysis: Dict[str, Any]):
        """Update data statistics display"""
        if not analysis:
            return
        
        # Format analysis as readable text
        stats_text = f"""
DATA ANALYSIS SUMMARY
====================

Data Range: {analysis.get('date_range', {}).get('start', 'N/A')} to {analysis.get('date_range', {}).get('end', 'N/A')}
Total Calendar Days: {analysis.get('total_calendar_days', 'N/A')}
Trading Days: {analysis.get('trading_days', 'N/A')}
Candle Count: {analysis.get('candle_count', 'N/A')}

Timeframe: {analysis.get('timeframe', 'N/A')}
Timeframe Confidence: {analysis.get('timeframe_confidence', 0):.2f}%

Price Statistics:
----------------
Min Price: {analysis.get('price_statistics', {}).get('min', 'N/A'):.2f}
Max Price: {analysis.get('price_statistics', {}).get('max', 'N/A'):.2f}
Open First: {analysis.get('price_statistics', {}).get('open_first', 'N/A'):.2f}
Close Last: {analysis.get('price_statistics', {}).get('close_last', 'N/A'):.2f}
Price Change: {analysis.get('price_statistics', {}).get('price_change', 'N/A'):.2f} ({analysis.get('price_statistics', {}).get('price_change_pct', 'N/A'):.2f}%)

Volatility Metrics:
-----------------
ATR (14): {analysis.get('volatility', {}).get('avg_true_range', 'N/A'):.2f}
Avg Daily Range %: {analysis.get('volatility', {}).get('avg_daily_range_pct', 'N/A'):.2f}%
Std Dev of Returns: {analysis.get('volatility', {}).get('std_dev_daily_returns', 'N/A'):.2f}%

Data Quality:
-----------
Missing Values: {analysis.get('data_quality', {}).get('missing_values', 'N/A')}
Has Gaps: {analysis.get('data_quality', {}).get('has_gaps', 'N/A')}
"""

        # Add volume information if available
        if 'volume' in analysis:
            volume_stats = analysis['volume']
            stats_text += f"""
Volume Statistics:
----------------
Total Volume: {volume_stats.get('total', 'N/A'):,}
Daily Average: {volume_stats.get('daily_avg', 'N/A'):,.2f}
Max Volume: {volume_stats.get('max', 'N/A'):,}
Min Volume: {volume_stats.get('min', 'N/A'):,}
"""
        
        # Add trend information
        if 'trend' in analysis:
            trend = analysis['trend']
            stats_text += f"""
Trend Analysis:
-------------
Primary Trend: {trend.get('primary_trend', 'N/A')}
Trend Strength: {trend.get('trend_strength', 'N/A')}
ADX Value: {trend.get('adx', 'N/A'):.2f}
Volatility %: {trend.get('volatility_pct', 'N/A'):.2f}%
"""
        
        self.data_stats_text.setText(stats_text)
    
    def update_parameter_recommendations(self):
        """Update parameter recommendations based on data analysis"""
        if self.df is None or self.df.empty:
            return
        
        try:
            # Analyze data to get parameter recommendations
            analysis = self.data_analyzer.analyze_ohlc_data(self.df)
            
            if 'parameter_recommendations' in analysis:
                rec = analysis['parameter_recommendations']
                
                # Format recommendations as readable text
                rec_text = f"""
PARAMETER RECOMMENDATIONS
========================

Based on the data analysis, here are the recommended parameters for this dataset:

ATR Length: {rec.get('atr_length', {}).get('optimal', 14)} (range: {rec.get('atr_length', {}).get('min', 10)}-{rec.get('atr_length', {}).get('max', 20)})
Factor: {rec.get('factor', {}).get('optimal', 3.0):.1f} (range: {rec.get('factor', {}).get('min', 2.0):.1f}-{rec.get('factor', {}).get('max', 4.0):.1f})
Buffer Multiplier: {rec.get('buffer_multiplier', {}).get('optimal', 0.3):.2f} (range: {rec.get('buffer_multiplier', {}).get('min', 0.2):.2f}-{rec.get('buffer_multiplier', {}).get('max', 0.4):.2f})
Hard Stop Distance: {rec.get('hard_stop_distance', {}).get('optimal', 50)} (range: {rec.get('hard_stop_distance', {}).get('min', 30)}-{rec.get('hard_stop_distance', {}).get('max', 70)})

Recommendation Notes:
-------------------
- Timeframe: {rec.get('notes', {}).get('timeframe', 'N/A')}
- Is Intraday: {rec.get('notes', {}).get('is_intraday', 'N/A')}
- Volatility %: {rec.get('notes', {}).get('volatility_pct', 'N/A'):.2f}%
"""
                
                self.recommendation_text.setText(rec_text)
                
                # Update parameter input fields with recommended values
                self.atr_length_input.setValue(rec.get('atr_length', {}).get('optimal', 14))
                self.factor_input.setValue(rec.get('factor', {}).get('optimal', 3.0))
                self.buffer_input.setValue(rec.get('buffer_multiplier', {}).get('optimal', 0.3))
                self.stop_input.setValue(rec.get('hard_stop_distance', {}).get('optimal', 50))
            else:
                self.recommendation_text.setText("No parameter recommendations available.")
        
        except Exception as e:
            self._log('error', f"Error updating parameter recommendations: {str(e)}")
            self.recommendation_text.setText(f"Error generating parameter recommendations: {str(e)}")
            
    def on_calculate_supertrend(self):
        """Calculate SuperTrend based on current parameters"""
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "Warning", "Please load data first.")
            return
        
        try:
            # Get parameters from UI
            atr_length = self.atr_length_input.value()
            factor = self.factor_input.value()
            buffer = self.buffer_input.value()
            
            # Create progress dialog
            progress = QProgressDialog("Calculating SuperTrend...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setValue(10)
            QApplication.processEvents()
            
            # Get implementation selection
            implementation = self.implementation_combo.currentText().lower()
            if implementation == 'auto':
                implementation = None
            
            # Calculate SuperTrend
            self.status_bar.showMessage("Calculating SuperTrend...")
            
            supertrend = SuperTrend(self.log)
            self.df_st = supertrend.calculate(
                self.df, 
                atr_length, 
                factor, 
                buffer, 
                force_implementation=implementation
            )
            
            progress.setValue(90)
            QApplication.processEvents()
            
            self.status_bar.showMessage(f"SuperTrend calculated with ATR={atr_length}, Factor={factor}, Buffer={buffer}")
            self._log('info', f"SuperTrend calculated with ATR={atr_length}, Factor={factor}, Buffer={buffer}")
            
            # Switch to Charts tab and update chart
            self.tabs.setCurrentIndex(self.tabs.indexOf(self.tabs.findChild(QWidget, "Charts")))
            self.update_chart()
            
            progress.setValue(100)
            
            # Show a message
            QMessageBox.information(self, "Success", "SuperTrend calculation complete. Switching to Charts tab.")
            
        except Exception as e:
            self.status_bar.showMessage(f"Error calculating SuperTrend: {str(e)}")
            self._log('error', f"Error calculating SuperTrend: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error calculating SuperTrend:\n{str(e)}")
    
    def on_run_backtest(self):
        """Run backtest with current parameters"""
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "Warning", "Please load data first.")
            return
        
        try:
            # Get parameters from UI
            atr_length = self.atr_length_input.value()
            factor = self.factor_input.value()
            buffer = self.buffer_input.value()
            stop = self.stop_input.value()
            time_exit = self.time_exit_input.value()
            include_monte_carlo = self.monte_carlo_check.isChecked()
            
            # Create progress dialog
            progress = QProgressDialog("Running backtest...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setValue(10)
            QApplication.processEvents()
            
            # Get implementation selection
            implementation = self.implementation_combo.currentText().lower()
            if implementation == 'auto':
                implementation = None
            
            # Run backtest
            self.status_bar.showMessage("Running backtest...")
            self._log('info', f"Running backtest with ATR={atr_length}, Factor={factor}, Buffer={buffer}, Stop={stop}")
            
            # Calculate SuperTrend if not already calculated
            progress.setLabelText("Calculating SuperTrend...")
            progress.setValue(20)
            QApplication.processEvents()
            
            supertrend = SuperTrend(self.log)
            self.df_st = supertrend.calculate(
                self.df, 
                atr_length, 
                factor, 
                buffer, 
                force_implementation=implementation
            )
            
            # Run backtest
            progress.setLabelText("Processing signals and trades...")
            progress.setValue(40)
            QApplication.processEvents()
            
            backtester = Backtester(self.log)
            self.backtest_results = backtester.run_backtest(
                self.df,
                atr_length,
                factor,
                buffer,
                stop,
                time_exit
            )
            
            # Run Monte Carlo if requested
            if include_monte_carlo:
                progress.setLabelText("Running Monte Carlo simulation...")
                progress.setValue(60)
                QApplication.processEvents()
                
                monte_carlo = MonteCarloSimulation(self.log)
                monte_carlo_results = monte_carlo.run_simulation(
                    self.backtest_results['trades'],
                    num_simulations=500,
                    initial_capital=self.capital_input.value()
                )
                
                # Add Monte Carlo results
                self.backtest_results['monte_carlo'] = monte_carlo_results
            
            progress.setLabelText("Updating results display...")
            progress.setValue(80)
            QApplication.processEvents()
            
            # Update UI with results
            self.update_backtest_results()
            
            progress.setValue(100)
            
            self.status_bar.showMessage(f"Backtest completed with {len(self.backtest_results['trades'])} trades")
            self._log('info', f"Backtest completed with {len(self.backtest_results['trades'])} trades")
            
        except Exception as e:
            self.status_bar.showMessage(f"Error running backtest: {str(e)}")
            self._log('error', f"Error running backtest: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error running backtest:\n{str(e)}")
    
    def update_backtest_results(self):
        """Update UI with backtest results"""
        if not self.backtest_results:
            return
        
        # Update backtest summary
        performance = self.backtest_results.get('performance', {})
        trades = self.backtest_results.get('trades', [])
        params = self.backtest_results.get('parameters', {})
        
        # Format summary text
        summary_text = f"""
BACKTEST SUMMARY
===============

Parameters:
-----------
ATR Length: {params.get('atr_length', 'N/A')}
Factor: {params.get('factor', 'N/A')}
Buffer Multiplier: {params.get('buffer_multiplier', 'N/A')}
Hard Stop Distance: {params.get('hard_stop_distance', 'N/A')}
Time Exit Hours: {params.get('time_exit_hours', 'N/A')}

Performance Summary:
------------------
Total Trades: {performance.get('trade_count', 0)}
Winning Trades: {performance.get('winning_trades', 0)} ({performance.get('win_rate', 0)*100:.2f}%)
Losing Trades: {performance.get('losing_trades', 0)} ({(1-performance.get('win_rate', 0))*100:.2f}%)
Profit Factor: {performance.get('profit_factor', 0):.2f}
Total Return: {performance.get('total_profit_pct', 0):.2f}%
Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}
Max Drawdown: {performance.get('max_drawdown', 0):.2f}%
Risk-Adjusted Return: {performance.get('risk_adjusted_return', 0):.2f}

Performance Metrics:
------------------
Average Trade: {performance.get('average_profit_pct', 0):.2f}%
Average Duration: {performance.get('avg_trade_duration', 0):.2f} hours
Max Consecutive Wins: {performance.get('max_consecutive_wins', 0)}
Max Consecutive Losses: {performance.get('max_consecutive_losses', 0)}
Long Win Rate: {performance.get('long_win_rate', 0)*100:.2f}%
Short Win Rate: {performance.get('short_win_rate', 0)*100:.2f}%
"""

        # Add Monte Carlo results if available
        monte_carlo = self.backtest_results.get('monte_carlo', {})
        if monte_carlo:
            summary_text += f"""
Monte Carlo Analysis ({len(monte_carlo.get('simulation_data', []))} simulations):
-------------------------------------------------
Mean Final Equity: {monte_carlo.get('equity', {}).get('mean', 0):.2f}
5% Worst Case: {monte_carlo.get('equity', {}).get('percentiles', {}).get('5', 0):.2f}
Mean Max Drawdown: {monte_carlo.get('drawdown', {}).get('mean', 0):.2f}%
Probability of Profit: {monte_carlo.get('probability', {}).get('profit', 0)*100:.2f}%
"""
        
        self.backtest_summary.setText(summary_text)
        
        # Update trades table
        if trades:
            # Convert trade objects to dictionaries if necessary
            if not isinstance(trades[0], dict):
                trades_list = [t.to_dict() for t in trades]
            else:
                trades_list = trades
                
            # Set up table
            self.trades_table.setRowCount(len(trades_list))
            headers = [
                "ID", "Type", "Entry Time", "Entry Price", "Exit Time", 
                "Exit Price", "Profit %", "Duration (hrs)"
            ]
            self.trades_table.setColumnCount(len(headers))
            self.trades_table.setHorizontalHeaderLabels(headers)
            
            # Fill data
            for row, trade in enumerate(trades_list):
                self.trades_table.setItem(row, 0, QTableWidgetItem(str(trade.get('trade_id', 'N/A'))))
                self.trades_table.setItem(row, 1, QTableWidgetItem(trade.get('position_type', 'N/A').capitalize()))
                self.trades_table.setItem(row, 2, QTableWidgetItem(str(trade.get('entry_time', 'N/A'))))
                self.trades_table.setItem(row, 3, QTableWidgetItem(f"{trade.get('entry_price', 0):.2f}"))
                self.trades_table.setItem(row, 4, QTableWidgetItem(str(trade.get('exit_time', 'N/A'))))
                self.trades_table.setItem(row, 5, QTableWidgetItem(f"{trade.get('exit_price', 0):.2f}"))
                
                # Format profit/loss with color
                profit_pct = trade.get('profit_pct', 0)
                profit_item = QTableWidgetItem(f"{profit_pct:.2f}%")
                profit_item.setForeground(QColor('green') if profit_pct > 0 else QColor('red'))
                self.trades_table.setItem(row, 6, profit_item)
                
                self.trades_table.setItem(row, 7, QTableWidgetItem(f"{trade.get('duration_hours', 0):.2f}"))
            
            # Resize columns
            self.trades_table.resizeColumnsToContents()
        
        # Update metrics table
        metrics = [
            ("Trade Count", performance.get('trade_count', 0)),
            ("Win Rate", f"{performance.get('win_rate', 0)*100:.2f}%"),
            ("Profit Factor", f"{performance.get('profit_factor', 0):.2f}"),
            ("Total Return", f"{performance.get('total_profit_pct', 0):.2f}%"),
            ("Sharpe Ratio", f"{performance.get('sharpe_ratio', 0):.2f}"),
            ("Sortino Ratio", f"{performance.get('sortino_ratio', 0):.2f}"),
            ("Max Drawdown", f"{performance.get('max_drawdown', 0):.2f}%"),
            ("Risk-Adjusted Return", f"{performance.get('risk_adjusted_return', 0):.2f}"),
            ("Average Trade", f"{performance.get('average_profit_pct', 0):.2f}%"),
            ("Average Duration", f"{performance.get('avg_trade_duration', 0):.2f} hours"),
            ("Max Consecutive Wins", performance.get('max_consecutive_wins', 0)),
            ("Max Consecutive Losses", performance.get('max_consecutive_losses', 0)),
            ("Long Win Rate", f"{performance.get('long_win_rate', 0)*100:.2f}%"),
            ("Short Win Rate", f"{performance.get('short_win_rate', 0)*100:.2f}%")
        ]
        
        self.metrics_table.setRowCount(len(metrics))
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        
        for row, (name, value) in enumerate(metrics):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(name))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(str(value)))
        
        self.metrics_table.resizeColumnsToContents()
  
    def on_run_optimization(self):
        """Run optimization with current parameters in a background thread"""
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "Warning", "Please load data first.")
            return
        
        if "No CUDA-compatible GPU detected" in self.log.get_log_entries():
            force_implementation = "cpu"
            self._log('info', "Forcing CPU implementation as no GPU detected")
        
        
        try:
            # Get parameter ranges from UI
            param_ranges = {
                'atr_lengths': list(range(
                    self.atr_min_input.value(), 
                    self.atr_max_input.value() + 1, 
                    self.atr_step_input.value()
                )),
                'factors': list(np.arange(
                    self.factor_min_input.value(), 
                    self.factor_max_input.value() + 0.01, 
                    self.factor_step_input.value()
                )),
                'buffers': list(np.arange(
                    self.buffer_min_input.value(), 
                    self.buffer_max_input.value() + 0.01, 
                    self.buffer_step_input.value()
                )),
                'stops': list(range(
                    self.stop_min_input.value(), 
                    self.stop_max_input.value() + 1, 
                    self.stop_step_input.value()
                ))
            }
            
            # Get optimization settings
            optimization_metric = self.metric_combo.currentText()
            min_trades = self.min_trades_input.value()
            method = self.method_combo.currentText()
            parallelism = self.parallelism_combo.currentText()
            
            # Check total combinations
            total_combinations = (
                len(param_ranges['atr_lengths']) * 
                len(param_ranges['factors']) * 
                len(param_ranges['buffers']) * 
                len(param_ranges['stops'])
            )
            
            if total_combinations > 10000:
                confirm = QMessageBox.question(
                    self,
                    "Confirm Optimization",
                    f"You're about to run optimization with {total_combinations:,} parameter combinations. "
                    "This might take a long time. Continue?",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if confirm != QMessageBox.Yes:
                    return
            
            # Reset progress display
            self.optimization_progress.setValue(0)
            self.progress_label.setText("0%")
            self.start_time_label.setText(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
            self.elapsed_time_label.setText("00:00:00")
            self.remaining_time_label.setText("--:--:--")
            self.eta_label.setText("--")
            self.total_combinations_label.setText(str(total_combinations))
            self.completed_combinations_label.setText("0")
            self.remaining_combinations_label.setText(str(total_combinations))
            self.valid_results_label.setText("0")
            
            # Clear top combinations table
            for row in range(5):
                for col in range(6):
                    self.top_combinations_table.setItem(row, col, QTableWidgetItem(""))
            
            # Create and start worker thread
            self.optimization_worker = OptimizationWorker(
                method, self.df, param_ranges, optimization_metric, 
                min_trades, self.time_exit_input.value(), parallelism
            )
            
            # Clear any existing signal connections to avoid duplicate signals
            try:  #add this line - add safe disconnection
                self.optimization_worker.progress_signal.disconnect()
                self.optimization_worker.result_signal.disconnect()
            except TypeError:
                # No connections to disconnect
                pass
            
            
            
            # Connect signals
            self.optimization_worker.progress_signal.connect(self.update_optimization_progress)
            self.optimization_worker.result_signal.connect(self.optimization_completed)
            
            # Update UI state
            self.stop_optimization_button.setEnabled(True)
            self.status_bar.showMessage(f"Running {method} optimization...")
            self._log('info', f"Running {method} optimization with {total_combinations} combinations")
            
            # Start worker
            self.optimization_worker.start()
            
        except Exception as e:
            self.status_bar.showMessage(f"Error setting up optimization: {str(e)}")
            self._log('error', f"Error setting up optimization: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error setting up optimization:\n{str(e)}")

    def update_optimization_progress(self, data):
        """Update UI with optimization progress"""
        
        self._log('info', f"Progress: {data.get('completed', 0)}/{data.get('total', 0)} combinations ({data.get('progress_pct', 0):.1f}%)")
        
        # Print to console to confirm callback is being called
        print(f"UI received progress update: {data.get('completed', 0)}/{data.get('total', 0)}")
    
        
        # Check for error
        if 'error' in data:
            self.status_bar.showMessage(f"Error in optimization: {data['error']}")
            self.stop_optimization()
            QMessageBox.critical(self, "Error", f"Error in optimization:\n{data['error']}")
            return
        
        # Update progress bar
        progress_pct = data.get('progress_pct', 0)
        self.optimization_progress.setValue(int(progress_pct))
        self.progress_label.setText(f"{progress_pct:.1f}%")
        
        # Update time labels
        elapsed = data.get('elapsed', 0)
        self.elapsed_time_label.setText(format_time_delta(elapsed))
        
        remaining = data.get('remaining', 0)
        self.remaining_time_label.setText(format_time_delta(remaining))
        
        # Calculate and display ETA
        if remaining > 0:
            eta = datetime.utcnow() + timedelta(seconds=remaining)
            self.eta_label.setText(eta.strftime("%H:%M:%S"))
        
        # Update combination counters
        completed = data.get('completed', 0)
        total = data.get('total', 0)
        
        self.total_combinations_label.setText(str(total))
        self.completed_combinations_label.setText(str(completed))
        self.remaining_combinations_label.setText(str(total - completed))
        self.valid_results_label.setText(str(data.get('valid_results', 0)))
        
        
        # Update top combinations table
        try:
            top_combinations = data.get('top_combinations', [])
            for row, combo in enumerate(top_combinations[:5]):  # Ensure we only process 5 rows max
                if row >= 5:  # Safety check
                    break
                
                params = combo.get('parameters', {})
                perf = combo.get('performance', {})
            
                # Safety check that the table has enough rows
                if self.top_combinations_table.rowCount() <= row:
                    continue
                
                # Update table cells safely with type checking
                if 'atr_length' in params:
                    self.top_combinations_table.setItem(row, 0, QTableWidgetItem(str(params['atr_length'])))
            
                if 'factor' in params:
                    self.top_combinations_table.setItem(row, 1, QTableWidgetItem(f"{params['factor']:.2f}"))
            
                if 'buffer_multiplier' in params:
                    self.top_combinations_table.setItem(row, 2, QTableWidgetItem(f"{params['buffer_multiplier']:.2f}"))
            
                if 'hard_stop_distance' in params:
                    self.top_combinations_table.setItem(row, 3, QTableWidgetItem(str(params['hard_stop_distance'])))
            
                if 'win_rate' in perf:
                    win_rate = perf['win_rate'] * 100
                    self.top_combinations_table.setItem(row, 4, QTableWidgetItem(f"{win_rate:.2f}%"))
            
                if 'profit_factor' in perf:
                    profit_factor = perf['profit_factor']
                    self.top_combinations_table.setItem(row, 5, QTableWidgetItem(f"{profit_factor:.2f}"))
        except Exception as e:
            print(f"Error updating combinations table: {str(e)}")
    
        # Force UI refresh immediately
        QApplication.processEvents()

    def optimization_completed(self, results):
        """Handle optimization completion"""
        self.optimization_results = results
        self.update_optimization_results()
        self.stop_optimization_button.setEnabled(False)
        self.status_bar.showMessage("Optimization completed successfully")
        
        # Show completion message
        QMessageBox.information(self, "Optimization Complete", 
                              "Optimization has completed successfully!")
        
        # Apply best parameters if available
        best_params = results.get('best_parameters', {})
        if best_params:
            self.atr_length_input.setValue(best_params.get('atr_length', 14))
            self.factor_input.setValue(best_params.get('factor', 3.0))
            self.buffer_input.setValue(best_params.get('buffer_multiplier', 0.3))
            self.stop_input.setValue(best_params.get('hard_stop_distance', 50))

    def stop_optimization(self):
        """Stop the running optimization"""
        if hasattr(self, 'optimization_worker') and self.optimization_worker and self.optimization_worker.isRunning():
            self.optimization_worker.stop()
            self.optimization_worker.wait()
            self.stop_optimization_button.setEnabled(False)
            self.status_bar.showMessage("Optimization stopped by user")

    def update_optimization_results(self):
        """Update optimization results display"""
        if not self.optimization_results:
            return
            
        # Get best parameters
        best_params = self.optimization_results.get('best_parameters', {})
        best_performance = self.optimization_results.get('best_performance', {})
        
        if not best_params or not best_performance:
            return
            
        # Update parameter display
        self.opt_atr_result.setText(str(best_params.get('atr_length', '-')))
        self.opt_factor_result.setText(f"{best_params.get('factor', '-'):.2f}")
        self.opt_buffer_result.setText(f"{best_params.get('buffer_multiplier', '-'):.2f}")
        self.opt_stop_result.setText(str(best_params.get('hard_stop_distance', '-')))
        
        # Update performance metrics
        profit_factor = best_performance.get('profit_factor', 0)
        win_rate = best_performance.get('win_rate', 0) * 100
        total_profit = best_performance.get('total_profit_pct', 0)
        trade_count = best_performance.get('trade_count', 0)
        
        self.opt_pf_result.setText(f"{profit_factor:.2f}")
        self.opt_wr_result.setText(f"{win_rate:.2f}%")
        self.opt_profit_result.setText(f"{total_profit:.2f}%")
        self.opt_trades_result.setText(str(trade_count))
        
        # Update chart tab if needed
        self.update_chart()
        
        # Force UI update explicitly
        QApplication.processEvents()

    def on_apply_best_parameters(self):
        """Apply the best parameters from optimization to the parameters tab"""
        if not self.optimization_results or not self.optimization_results.get('best_parameters'):
            QMessageBox.warning(self, "Warning", "No optimization results available.")
            return
            
        best_params = self.optimization_results.get('best_parameters', {})
        
        # Switch to parameters tab
        self.tabs.setCurrentIndex(1)  # Assuming parameters tab is index 1
        
        # Apply parameters
        self.atr_length_input.setValue(best_params.get('atr_length', 14))
        self.factor_input.setValue(best_params.get('factor', 3.0))
        self.buffer_input.setValue(best_params.get('buffer_multiplier', 0.3))
        self.stop_input.setValue(best_params.get('hard_stop_distance', 50))
        
        # Show confirmation
        self.status_bar.showMessage("Best parameters applied")
 
    def refresh_backtest_files(self):
        """Refresh the list of available backtest files"""
        try:
            backtest_files = self.results_storage.list_backtest_results()
            
            # Clear and update combo box
            self.backtest_file_combo.clear()
            
            if backtest_files:
                for result in backtest_files:
                    self.backtest_file_combo.addItem(
                        f"{result['name']} ({result['date']})",
                        result['filepath']
                    )
            else:
                self.backtest_file_combo.addItem("No backtest results found")
                
        except Exception as e:
            self._log('error', f"Error refreshing backtest files: {str(e)}")
    
    def refresh_optimization_files(self):
        """Refresh the list of available optimization files"""
        try:
            optimization_files = self.results_storage.list_optimization_results()
            
            # Clear and update combo box
            self.optimization_file_combo.clear()
            
            if optimization_files:
                for result in optimization_files:
                    self.optimization_file_combo.addItem(
                        f"{result['name']} ({result['date']})",
                        result['filepath']
                    )
            else:
                self.optimization_file_combo.addItem("No optimization results found")
                
        except Exception as e:
            self._log('error', f"Error refreshing optimization files: {str(e)}")
    
    def on_backtest_file_selected(self, index):
        """Handle selection of a backtest file"""
        if index < 0 or self.backtest_file_combo.currentText() == "No backtest results found":
            return
            
        filepath = self.backtest_file_combo.currentData()
        if filepath:
            # Show brief preview info
            try:
                result = self.results_storage.load_backtest_results(filepath)
                if result:
                    params = result.get('parameters', {})
                    perf = result.get('performance', {})
                    
                    # Format preview text
                    preview = f"""
Backtest Results Preview - {os.path.basename(filepath)}
-----------------------------------------
Parameters: ATR={params.get('atr_length', 'N/A')}, Factor={params.get('factor', 'N/A')}, Buffer={params.get('buffer_multiplier', 'N/A')}, Stop={params.get('hard_stop_distance', 'N/A')}
Total Trades: {perf.get('trade_count', 0)}
Win Rate: {perf.get('win_rate', 0)*100:.2f}%
Profit Factor: {perf.get('profit_factor', 0):.2f}
Total Profit: {perf.get('total_profit_pct', 0):.2f}%
Max Drawdown: {perf.get('max_drawdown', 0):.2f}%
                    """
                    
                    self.results_preview.setText(preview)
            except Exception as e:
                self._log('error', f"Error loading backtest preview: {str(e)}")
    
    def on_optimization_file_selected(self, index):
        """Handle selection of an optimization file"""
        if index < 0 or self.optimization_file_combo.currentText() == "No optimization results found":
            return
            
        filepath = self.optimization_file_combo.currentData()
        if filepath:
            # Show brief preview info
            try:
                result = self.results_storage.load_optimization_results(filepath)
                if result:
                    best_params = result.get('best_parameters', {})
                    meta = result.get('metadata', {})
                    
                    # Format preview text
                    preview = f"""
Optimization Results Preview - {os.path.basename(filepath)}
-----------------------------------------
Method: {meta.get('optimizer_type', 'Unknown')}
Target Metric: {meta.get('optimization_metric', 'N/A')}
Total Combinations: {meta.get('total_combinations', 'N/A')}
Best Parameters: ATR={best_params.get('atr_length', 'N/A')}, Factor={best_params.get('factor', 'N/A')}, Buffer={best_params.get('buffer_multiplier', 'N/A')}, Stop={best_params.get('hard_stop_distance', 'N/A')}
                    """
                    
                    self.results_preview.setText(preview)
            except Exception as e:
                self._log('error', f"Error loading optimization preview: {str(e)}")
    
    def on_load_backtest(self):
        """Load selected backtest results"""
        if self.backtest_file_combo.currentText() == "No backtest results found":
            return
            
        filepath = self.backtest_file_combo.currentData()
        if not filepath:
            return
            
        try:
            self._log('info', f"Loading backtest results from {filepath}")
            self.backtest_results = self.results_storage.load_backtest_results(filepath)
            
            if self.backtest_results:
                # Update parameter inputs with loaded values
                params = self.backtest_results.get('parameters', {})
                self.atr_length_input.setValue(params.get('atr_length', 14))
                self.factor_input.setValue(params.get('factor', 3.0))
                self.buffer_input.setValue(params.get('buffer_multiplier', 0.3))
                self.stop_input.setValue(params.get('hard_stop_distance', 50))
                
                # Update UI
                self.update_backtest_results()
                
                # Switch to backtest tab
                self.tabs.setCurrentIndex(self.tabs.indexOf(self.tabs.findChild(QWidget, "Backtest")))
                
                self.status_bar.showMessage(f"Backtest results loaded from {os.path.basename(filepath)}")
                
        except Exception as e:
            self._log('error', f"Error loading backtest results: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error loading backtest results:\n{str(e)}")
    
    def on_load_optimization(self):
        """Load selected optimization results"""
        if self.optimization_file_combo.currentText() == "No optimization results found":
            return
            
        filepath = self.optimization_file_combo.currentData()
        if not filepath:
            return
            
        try:
            self._log('info', f"Loading optimization results from {filepath}")
            self.optimization_results = self.results_storage.load_optimization_results(filepath)
            
            if self.optimization_results:
                # Update parameter inputs with best values
                best_params = self.optimization_results.get('best_parameters', {})
                self.atr_length_input.setValue(best_params.get('atr_length', 14))
                self.factor_input.setValue(best_params.get('factor', 3.0))
                self.buffer_input.setValue(best_params.get('buffer_multiplier', 0.3))
                self.stop_input.setValue(best_params.get('hard_stop_distance', 50))
                
                # Update UI
                self.update_optimization_results()
                
                # Switch to optimization tab
                self.tabs.setCurrentIndex(self.tabs.indexOf(self.tabs.findChild(QWidget, "Optimization")))
                
                self.status_bar.showMessage(f"Optimization results loaded from {os.path.basename(filepath)}")
                
        except Exception as e:
            self._log('error', f"Error loading optimization results: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error loading optimization results:\n{str(e)}")
    
    def on_generate_report(self):
        """Generate a report based on selected type and format"""
        report_type = self.report_type_combo.currentText()
        format_type = self.report_format_combo.currentText().lower()
        
        # Check if we have results to report
        if report_type == "Backtest Report" and not self.backtest_results:
            QMessageBox.warning(self, "Warning", "No backtest results available to generate a report.")
            return
            
        if report_type == "Optimization Report" and not self.optimization_results:
            QMessageBox.warning(self, "Warning", "No optimization results available to generate a report.")
            return
        
        try:
            # Generate report
            if report_type == "Backtest Report":
                report_content = self.report_generator.generate_backtest_report(self.backtest_results, format_type)
                report_name = "backtest_report"
            else:  # "Optimization Report"
                report_content = self.report_generator.generate_optimization_report(self.optimization_results, format_type)
                report_name = "optimization_report"
            
            # Save report
            filepath = self.results_storage.save_report(
                report_content, 
                format_type,
                report_name
            )
            
            # Show report in preview
            self.results_preview.setText(report_content)
            
            # Show success message
            if filepath:
                self.status_bar.showMessage(f"Report saved to {filepath}")
                QMessageBox.information(self, "Success", f"Report generated and saved to:\n{filepath}")
            
        except Exception as e:
            self._log('error', f"Error generating report: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error generating report:\n{str(e)}")
    
    def on_export_chart(self):
        """Export the current chart to a file"""
        if not hasattr(self, 'figure') or self.figure is None:
            QMessageBox.warning(self, "Warning", "No chart available to export.")
            return
        
        try:
            # Get the chart name
            chart_type = self.chart_type_combo.currentText()
            chart_name = chart_type.lower().replace(' ', '_')
            
            # Save chart
            filepath = self.results_storage.save_chart(self.figure, chart_name)
            
            # Show success message
            if filepath:
                self.status_bar.showMessage(f"Chart saved to {filepath}")
                QMessageBox.information(self, "Success", f"Chart exported to:\n{filepath}")
            
        except Exception as e:
            self._log('error', f"Error exporting chart: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error exporting chart:\n{str(e)}")
    
    def on_chart_type_changed(self, index):
        """Handle chart type change"""
        self.update_chart()
    
    def on_time_period_changed(self, index):
        """Handle time period change"""
        self.update_chart()
    
    def update_chart(self):
        """Update the chart based on current settings"""
        chart_type = self.chart_type_combo.currentText()
        time_period = self.time_period_combo.currentText()
        
        try:
            # Clear previous figure
            self.figure.clear()
            
            # Filter data by time period if needed
            df_filtered = self.filter_data_by_time_period(self.df, time_period)
            
            # Create chart based on selected type
            if chart_type == "Price with SuperTrend":
                # Check if we have SuperTrend data
                if self.df_st is not None:
                    df_st_filtered = self.filter_data_by_time_period(self.df_st, time_period)
                else:
                    df_st_filtered = df_filtered
                    
                # Draw price chart with SuperTrend
                self.draw_price_supertrend_chart(df_st_filtered)
                
            elif chart_type == "Equity Curve":
                # Check if we have backtest results
                if self.backtest_results:
                    # Draw equity curve
                    self.draw_equity_curve_chart(self.backtest_results)
                else:
                    # No data to draw
                    ax = self.figure.add_subplot(111)
                    ax.text(0.5, 0.5, "No backtest results available", ha='center', va='center')
                    ax.set_axis_off()
            
            elif chart_type == "Drawdown":
                # Check if we have backtest results
                if self.backtest_results:
                    # Draw drawdown chart
                    self.draw_drawdown_chart(self.backtest_results)
                else:
                    # No data to draw
                    ax = self.figure.add_subplot(111)
                    ax.text(0.5, 0.5, "No backtest results available", ha='center', va='center')
                    ax.set_axis_off()
            
            elif chart_type == "Trade Performance":
                # Check if we have backtest results
                if self.backtest_results:
                    # Draw trade performance chart
                    self.draw_trade_performance_chart(self.backtest_results)
                else:
                    # No data to draw
                    ax = self.figure.add_subplot(111)
                    ax.text(0.5, 0.5, "No backtest results available", ha='center', va='center')
                    ax.set_axis_off()
            
            elif chart_type == "Parameter Analysis":
                # Check if we have optimization results
                if self.optimization_results:
                    # Draw parameter analysis chart
                    self.draw_parameter_analysis_chart(self.optimization_results)
                else:
                    # No data to draw
                    ax = self.figure.add_subplot(111)
                    ax.text(0.5, 0.5, "No optimization results available", ha='center', va='center')
                    ax.set_axis_off()
            
            # Update canvas
            self.canvas.draw()
            
        except Exception as e:
            self._log('error', f"Error updating chart: {str(e)}")
            # Show empty chart with error message
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Error creating chart:\n{str(e)}", ha='center', va='center')
            ax.set_axis_off()
            self.canvas.draw()
    
    def filter_data_by_time_period(self, df: pd.DataFrame, time_period: str) -> pd.DataFrame:
        """Filter DataFrame by selected time period"""
        if df is None or df.empty:
            return df
            
        if time_period == "All Data":
            return df
            
        # Get today's date
        today = pd.Timestamp.now()
        
        if time_period == "Last Month":
            start_date = today - pd.Timedelta(days=30)
        elif time_period == "Last Week":
            start_date = today - pd.Timedelta(days=7)
        elif time_period == "Last Day":
            start_date = today - pd.Timedelta(days=1)
        else:
            return df
        
        # Filter data
        return df[df.index >= start_date]
    
    def draw_price_supertrend_chart(self, df: pd.DataFrame):
        """Draw price chart with SuperTrend"""
        if df is None or df.empty:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_axis_off()
            return
            
        # Create subplot
        ax = self.figure.add_subplot(111)
        
        # Plot price
        ax.plot(df.index, df['close'], color='black', linewidth=1.0, label='Close')
        
        # Plot SuperTrend if available
        if 'supertrend' in df.columns and 'direction' in df.columns:
            uptrend_mask = df['direction'] < 0
            downtrend_mask = df['direction'] > 0
            
            # Plot supertrend line
            ax.plot(df.index[uptrend_mask], df['supertrend'][uptrend_mask], color='green', linewidth=1.0, label='SuperTrend (Bull)')
            ax.plot(df.index[downtrend_mask], df['supertrend'][downtrend_mask], color='red', linewidth=1.0, label='SuperTrend (Bear)')
            
            # Plot buffer zones
            if 'up_trend_buffer' in df.columns:
                ax.plot(df.index, df['up_trend_buffer'], color='green', alpha=0.5, linewidth=0.7, linestyle='--', label='Uptrend Buffer')
            
            if 'down_trend_buffer' in df.columns:
                ax.plot(df.index, df['down_trend_buffer'], color='red', alpha=0.5, linewidth=0.7, linestyle='--', label='Downtrend Buffer')
            
            # Plot buy/sell signals
            if 'buy_signal' in df.columns and 'sell_signal' in df.columns:
                buy_signals = df[df['buy_signal']]
                sell_signals = df[df['sell_signal']]
                
                ax.scatter(buy_signals.index, buy_signals['close'], color='green', marker='^', s=80, zorder=5, label='Buy Signal')
                ax.scatter(sell_signals.index, sell_signals['close'], color='red', marker='v', s=80, zorder=5, label='Sell Signal')
        
        # Set title and labels
        params = getattr(self, 'backtest_results', {}).get('parameters', {})
        if params:
            title = f"SuperTrend (ATR={params.get('atr_length', 'N/A')}, Factor={params.get('factor', 'N/A')}, Buffer={params.get('buffer_multiplier', 'N/A')})"
        else:
            title = "Price Chart with SuperTrend"
            
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.figure.autofmt_xdate()
        
        # Add legend
        ax.legend(loc='upper left')
    
    def draw_equity_curve_chart(self, results: Dict[str, Any]):
        """Draw equity curve chart from backtest results"""
        equity_curve = results.get('performance', {}).get('equity_curve', None)
        
        if equity_curve is None or (isinstance(equity_curve, dict) and equity_curve.get('_type') != 'DataFrame'):
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No equity curve data available", ha='center', va='center')
            ax.set_axis_off()
            return
            
        # Create subplot
        ax = self.figure.add_subplot(111)
        
        # Convert serialized DataFrame back to actual DataFrame
        if isinstance(equity_curve, dict) and equity_curve.get('_type') == 'DataFrame':
            try:
                df_equity = pd.DataFrame(equity_curve['data'])
                df_equity.index = pd.to_datetime(equity_curve['index'])
                
                # Plot equity curve
                ax.plot(df_equity.index, df_equity['equity'], color='blue', linewidth=1.2)
                
                # Add drawdown shading
                if 'drawdown' in df_equity.columns:
                    prev_idx = df_equity.index[0]
                    for i in range(1, len(df_equity)):
                        if df_equity['drawdown'].iloc[i] > 0:
                            ax.axvspan(prev_idx, df_equity.index[i], color='red', alpha=0.1)
                        prev_idx = df_equity.index[i]
                
                # Set title and labels
                ax.set_title("Equity Curve")
                ax.set_xlabel('Date')
                ax.set_ylabel('Equity')
                ax.grid(True, alpha=0.3)
                
                # Format x-axis dates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                self.figure.autofmt_xdate()
                
                # Add performance stats as text
                performance = results.get('performance', {})
                stats_text = f"Total Return: {performance.get('total_profit_pct', 0):.2f}%  |  "
                stats_text += f"Max DD: {performance.get('max_drawdown', 0):.2f}%  |  "
                stats_text += f"Sharpe: {performance.get('sharpe_ratio', 0):.2f}"
                
                self.figure.text(0.5, 0.01, stats_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
                
            except Exception as e:
                self._log('error', f"Error plotting equity curve: {str(e)}")
                ax.text(0.5, 0.5, f"Error plotting equity curve:\n{str(e)}", ha='center', va='center')
                ax.set_axis_off()
        else:
            ax.text(0.5, 0.5, "No equity curve data available", ha='center', va='center')
            ax.set_axis_off()
    
    def draw_drawdown_chart(self, results: Dict[str, Any]):
        """Draw drawdown chart from backtest results"""
        equity_curve = results.get('performance', {}).get('equity_curve', None)
        
        if equity_curve is None or (isinstance(equity_curve, dict) and equity_curve.get('_type') != 'DataFrame'):
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No drawdown data available", ha='center', va='center')
            ax.set_axis_off()
            return
            
        # Create subplot
        ax = self.figure.add_subplot(111)
        
        # Convert serialized DataFrame back to actual DataFrame
        if isinstance(equity_curve, dict) and equity_curve.get('_type') == 'DataFrame':
            try:
                df_equity = pd.DataFrame(equity_curve['data'])
                df_equity.index = pd.to_datetime(equity_curve['index'])
                
                if 'drawdown_pct' in df_equity.columns:
                    # Plot drawdown
                    ax.fill_between(df_equity.index, 0, -df_equity['drawdown_pct'], color='red', alpha=0.5)
                    
                    # Add max drawdown line
                    max_dd = df_equity['drawdown_pct'].max()
                    ax.axhline(y=-max_dd, color='r', linestyle='--', alpha=0.8)
                    ax.text(df_equity.index[-1], -max_dd, f" Max DD: {max_dd:.2f}%", va='center')
                    
                    # Set title and labels
                    ax.set_title("Drawdown Chart")
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Drawdown (%)')
                    ax.grid(True, alpha=0.3)
                    
                    # Format y-axis as percentage
                    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
                    
                    # Format x-axis dates
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    self.figure.autofmt_xdate()
                    
                else:
                    ax.text(0.5, 0.5, "No drawdown data available", ha='center', va='center')
                    ax.set_axis_off()
                
            except Exception as e:
                self._log('error', f"Error plotting drawdown: {str(e)}")
                ax.text(0.5, 0.5, f"Error plotting drawdown:\n{str(e)}", ha='center', va='center')
                ax.set_axis_off()
        else:
            ax.text(0.5, 0.5, "No drawdown data available", ha='center', va='center')
            ax.set_axis_off()
    
    def draw_trade_performance_chart(self, results: Dict[str, Any]):
        """Draw trade performance chart from backtest results"""
        trades = results.get('trades', [])
        
        if not trades:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No trade data available", ha='center', va='center')
            ax.set_axis_off()
            return
            
        # Convert trade objects to dictionaries if necessary
        if trades and not isinstance(trades[0], dict):
            trades_list = [t.to_dict() for t in trades]
        else:
            trades_list = trades
            
        # Create subplots
        fig = self.figure
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Trade P&L chart
        ax1 = fig.add_subplot(gs[0, :])
        
        # Extract data
        trade_ids = [t.get('trade_id', i) for i, t in enumerate(trades_list)]
        profits = [t.get('profit_pct', 0) for t in trades_list]
        colors = ['green' if p > 0 else 'red' for p in profits]
        
        # Plot bars
        ax1.bar(trade_ids, profits, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Set title and labels
        ax1.set_title("Trade Performance (P&L %)")
        ax1.set_xlabel('Trade #')
        ax1.set_ylabel('Profit/Loss (%)')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Win/Loss pie chart
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Calculate win/loss count
        win_count = sum(1 for p in profits if p > 0)
        loss_count = sum(1 for p in profits if p < 0)
        even_count = sum(1 for p in profits if p == 0)
        
        # Create pie chart
        labels = ['Wins', 'Losses', 'Breakeven']
        sizes = [win_count, loss_count, even_count]
        colors = ['green', 'red', 'gray']
        explode = (0.1, 0, 0)
        
        if sum(sizes) > 0:  # Only create pie if we have trades
            ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
            ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            ax2.set_title(f"Win/Loss Distribution (Win Rate: {win_count/len(trades_list)*100:.1f}%)")
        else:
            ax2.text(0.5, 0.5, "No trade data", ha='center', va='center')
            ax2.set_axis_off()
        
        # Trade duration chart
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Extract durations
        durations = [t.get('duration_hours', 0) for t in trades_list]
        
        if durations:
            # Create histogram
            ax3.hist(durations, bins=10, color='blue', alpha=0.7)
            ax3.set_title("Trade Duration Distribution")
            ax3.set_xlabel('Duration (hours)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No duration data", ha='center', va='center')
            ax3.set_axis_off()
    
    def draw_parameter_analysis_chart(self, results: Dict[str, Any]):
        """Draw parameter analysis chart from optimization results"""
        param_analysis = results.get('parameter_analysis', {})
        
        if not param_analysis:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No parameter analysis data available", ha='center', va='center')
            ax.set_axis_off()
            return
        
        # Generate chart using report generator
        chart_fig = self.report_generator.generate_chart_optimization(results)
        
        # Copy the figure to our canvas
        for ax in chart_fig.get_axes():
            self.figure.add_axes(ax)
    
    def on_load_data(self):
        """Open file dialog to select data file"""
        self.on_browse_file()
    
    def on_save_results(self):
        """Save current results to file"""
        # Check what results we have
        if hasattr(self, 'backtest_results') and self.backtest_results:
            result_type = "backtest"
            results = self.backtest_results
        elif hasattr(self, 'optimization_results') and self.optimization_results:
            result_type = "optimization"
            results = self.optimization_results
        else:
            QMessageBox.warning(self, "Warning", "No results to save.")
            return
        
        # Get name from user
        name, ok = QInputDialog.getText(
            self, 
            "Save Results", 
            f"Enter a name for these {result_type} results:",
            QLineEdit.Normal, 
            f"{result_type}_{datetime.now().strftime('%Y%m%d')}"
        )
        
        if not ok or not name:
            return
        
        # Save results
        try:
            if result_type == "backtest":
                filepath = self.results_storage.save_backtest_results(results, name)
            else:  # optimization
                filepath = self.results_storage.save_optimization_results(results, name)
            
            if filepath:
                self.status_bar.showMessage(f"Results saved to {filepath}")
                QMessageBox.information(self, "Success", f"Results saved to:\n{filepath}")
                
                # Refresh file lists
                self.refresh_backtest_files()
                self.refresh_optimization_files()
            
        except Exception as e:
            self._log('error', f"Error saving results: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error saving results:\n{str(e)}")
    
    def on_export_report(self):
        """Export a report"""
        # Switch to Results tab
        self.tabs.setCurrentIndex(self.tabs.indexOf(self.tabs.findChild(QWidget, "Results")))
        
        # Call generate report method
        self.on_generate_report()
    
    def on_about(self):
        """Show about dialog"""
        about_text = f"""
<h2>SuperTrend Backtester v{APP_VERSION}</h2>

<p>Created by: {self.current_user}</p>
<p>Current UTC: {self.current_utc}</p>

<p>This application implements the SuperTrend technical indicator with CPU and GPU optimizations, 
along with comprehensive backtesting and parameter optimization tools.</p>

<p>Features include:</p>
<ul>
<li>SuperTrend implementation (CPU, GPU with CUDA)</li>
<li>Backtesting engine with trade management</li>
<li>Parameter optimization (Grid Search, Bayesian, Genetic)</li>
<li>Performance analysis and visualization</li>
<li>Result management and reporting</li>
</ul>
"""
        QMessageBox.about(self, "About SuperTrend Backtester", about_text)


# ==============================================================================
# MAIN APPLICATION & ENTRY POINT
# ==============================================================================

def main():
    """Main entry point for the application"""
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create a unique run directory with timestamp and ID
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    run_id = generate_unique_id()  # This function already exists in the code
    run_dir = f"SuperTrend_Run_{timestamp}_{run_id}"
    
    # Create the full path for this run
    base_dir = os.path.join(script_dir, run_dir)
    os.makedirs(base_dir, exist_ok=True)
    
    # Set up logging with this new base directory
    log_manager = LogManager(base_dir)
    log_manager.setup_logging()
    log_manager.info(f"Starting SuperTrend Backtester v{APP_VERSION}")
    
    # Create QApplication instance
    app = QApplication(sys.argv)
    
    try:
        # Apply styles
        if hasattr(app, 'setStyle') and 'Fusion' in QStyleFactory.keys():
            app.setStyle('Fusion')
        
        # Create main window
        main_window = SuperTrendGUI(log_manager)
        main_window.show()
        
        # Run application
        sys.exit(app.exec_())
        
    except Exception as e:
        log_manager.error(f"Unhandled exception: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        log_manager.info("Application shutting down")
        log_manager.shutdown()

if __name__ == "__main__":
    main()
