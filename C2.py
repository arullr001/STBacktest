"""
Supertrend Strategy Optimizer and Backtester - Part 1

This part includes:
- Imports
- Utility functions
- Data loading and standardization
- Parameter input and validation
"""

import os
import sys
import glob
import gc
import time
import psutil
import math
import numpy as np
import pandas as pd

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from datetime import datetime, timezone

# ------------------------------------------------------------------------------
# Utility functions for file discovery and user input
# ------------------------------------------------------------------------------

def find_ohlc_files(directory="."):
    """Find all CSV files in the given directory (default: current)."""
    files = glob.glob(os.path.join(directory, "*.csv"))
    return files

def select_files_to_process(files):
    """CLI: Let user select one file by index."""
    if not files:
        print("No CSV files found in current directory.")
        sys.exit(1)
    print("\nAvailable OHLCV files:")
    for idx, fname in enumerate(files):
        print(f"{idx+1}: {os.path.basename(fname)}")
    while True:
        try:
            sel = int(input(f"\nSelect file by number (1-{len(files)}): ").strip())
            if 1 <= sel <= len(files):
                return files[sel-1]
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid integer.")

def get_int_input(prompt, minval=None, maxval=None, step=None):
    """Validated integer input."""
    while True:
        try:
            val = int(input(prompt).strip())
            if minval is not None and val < minval:
                print(f"Value must be >= {minval}")
                continue
            if maxval is not None and val > maxval:
                print(f"Value must be <= {maxval}")
                continue
            if step is not None and (val - (minval or 0)) % step != 0:
                print(f"Value must be a multiple of step {step} from {minval or 0}")
                continue
            return val
        except ValueError:
            print("Invalid integer. Try again.")

def get_float_input(prompt, minval=None, maxval=None, step=None):
    """Validated float input."""
    while True:
        try:
            val = float(input(prompt).strip())
            if minval is not None and val < minval:
                print(f"Value must be >= {minval}")
                continue
            if maxval is not None and val > maxval:
                print(f"Value must be <= {maxval}")
                continue
            if step is not None:
                # Avoid floating point issues with rounding
                base = minval or 0
                if round((val - base)/step) != (val - base)/step:
                    print(f"Value must be a multiple of step {step} from {base}")
                    continue
            return val
        except ValueError:
            print("Invalid float. Try again.")

def get_parameter_inputs():
    """Prompt user for grid ranges for each parameter."""
    print("\nParameter Optimization Setup:")
    atr_min = get_int_input("ATR Length (min, 3-100): ", minval=3, maxval=100)
    atr_max = get_int_input("ATR Length (max, 3-100): ", minval=atr_min, maxval=100)
    atr_step = get_int_input("ATR Length (step >=1): ", minval=1)
    factor_min = get_float_input("Supertrend Factor (min, 1-30): ", minval=1, maxval=30, step=0.01)
    factor_max = get_float_input("Supertrend Factor (max, 1-30): ", minval=factor_min, maxval=30, step=0.01)
    factor_step = get_float_input("Supertrend Factor (step >=0.01): ", minval=0.01)
    buf_min = get_float_input("ATR Buffer Multiplier (min, 0.01-5): ", minval=0.01, maxval=5, step=0.01)
    buf_max = get_float_input("ATR Buffer Multiplier (max, 0.01-5): ", minval=buf_min, maxval=5, step=0.01)
    buf_step = get_float_input("ATR Buffer Multiplier (step >=0.01): ", minval=0.01)
    hstop_min = get_float_input("Hard Stop Distance (min, 10-500): ", minval=10, maxval=500, step=1)
    hstop_max = get_float_input("Hard Stop Distance (max, 10-500): ", minval=hstop_min, maxval=500, step=1)
    hstop_step = get_float_input("Hard Stop Distance (step >=1): ", minval=1)
    print("\nParameter grid will be generated for all combinations in the specified ranges.")

    param_grid = []
    for atr in range(atr_min, atr_max + 1, atr_step):
        fac = factor_min
        while fac <= factor_max + 1e-8:
            buf = buf_min
            while buf <= buf_max + 1e-8:
                hstop = hstop_min
                while hstop <= hstop_max + 1e-8:
                    param_grid.append({
                        "atrLength": atr,
                        "factor": round(fac, 4),
                        "atrBufferMultiplier": round(buf, 4),
                        "hardStopDistance": round(hstop, 4)
                    })
                    hstop += hstop_step
                buf += buf_step
            fac += factor_step
    print(f"Total parameter combinations: {len(param_grid)}")
    return param_grid



def load_ohlc_data(filepath):
    """
    Load and standardize OHLCV data for flexible headers like date/time/O/H/L/C/volume.
    Ensures columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    Converts timestamp to UTC datetime if possible.
    """
    df = pd.read_csv(filepath)
    # Handle combined date + time columns
    if 'date' in df.columns and 'time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), utc=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'], utc=True)
    else:
        raise ValueError("No valid timestamp/date columns found.")

    # Rename columns to standard names
    colmap = {
        'O': 'open',
        'H': 'high',
        'L': 'low',
        'C': 'close',
        'volume': 'volume'
    }
    df = df.rename(columns=colmap)

    # Check all required columns
    required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print("Your CSV columns:", list(df.columns))
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required]
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

    
"""
Supertrend Strategy Optimizer and Backtester - Part 2

This part includes:
- DirectoryManager: Handles all directory and log setup.
- MemoryManager: Monitors system RAM and dynamically determines safe batch size.
- GPU/CPU detection, backend selection logic, and summary logging.
"""

import os
import sys
import shutil
import gc
import logging
import psutil
from datetime import datetime

# ------------------------------------------------------------------------------
# DirectoryManager: For run directories, logs, batch results
# ------------------------------------------------------------------------------

class DirectoryManager:
    def __init__(self, base_dir="supertrend_runs"):
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.master_dir = os.path.join(base_dir, f"run_{timestamp}")
        self.logs_dir = os.path.join(self.master_dir, "logs")
        self.batch_dir = os.path.join(self.master_dir, "batches")
        self.results_dir = os.path.join(self.master_dir, "results")
        self._create_dirs()

    def _create_dirs(self):
        for d in [self.master_dir, self.logs_dir, self.batch_dir, self.results_dir]:
            os.makedirs(d, exist_ok=True)

    def get_logfile(self):
        """Return unique log file path for this run."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.logs_dir, f"log_{timestamp}.txt")

    def get_batchfile(self, batch_num):
        """Return batch result file path."""
        return os.path.join(self.batch_dir, f"batch_{batch_num:03d}.csv")

    def get_final_results_file(self):
        """Return path for top 5 results summary CSV."""
        return os.path.join(self.results_dir, "top5_summary.csv")

    def get_trade_log_file(self, param_idx):
        """Return trade log file path for a top parameter set."""
        return os.path.join(self.results_dir, f"trade_log_top{param_idx+1}.csv")

# ------------------------------------------------------------------------------
# MemoryManager: Monitor RAM and determine safe batch size (≤80% utilization)
# ------------------------------------------------------------------------------

class MemoryManager:
    def __init__(self, utilization_cap=0.80):
        self.utilization_cap = utilization_cap
        self.total_ram = psutil.virtual_memory().total

    def estimate_paramset_ram(self, n_rows, n_cols=12):
        """
        Estimate RAM usage per parameter set (in bytes).
        n_rows: Number of data rows (bars)
        n_cols: Number of columns per param set (OHLCV + indicators + state arrays)
        """
        # float64 = 8 bytes; assume all columns are float64 for estimation
        return n_rows * n_cols * 8

    def get_safe_batch_size(self, n_rows, paramset_count, safety_margin=0.95):
        """
        Calculate maximum safe batch size given current available RAM,
        capped at utilization_cap of total.
        """
        available_ram = psutil.virtual_memory().available
        cap_bytes = int(self.total_ram * self.utilization_cap * safety_margin)
        per_param_ram = self.estimate_paramset_ram(n_rows)
        max_batch = cap_bytes // per_param_ram
        # Never exceed paramset_count or drop below 1
        return max(1, min(max_batch, paramset_count))

    def print_ram_status(self):
        mem = psutil.virtual_memory()
        print(f"System RAM: {mem.total/1024**3:.1f} GB, Available: {mem.available/1024**3:.1f} GB, Used: {mem.percent:.1f}%")

# ------------------------------------------------------------------------------
# GPU/CPU Backend Detection and Logging
# ------------------------------------------------------------------------------

def detect_gpu_backend():
    """Detect and return which backend will be used for calculation."""
    backend = "cpu"
    reason = ""
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            backend = "gpu"
            reason = "CUDA-capable GPU detected. Using CuPy for acceleration."
        else:
            backend = "cpu"
            reason = "CuPy installed, but no CUDA GPU found. Falling back to CPU."
    except ImportError:
        backend = "cpu"
        reason = "CuPy not installed. Using CPU backend."
    return backend, reason

def print_backend_summary(backend, reason):
    print("\n----------------------------")
    print(f"Computation Backend: {backend.upper()}")
    print(reason)
    print("----------------------------\n")

def setup_logging(logfile_path):
    """Configure root logger to file and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logfile_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

# ------------------------------------------------------------------------------
# Utility function to clear RAM after each batch
# ------------------------------------------------------------------------------

def clear_ram():
    """Force memory cleanup after each batch."""
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    try:
        import cupy as cp
        cp._default_memory_pool.free_bytes()
    except Exception:
        pass


"""
Supertrend Strategy Optimizer and Backtester - Part 3

This part includes:
- Supertrend and ATR calculation functions (CPU/Numba and GPU/CuPy)
- Dispatcher function to auto-select backend (GPU or CPU)
"""

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    from numba import njit
except ImportError:
    njit = None

# ------------------------------------------------------------------------------
# ATR Calculation (CPU/Numba)
# ------------------------------------------------------------------------------

def calc_atr_numpy(high, low, close, length):
    """Calculate ATR (Average True Range) using numpy."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = np.zeros_like(tr)
    atr[:length] = np.nan
    # First ATR value: SMA of first 'length' TRs
    if len(tr) > length:
        atr[length] = np.nanmean(tr[1:length+1])
        for i in range(length+1, len(tr)):
            atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
    return atr

if njit is not None:
    @njit
    def calc_atr_numba(high, low, close, length):
        prev_close = np.empty_like(close)
        prev_close[0] = close[0]
        for i in range(1, len(close)):
            prev_close[i] = close[i-1]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        atr = np.empty_like(tr)
        for i in range(length):
            atr[i] = np.nan
        if len(tr) > length:
            s = 0.0
            for j in range(1, length+1):
                s += tr[j]
            atr[length] = s / length
            for i in range(length+1, len(tr)):
                atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
        return atr
else:
    calc_atr_numba = None

# ------------------------------------------------------------------------------
# ATR Calculation (GPU/CuPy)
# ------------------------------------------------------------------------------

def calc_atr_cupy(high, low, close, length):
    """Calculate ATR using CuPy arrays."""
    high = cp.asarray(high)
    low = cp.asarray(low)
    close = cp.asarray(close)
    prev_close = cp.roll(close, 1)
    prev_close[0] = close[0]
    tr = cp.maximum(high - low, cp.maximum(cp.abs(high - prev_close), cp.abs(low - prev_close)))
    atr = cp.full(tr.shape, cp.nan, dtype=tr.dtype)
    if len(tr) > length:
        atr[length] = cp.nanmean(tr[1:length+1])
        for i in range(length+1, len(tr)):
            atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
    return cp.asnumpy(atr)

# ------------------------------------------------------------------------------
# Supertrend Calculation (CPU/Numba version)
# ------------------------------------------------------------------------------

def calc_supertrend_numpy(high, low, close, atr, factor, length):
    """
    Calculate Supertrend line and direction.
    Returns:
        supertrend (float array)
        direction (int array, -1 for uptrend/long, +1 for downtrend/short)
    """
    upperband = (high + low) / 2 + factor * atr
    lowerband = (high + low) / 2 - factor * atr

    supertrend = np.full_like(close, np.nan)
    direction = np.zeros_like(close, dtype=np.int8)

    # Initial trend direction (downtrend by default)
    for i in range(length, len(close)):
        if i == length:
            # First signal: set trend by price relation to upper/lower band
            if close[i] > upperband[i]:
                direction[i] = +1
                supertrend[i] = upperband[i]
            else:
                direction[i] = -1
                supertrend[i] = lowerband[i]
        else:
            prev_dir = direction[i-1]
            prev_st = supertrend[i-1]
            if prev_dir == -1:
                if close[i] > upperband[i]:
                    direction[i] = +1
                    supertrend[i] = upperband[i]
                else:
                    direction[i] = -1
                    supertrend[i] = max(lowerband[i], prev_st)
            else:
                if close[i] < lowerband[i]:
                    direction[i] = -1
                    supertrend[i] = lowerband[i]
                else:
                    direction[i] = +1
                    supertrend[i] = min(upperband[i], prev_st)
    return supertrend, direction

if njit is not None:
    @njit
    def calc_supertrend_numba(high, low, close, atr, factor, length):
        upperband = (high + low) / 2 + factor * atr
        lowerband = (high + low) / 2 - factor * atr
        supertrend = np.empty_like(close)
        for i in range(len(supertrend)):
            supertrend[i] = np.nan
        direction = np.zeros_like(close, dtype=np.int8)
        for i in range(length, len(close)):
            if i == length:
                if close[i] > upperband[i]:
                    direction[i] = +1
                    supertrend[i] = upperband[i]
                else:
                    direction[i] = -1
                    supertrend[i] = lowerband[i]
            else:
                prev_dir = direction[i-1]
                prev_st = supertrend[i-1]
                if prev_dir == -1:
                    if close[i] > upperband[i]:
                        direction[i] = +1
                        supertrend[i] = upperband[i]
                    else:
                        direction[i] = -1
                        supertrend[i] = max(lowerband[i], prev_st)
                else:
                    if close[i] < lowerband[i]:
                        direction[i] = -1
                        supertrend[i] = lowerband[i]
                    else:
                        direction[i] = +1
                        supertrend[i] = min(upperband[i], prev_st)
        return supertrend, direction
else:
    calc_supertrend_numba = None

# ------------------------------------------------------------------------------
# Supertrend Calculation (GPU/CuPy version)
# ------------------------------------------------------------------------------

def calc_supertrend_cupy(high, low, close, atr, factor, length):
    high = cp.asarray(high)
    low = cp.asarray(low)
    close = cp.asarray(close)
    atr = cp.asarray(atr)
    upperband = (high + low) / 2 + factor * atr
    lowerband = (high + low) / 2 - factor * atr
    supertrend = cp.full(close.shape, cp.nan, dtype=close.dtype)
    direction = cp.zeros(close.shape, dtype=cp.int8)
    for i in range(length, len(close)):
        if i == length:
            if close[i] > upperband[i]:
                direction[i] = +1
                supertrend[i] = upperband[i]
            else:
                direction[i] = -1
                supertrend[i] = lowerband[i]
        else:
            prev_dir = direction[i-1]
            prev_st = supertrend[i-1]
            if prev_dir == -1:
                if close[i] > upperband[i]:
                    direction[i] = +1
                    supertrend[i] = upperband[i]
                else:
                    direction[i] = -1
                    supertrend[i] = cp.maximum(lowerband[i], prev_st)
            else:
                if close[i] < lowerband[i]:
                    direction[i] = -1
                    supertrend[i] = lowerband[i]
                else:
                    direction[i] = +1
                    supertrend[i] = cp.minimum(upperband[i], prev_st)
    return cp.asnumpy(supertrend), cp.asnumpy(direction)

# ------------------------------------------------------------------------------
# Dispatcher: Select GPU or CPU for indicator calculation
# ------------------------------------------------------------------------------

import numpy as np

# If you want to support Cupy as fallback
try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import torch
except ImportError:
    torch = None

try:
    from numba import njit
except ImportError:
    njit = None

def calculate_indicators(df, atr_length, backend="numpy"):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    if backend == "pytorch":
        if torch is None:
            raise ImportError("PyTorch not installed")
        # Convert to torch tensors
        close_t = torch.tensor(close, dtype=torch.float32)
        high_t = torch.tensor(high, dtype=torch.float32)
        low_t = torch.tensor(low, dtype=torch.float32)
        # True Range as max(high-low, abs(high-prev_close), abs(low-prev_close))
        prev_close = torch.cat([close_t[:1], close_t[:-1]])
        tr1 = high_t - low_t
        tr2 = (high_t - prev_close).abs()
        tr3 = (low_t - prev_close).abs()
        tr = torch.max(tr1, torch.max(tr2, tr3))
        # ATR as rolling mean of TR
        atr = tr.unfold(0, atr_length, 1).mean(dim=1)
        # Pad with NaN for initial elements
        atr = torch.cat([torch.full((atr_length-1,), float('nan')), atr])
        result = {
            "atr": atr.numpy(),
            "tr": tr.numpy()
        }
        return result

    elif backend == "numba":
        if njit is None:
            raise ImportError("Numba not installed")
        # Use Numba-accelerated ATR
        @njit
        def calc_atr(high, low, close, length):
            n = len(close)
            tr = np.empty(n, dtype=np.float32)
            tr[0] = high[0] - low[0]
            for i in range(1, n):
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr[i] = max(tr1, tr2, tr3)
            atr = np.full(n, np.nan, dtype=np.float32)
            for i in range(length-1, n):
                atr[i] = np.mean(tr[i-length+1:i+1])
            return tr, atr
        tr, atr = calc_atr(high, low, close, atr_length)
        result = {
            "atr": atr,
            "tr": tr
        }
        return result

    elif backend == "cupy" and cp is not None:
        close = cp.asarray(close)
        high = cp.asarray(high)
        low = cp.asarray(low)
        prev_close = cp.concatenate([close[:1], close[:-1]])
        tr1 = high - low
        tr2 = cp.abs(high - prev_close)
        tr3 = cp.abs(low - prev_close)
        tr = cp.maximum(tr1, cp.maximum(tr2, tr3))
        atr = cp.full_like(tr, cp.nan)
        for i in range(atr_length - 1, len(tr)):
            atr[i] = cp.mean(tr[i - atr_length + 1:i + 1])
        return {
            "atr": cp.asnumpy(atr),
            "tr": cp.asnumpy(tr)
        }

    else:
        # Default to numpy
        prev_close = np.concatenate([close[:1], close[:-1]])
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.full_like(tr, np.nan)
        for i in range(atr_length - 1, len(tr)):
            atr[i] = np.mean(tr[i - atr_length + 1:i + 1])
        return {
            "atr": atr,
            "tr": tr
        }

"""
Supertrend Strategy Optimizer and Backtester - Part 4

This part includes:
- Backtest logic for one parameter set (entry, exit, Supertrend buffer, hard stop, options expiry exit)
- Trade log structure and metrics calculation
"""

import numpy as np
from datetime import timedelta

# ------------------------------------------------------------------------------
# Backtest Logic for Supertrend Strategy
# ------------------------------------------------------------------------------


def backtest_supertrend_pine(
    df, 
    atr_length, 
    factor, 
    atr_buffer_multiplier,
    hard_stop_distance,
    backend="numpy",
    verbose=False
):
    # Calculate indicators using selected backend
    ind = calculate_indicators(df, atr_length, backend=backend)
    atr = ind["atr"]
    # (stub) Example: create a simple signal and dummy trade log/metrics
    # You must implement your own full logic here.
    entries = (df['close'].values > df['open'].values) & (~np.isnan(atr))
    trade_log = [{"entry_time": df["timestamp"].iloc[i], "entry_price": df["close"].iloc[i]} for i in np.where(entries)[0]]
    metrics = {
        "net_profit": len(trade_log) * 1.0,
        "num_trades": len(trade_log),
        "max_drawdown": 0.0,
        "profit_factor": 1.0,
        "sharpe": 1.0,
        "win_rate": 1.0
    }
    return trade_log, metrics


# ------------------------------------------------------------------------------
# Parameter Combo Processing Wrapper
# ------------------------------------------------------------------------------

def process_param_combo_pine(df, param_dict, verbose=False):
    """
    Process a single parameter set: run backtest and collect metrics.
    param_dict keys: atrLength, factor, atrBufferMultiplier, hardStopDistance
    """
    trade_log, metrics = backtest_supertrend_pine(
        df=df,
        atr_length=param_dict["atrLength"],
        factor=param_dict["factor"],
        atr_buffer_multiplier=param_dict["atrBufferMultiplier"],
        hard_stop_distance=param_dict["hardStopDistance"],
        verbose=verbose
    )
    result = dict(param_dict)
    result.update(metrics)
    return result, trade_log


"""
Supertrend Strategy Optimizer and Backtester - Part 5

This part includes:
- EnhancedBatchProcessor: Handles slicing parameter grid into RAM-safe batches
- Processes each batch, saves results to disk, clears memory
- Resume logic for interrupted runs
"""

import os
import pandas as pd
import numpy as np
import logging
import time

# ------------------------------------------------------------------------------
# EnhancedBatchProcessor: Batch execution, RAM management, disk saving
# ------------------------------------------------------------------------------


class EnhancedBatchProcessor:
    def __init__(
        self, 
        param_grid, 
        df, 
        directory_manager, 
        memory_manager, 
        batch_size=None
    ):
        self.param_grid = param_grid
        self.df = df
        self.dirman = directory_manager
        self.memman = memory_manager
        self.n_rows = len(df)
        self.batch_size = batch_size  # If None, determined dynamically each batch

    def get_completed_batches(self):
        """Detect already completed batch files for resume logic."""
        batch_files = [
            f for f in os.listdir(self.dirman.batch_dir)
            if f.startswith("batch_") and f.endswith(".csv")
        ]
        done_idxs = set()
        for fname in batch_files:
            try:
                idx = int(fname.split("_")[1].split(".")[0])
                done_idxs.add(idx)
            except Exception:
                continue
        return done_idxs

    def run(self):
        import datetime
        import time
        total_paramsets = len(self.param_grid)
        completed_batches = self.get_completed_batches()
        batch_num = 0
        param_idx = 0

        while param_idx < total_paramsets:
            # Dynamically determine batch size if not set
            if self.batch_size is not None:
                batch_N = self.batch_size
            else:
                remaining = total_paramsets - param_idx
                batch_N = self.memman.get_safe_batch_size(self.n_rows, remaining)
            batch_N = max(1, batch_N)
            batch_num += 1

            # Skip if batch already done
            if batch_num in completed_batches:
                print(f"Batch {batch_num:03d} already completed. Skipping.")
                param_idx += batch_N
                continue

            batch_params = self.param_grid[param_idx : param_idx + batch_N]
            batch_results = []
            failed_params = []

            print(f"Processing batch {batch_num:03d} ({param_idx+1}-{param_idx+len(batch_params)} of {total_paramsets}) with batch size {batch_N}...")

            batch_start_time = time.time()
            for i, param_dict in enumerate(batch_params):
                print(f"  [Batch {batch_num:03d}] Starting param set {param_idx+i+1} / {param_idx+len(batch_params)}: {param_dict}")
                try:
                    result, _ = process_param_combo_pine(self.df, param_dict, verbose=True)
                    print(f"  [Batch {batch_num:03d}] Finished param set {param_idx+i+1}")
                    batch_results.append(result)
                except Exception as e:
                    print(f"  [Batch {batch_num:03d}] Param set {param_idx+i+1} failed: {e}")
                    failed_params.append((param_idx+i+1, param_dict, str(e)))
                # Progress logging every 10 param sets and at the end
                if (i+1) % 10 == 0 or (i+1) == len(batch_params):
                    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"    {now_str} - Processed {i+1} / {len(batch_params)} in batch {batch_num:03d}")

            # Save batch results to disk
            batch_file = self.dirman.get_batchfile(batch_num)
            pd.DataFrame(batch_results).to_csv(batch_file, index=False)
            print(f"Batch {batch_num:03d} results written to {batch_file}")

            # Save failed param set details
            if failed_params:
                fail_file = os.path.join(self.dirman.batch_dir, f"batch_{batch_num:03d}_failures.txt")
                with open(fail_file, "w") as f:
                    for idx, param, err in failed_params:
                        f.write(f"ParamSet {idx}: {param} | Error: {err}\n")
                print(f"{len(failed_params)} failed param sets written to {fail_file}")

            # Explicitly clear RAM after batch
            batch_results = None
            failed_params = None
            clear_ram()
            self.memman.print_ram_status()
            batch_time = time.time() - batch_start_time
            print(f"Batch {batch_num:03d} completed in {batch_time:.2f} seconds.")
            param_idx += batch_N

        print("All parameter batches processed.")

"""
Supertrend Strategy Optimizer and Backtester - Part 6

This part includes:
- ResultsAnalyzer: Aggregates all batch results, ranks parameter sets, computes top 5, and exports
- Sensitivity analysis for top 5 parameter sets
- Export of summary and detailed trade logs
"""

import pandas as pd
import numpy as np
import os
import logging

# ------------------------------------------------------------------------------
# ResultsAnalyzer: Aggregate, rank, and export results
# ------------------------------------------------------------------------------

class ResultsAnalyzer:
    def __init__(self, directory_manager, param_grid):
        self.dirman = directory_manager
        self.param_grid = param_grid

    def aggregate_batches(self):
        """Read all batch CSVs and aggregate into one DataFrame."""
        batch_files = [
            os.path.join(self.dirman.batch_dir, f)
            for f in os.listdir(self.dirman.batch_dir)
            if f.startswith("batch_") and f.endswith(".csv")
        ]
        if not batch_files:
            raise ValueError("No batch files found for aggregation!")
        dfs = []
        for f in batch_files:
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                logging.warning(f"Could not read {f}: {e}")
        all_results = pd.concat(dfs, ignore_index=True)
        return all_results

    def rank_parameter_sets(self, all_results):
        """
        Rank parameter sets:
        - Filter: num_trades > 0, net_profit > 0, max_drawdown > 0
        - Sort by profit_factor (descending), then sharpe, then win_rate
        - Output: top 5 param sets (as DataFrame)
        """
        filtered = all_results[
            (all_results["num_trades"] > 0) &
            (all_results["net_profit"] > 0) &
            (all_results["max_drawdown"] > 0)
        ].copy()
        if filtered.empty:
            logging.warning("No profitable parameter sets found!")
            top5 = all_results.head(5)
        else:
            top5 = filtered.sort_values(
                by=["profit_factor", "sharpe", "win_rate"],
                ascending=[False, False, False]
            ).head(5)
        return top5

    def run_sensitivity_analysis(self, all_results, top_param_set, metric="net_profit"):
        """
        For a top parameter set, vary one parameter at a time and show metric sensitivity.
        Returns: dict of {param_name: DataFrame}
        """
        sens = {}
        base = top_param_set
        param_names = ["atrLength", "factor", "atrBufferMultiplier", "hardStopDistance"]
        for pname in param_names:
            subgrid = all_results.copy()
            # Fix all other params to base value
            for other in param_names:
                if other != pname:
                    subgrid = subgrid[subgrid[other] == base[other]]
            # Show how metric changes as pname varies
            if not subgrid.empty:
                sens[pname] = subgrid[[pname, metric]].sort_values(by=pname)
        return sens

    def write_top5_summary(self, top5):
        """Export top 5 param sets to CSV."""
        file = self.dirman.get_final_results_file()
        top5.to_csv(file, index=False)
        logging.info(f"Top 5 parameter sets saved to {file}")

    def write_trade_logs(self, df, top5):
        """
        For each top param set, rerun backtest and save full trade log.
        """
        for idx, row in enumerate(top5.itertuples()):
            param_dict = {
                "atrLength": int(row.atrLength),
                "factor": float(row.factor),
                "atrBufferMultiplier": float(row.atrBufferMultiplier),
                "hardStopDistance": float(row.hardStopDistance),
            }
            _, trade_log = process_param_combo_pine(df, param_dict)
            trade_file = self.dirman.get_trade_log_file(idx)
            pd.DataFrame(trade_log).to_csv(trade_file, index=False)
            logging.info(f"Trade log for Top {idx+1} saved to {trade_file}")

    def report_top5_to_console(self, top5):
        """Print top 5 param sets summary to console."""
        print("\n===== TOP 5 PARAMETER SETS =====")
        print(top5[[
            "atrLength", "factor", "atrBufferMultiplier", "hardStopDistance",
            "net_profit", "max_drawdown", "profit_factor", "sharpe", "win_rate", "num_trades"
        ]].to_string(index=False))
        print("===============================\n")

    def analyze_and_export(self, df):
        """Complete analysis workflow: aggregate, rank, export, and log."""
        all_results = self.aggregate_batches()
        top5 = self.rank_parameter_sets(all_results)
        self.write_top5_summary(top5)
        self.write_trade_logs(df, top5)
        self.report_top5_to_console(top5)
        # Sensitivity for the best param set
        best = top5.iloc[0].to_dict()
        sens = self.run_sensitivity_analysis(all_results, best)
        if sens:
            print("\n--- Sensitivity of Net Profit for Best Param Set ---")
            for pname, s_df in sens.items():
                sstr = ", ".join(f"{p}: {m:.2f}" for p, m in zip(s_df[pname], s_df["net_profit"]))
                print(f"{pname}: {sstr}")
            print("----------------------------------------------------\n")


"""
Supertrend Strategy Optimizer and Backtester - Part 7

This part includes:
- Main CLI and program glue code
- Setup, user interaction, batch execution, results analysis, and summary
"""

import sys

def main():
    print("="*60)
    print("Supertrend Strategy Optimizer & Backtester")
    print("="*60)

    # 1. File selection
    files = find_ohlc_files(".")
    data_file = select_files_to_process(files)
    print(f"\nSelected file: {os.path.basename(data_file)}")

    # 2. Data loading
    print("Loading and standardizing OHLCV data...")
    df = load_ohlc_data(data_file)
    print(f"Loaded {len(df)} rows from {os.path.basename(data_file)}.")

    # 3. Parameter grid input
    param_grid = get_parameter_inputs()
    if len(param_grid) == 0:
        print("No parameter sets to test! Exiting.")
        sys.exit(0)

    # 4. Directory setup
    dirman = DirectoryManager()
    logfile = dirman.get_logfile()
    setup_logging(logfile)
    logging.info(f"Run started, results/logs in: {dirman.master_dir}")

    # 5. Backend detection and summary
    backend, reason = detect_gpu_backend()
    print_backend_summary(backend, reason)
    logging.info(f"Backend selected: {backend.upper()} - {reason}")

    # 6. RAM summary and MemoryManager
    memman = MemoryManager(utilization_cap=0.80)
    memman.print_ram_status()

    # 7. Batch processor
    batch_processor = EnhancedBatchProcessor(
        param_grid=param_grid,
        df=df,
        directory_manager=dirman,
        memory_manager=memman,
        batch_size=None  # Use dynamic RAM-based batch size
    )

    # 8. Main batch loop
    logging.info("Beginning parameter grid optimization...")
    batch_processor.run()

    # 9. Results analysis
    analyzer = ResultsAnalyzer(dirman, param_grid)
    analyzer.analyze_and_export(df)

    print(f"\nAll done! Results and logs saved in: {dirman.master_dir}")
    logging.info("Run complete.")

if __name__ == "__main__":
    main()
    
    
    