import os
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
import glob
import psutil
import warnings
import traceback
import csv
import gc

from tqdm import tqdm
from backtrader import Cerebro, Strategy, DataFeed
from ray import init, shutdown

# Initialize RAY for distributed computing
init(ignore_reinit_error=True)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Constants
RAM_THRESHOLD = 0.5  # Trigger CSV dump when RAM usage exceeds 50%
OPTIMIZATION_CSV = "optimization_results.csv"
MASTER_DIR_NAME = f"Execution_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# Directory structure
os.makedirs(MASTER_DIR_NAME, exist_ok=True)
CSV_DUMP_DIR = os.path.join(MASTER_DIR_NAME, "csv_dumps")
TRADE_LOG_DIR = os.path.join(MASTER_DIR_NAME, "trade_logs")
ERROR_LOG_FILE = os.path.join(MASTER_DIR_NAME, "error_log.txt")
os.makedirs(CSV_DUMP_DIR, exist_ok=True)
os.makedirs(TRADE_LOG_DIR, exist_ok=True)

# Error logging mechanism
def log_error(e):
    """Log errors to a text file."""
    with open(ERROR_LOG_FILE, "a") as log_file:
        log_file.write(f"{datetime.now()}: {traceback.format_exc()}\n")


# Function for RAM usage monitoring
def is_ram_overloaded():
    """Check if the RAM usage exceeds the threshold."""
    memory = psutil.virtual_memory()
    return memory.percent > (RAM_THRESHOLD * 100)


# Function to write results to CSV
def dump_results_to_csv(results):
    """Append results to a CSV file."""
    file_path = os.path.join(CSV_DUMP_DIR, OPTIMIZATION_CSV)
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
    # Clear memory after every dump
    results.clear()
    gc.collect()


# Backtest using Backtrader
class SuperTrendStrategy(Strategy):
    """SuperTrend Strategy with trade logging."""
    params = (
        ("period", 14),
        ("multiplier", 3.0),
    )

    def __init__(self):
        self.atr = self.datas[0].atr(self.params.period)
        self.supertrend = self.datas[0].supertrend(self.params.multiplier)
        self.trades = []  # Store trade data

    def next(self):
        if self.supertrend.trend == 1 and not self.position:
            self.buy()
            self.trades.append(
                {"action": "BUY", "datetime": self.datas[0].datetime.datetime(0), "price": self.data.close[0]}
            )
        elif self.supertrend.trend == -1 and self.position:
            self.sell()
            self.trades.append(
                {"action": "SELL", "datetime": self.datas[0].datetime.datetime(0), "price": self.data.close[0]}
            )


def backtest_and_log_trades(df, period, multiplier):
    """Run backtest using Backtrader and return trades."""
    cerebro = Cerebro()
    data = DataFeed(dataname=df)
    cerebro.adddata(data)
    strategy = cerebro.addstrategy(SuperTrendStrategy, period=period, multiplier=multiplier)
    cerebro.run()
    return strategy[0].trades


def analyze_csv_dumps():
    """Analyze all CSV dumps and find the top 3 combinations."""
    all_files = glob.glob(os.path.join(CSV_DUMP_DIR, "*.csv"))
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    top_combinations = df.nlargest(3, "profit")
    return top_combinations


def resume_last_processed():
    """Resume processing from the last processed combination."""
    file_path = os.path.join(CSV_DUMP_DIR, OPTIMIZATION_CSV)
    if not os.path.isfile(file_path):
        return None
    df = pd.read_csv(file_path)
    if df.empty:
        return None
    last_combination = df.iloc[-1].to_dict()
    return last_combination


# Main function
def main():
    print("=" * 50)
    print(" SUPER TREND STRATEGY OPTIMIZER ".center(50, "="))
    print("=" * 50)

    try:
        # Load OHLC data
        file_path = input("Enter the path to the OHLC data file: ").strip()
        df = pd.read_csv(file_path)

        # Define parameter ranges
        params = {
            "periods": range(5, 50, 5),
            "multipliers": np.arange(1.0, 10.0, 0.5),
            "price_ranges": np.arange(0.01, 1.0, 0.1),
        }

        # Resume from the last processed combination if available
        last_processed = resume_last_processed()
        param_combinations = list(product(params["periods"], params["multipliers"], params["price_ranges"]))
        if last_processed:
            start_index = param_combinations.index(tuple(last_processed.values())) + 1
            param_combinations = param_combinations[start_index:]

        # Optimize parameters
        results = []
        for combo in tqdm(param_combinations):
            if is_ram_overloaded():
                dump_results_to_csv(results)

            period, multiplier, price_range = combo
            try:
                result = backtest_and_log_trades(df, period, multiplier)
                results.append({"period": period, "multiplier": multiplier, "price_range": price_range, "profit": result})
            except Exception as e:
                log_error(e)

        dump_results_to_csv(results)

        # Analyze CSV dumps to find the top 3 combinations
        top_combinations = analyze_csv_dumps()

        # Generate trade logs for the top 3 combinations
        all_trades = []
        for _, row in top_combinations.iterrows():
            trades = backtest_and_log_trades(df, period=row["period"], multiplier=row["multiplier"])
            all_trades.extend(trades)

        # Save trades to a CSV file
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv(os.path.join(TRADE_LOG_DIR, "top_combinations_trades.csv"), index=False)

        # Delete error log file if no errors occurred
        if os.path.isfile(ERROR_LOG_FILE) and os.stat(ERROR_LOG_FILE).st_size == 0:
            os.remove(ERROR_LOG_FILE)

    except Exception as e:
        log_error(e)


if __name__ == "__main__":
    main()