# Part 1: Main structure and imports
import os
import csv
import hashlib
import datetime
import numpy as np
import cupy as cp
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

class DataChecker:
    def __init__(self):
        self.current_time = "2025-06-14 18:43:31"
        self.current_user = "arullr001"
        self.version = "1.0.0"
        self.gpu_device = None
        
    def initialize_gpu(self):
        """Initialize NVIDIA GPU for processing"""
        try:
            self.gpu_device = cp.cuda.Device(1)  # GTX 1050 Ti
            self.gpu_device.use()
            return True
        except Exception as e:
            raise RuntimeError(f"GPU initialization failed: {str(e)}")

    def create_directory_structure(self, file_path: str) -> Path:
        """Create timestamped directory for processing"""
        base_name = Path(file_path).stem
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"processed_data/{base_name}/{timestamp}")
        
        # Create required directories
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "raw_data").mkdir(exist_ok=True)
        
        return output_dir

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of input file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
		
		
		
# Part 2: Data loading and validation functions
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate CSV data"""
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_columns = ['date', 'time', 'O', 'H', 'L', 'C', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Combine date and time columns
            df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], utc=True)
            
            return df
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}")

    def detect_timeframe(self, df: pd.DataFrame) -> str:
        """Detect timeframe from time series data"""
        time_diff = df['timestamp'].diff().mode()[0]
        minutes = time_diff.total_seconds() / 60
        
        timeframe_map = {
            1: "1min",
            5: "5min",
            15: "15min",
            30: "30min",
            60: "1hour",
            240: "4hour",
            1440: "1day"
        }
        
        return timeframe_map.get(minutes, f"{int(minutes)}min")

    def validate_price_data(self, df: pd.DataFrame) -> List[Dict]:
        """Validate OHLC price data"""
        errors = []
        
        # Convert to GPU arrays for faster processing
        with cp.cuda.Device(1):
            h = cp.array(df['H'].values)
            l = cp.array(df['L'].values)
            o = cp.array(df['O'].values)
            c = cp.array(df['C'].values)
            
            # Check price inversions
            invalid_hl = cp.where(h < l)[0]
            invalid_ho = cp.where(h < o)[0]
            invalid_hc = cp.where(h < c)[0]
            invalid_ol = cp.where(o < l)[0]
            invalid_cl = cp.where(c < l)[0]
            
        # Convert GPU arrays back to CPU and process errors
        for idx in cp.asnumpy(invalid_hl):
            errors.append({
                'row': idx + 2,  # +2 for CSV header and 0-based index
                'timestamp': df.iloc[idx]['timestamp'],
                'type': 'PRICE_INVERSION',
                'severity': 'CRITICAL',
                'description': f"High price ({df.iloc[idx]['H']}) < Low price ({df.iloc[idx]['L']})",
                'context': df.iloc[max(0, idx-1):min(len(df), idx+2)]
            })
            
        return errors

    def validate_time_series(self, df: pd.DataFrame, timeframe: str) -> List[Dict]:
        """Validate time series continuity"""
        warnings = []
        
        # Convert timeframe to minutes
        tf_minutes = int(''.join(filter(str.isdigit, timeframe)))
        expected_diff = pd.Timedelta(minutes=tf_minutes)
        
        # Find gaps
        time_diff = df['timestamp'].diff()
        gaps = df[time_diff > expected_diff].index
        
        for idx in gaps:
            warnings.append({
                'row': idx + 2,
                'timestamp': df.iloc[idx]['timestamp'],
                'type': 'TIME_GAP',
                'severity': 'WARNING',
                'description': f"Gap of {time_diff[idx]} detected",
                'context': df.iloc[max(0, idx-1):min(len(df), idx+2)]
            })
            
        return warnings
		
		
# Part 3: Health score and statistical analysis
    def calculate_health_score(self, df: pd.DataFrame, errors: List[Dict], warnings: List[Dict]) -> Dict:
        """Calculate data health score and statistics"""
        total_rows = len(df)
        
        # Initialize statistics dictionary
        stats = {
            'timestamp': "2025-06-14 18:46:04",
            'user': "arullr001",
            'data_health': {
                'total_score': 0,
                'components': {
                    'price_integrity': 0,
                    'time_series': 0,
                    'data_quality': 0
                }
            },
            'statistics': {
                'price_range': {
                    'high': df['H'].max(),
                    'low': df['L'].min(),
                    'spread': df['H'].max() - df['L'].min()
                },
                'volume': {
                    'total': df['volume'].sum(),
                    'average': df['volume'].mean(),
                    'zero_volume_periods': (df['volume'] == 0).sum(),
                    'zero_volume_percentage': (df['volume'] == 0).sum() / total_rows * 100
                },
                'identical_prices': {
                    'count': len(df[df['O'] == df['H']][df['H'] == df['L']][df['L'] == df['C']]),
                    'percentage': len(df[df['O'] == df['H']][df['H'] == df['L']][df['L'] == df['C']]) / total_rows * 100
                }
            }
        }
        
        # Calculate component scores
        price_errors = len([e for e in errors if e['type'] == 'PRICE_INVERSION'])
        time_gaps = len([w for w in warnings if w['type'] == 'TIME_GAP'])
        
        # Price integrity score (40% of total)
        price_score = 40 * (1 - (price_errors / total_rows))
        
        # Time series score (40% of total)
        time_score = 40 * (1 - (time_gaps / total_rows))
        
        # Data quality score (20% of total)
        quality_score = 20 * (1 - (stats['statistics']['zero_volume_percentage'] / 100))
        
        # Update health scores
        stats['data_health']['components']['price_integrity'] = round(price_score, 2)
        stats['data_health']['components']['time_series'] = round(time_score, 2)
        stats['data_health']['components']['data_quality'] = round(quality_score, 2)
        stats['data_health']['total_score'] = round(sum([
            price_score,
            time_score,
            quality_score
        ]), 2)
        
        return stats

    def analyze_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze price and volume patterns"""
        with cp.cuda.Device(1):
            # Convert to GPU arrays
            high = cp.array(df['H'].values)
            low = cp.array(df['L'].values)
            close = cp.array(df['C'].values)
            volume = cp.array(df['volume'].values)
            
            # Calculate basic patterns
            price_range = high - low
            price_movement = cp.abs(close[1:] - close[:-1])
            volume_changes = cp.abs(volume[1:] - volume[:-1])
            
        # Convert back to CPU for analysis
        patterns = {
            'price_patterns': {
                'average_range': float(cp.mean(price_range)),
                'max_range': float(cp.max(price_range)),
                'average_movement': float(cp.mean(price_movement)),
                'max_movement': float(cp.max(price_movement))
            },
            'volume_patterns': {
                'average_change': float(cp.mean(volume_changes)),
                'max_change': float(cp.max(volume_changes)),
                'zero_volume_sequences': self._find_zero_volume_sequences(df)
            }
        }
        
        return patterns

    def _find_zero_volume_sequences(self, df: pd.DataFrame) -> List[Dict]:
        """Find sequences of zero volume periods"""
        zero_volume = df['volume'] == 0
        sequences = []
        
        if not zero_volume.any():
            return sequences
            
        # Find start and end of zero volume sequences
        sequence_start = None
        for idx, is_zero in enumerate(zero_volume):
            if is_zero and sequence_start is None:
                sequence_start = idx
            elif not is_zero and sequence_start is not None:
                sequences.append({
                    'start': df.iloc[sequence_start]['timestamp'],
                    'end': df.iloc[idx-1]['timestamp'],
                    'duration': idx - sequence_start,
                    'start_row': sequence_start + 2,
                    'end_row': idx + 1
                })
                sequence_start = None
                
        # Handle sequence at end of data
        if sequence_start is not None:
            sequences.append({
                'start': df.iloc[sequence_start]['timestamp'],
                'end': df.iloc[-1]['timestamp'],
                'duration': len(df) - sequence_start,
                'start_row': sequence_start + 2,
                'end_row': len(df) + 1
            })
            
        return sequences
		
		
# Part 4: Report generation and file output
    def generate_reports(self, df: pd.DataFrame, results: Dict, output_dir: Path):
        """Generate TXT and CSV reports"""
        self._generate_txt_report(df, results, output_dir)
        self._generate_csv_report(results, output_dir)

    def _generate_txt_report(self, df: pd.DataFrame, results: Dict, output_dir: Path):
        """Generate detailed TXT report"""
        report_path = output_dir / f"report_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w') as f:
            # Header
            f.write("===== Data Validation Report =====\n")
            f.write(f"Generated: 2025-06-14 18:47:01 UTC\n")
            f.write(f"User: arullr001\n")
            f.write(f"File: {output_dir.parent.name}\n\n")

            # File Information
            f.write("File Information:\n")
            f.write(f"- SHA256 Hash: {results['file_hash']}\n")
            f.write(f"- File Size: {results['file_size']:.2f} KB\n")
            f.write("- Time Zone: UTC\n\n")

            # Data Overview
            f.write("Data Overview:\n")
            f.write(f"- Total Rows: {len(df)}\n")
            f.write(f"- Date Range: {df['timestamp'].min()} UTC to {df['timestamp'].max()} UTC\n")
            f.write(f"- Detected Timeframe: {results['timeframe']}\n")
            f.write(f"- Columns: {', '.join(df.columns)}\n\n")

            # Health Score
            f.write("Data Health Score:\n")
            f.write(f"Overall Score: {results['health']['data_health']['total_score']}/100\n")
            for component, score in results['health']['data_health']['components'].items():
                f.write(f"- {component.replace('_', ' ').title()}: {score}/100\n")
            f.write("\n")

            # Critical Errors
            f.write("Critical Errors:\n")
            if results['errors']:
                for error in results['errors']:
                    f.write(f"\nRow {error['row']} - {error['type']}:\n")
                    f.write(f"Timestamp: {error['timestamp']}\n")
                    f.write(f"Description: {error['description']}\n")
                    f.write("Context:\n")
                    for _, row in error['context'].iterrows():
                        f.write(f"{row['timestamp']}: O={row['O']}, H={row['H']}, L={row['L']}, C={row['C']}, V={row['volume']}\n")
            else:
                f.write("No critical errors found.\n")
            f.write("\n")

            # Warnings
            f.write("Warnings:\n")
            if results['warnings']:
                for warning in results['warnings']:
                    f.write(f"\nRow {warning['row']} - {warning['type']}:\n")
                    f.write(f"Timestamp: {warning['timestamp']}\n")
                    f.write(f"Description: {warning['description']}\n")
            else:
                f.write("No warnings found.\n")
            f.write("\n")

            # Statistical Summary
            f.write("Statistical Summary:\n")
            stats = results['health']['statistics']
            f.write("Price Statistics:\n")
            f.write(f"- Range: {stats['price_range']['low']} - {stats['price_range']['high']}\n")
            f.write(f"- Spread: {stats['price_range']['spread']}\n\n")
            
            f.write("Volume Statistics:\n")
            f.write(f"- Total Volume: {stats['volume']['total']}\n")
            f.write(f"- Average Volume: {stats['volume']['average']:.2f}\n")
            f.write(f"- Zero Volume Periods: {stats['volume']['zero_volume_periods']} ")
            f.write(f"({stats['volume']['zero_volume_percentage']:.2f}%)\n\n")

            f.write("Price Patterns:\n")
            f.write(f"- Identical OHLC Periods: {stats['identical_prices']['count']} ")
            f.write(f"({stats['identical_prices']['percentage']:.2f}%)\n")

    def _generate_csv_report(self, results: Dict, output_dir: Path):
        """Generate CSV report of issues"""
        csv_path = output_dir / f"issues_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Combine errors and warnings
        issues = results['errors'] + results['warnings']
        
        if not issues:
            return
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp_utc',
                'row_number',
                'error_type',
                'severity',
                'description',
                'context_before',
                'error_row',
                'context_after'
            ])
            
            writer.writeheader()
            
            for issue in issues:
                context_df = issue['context']
                writer.writerow({
                    'timestamp_utc': issue['timestamp'],
                    'row_number': issue['row'],
                    'error_type': issue['type'],
                    'severity': issue['severity'],
                    'description': issue['description'],
                    'context_before': context_df.iloc[0].to_json() if len(context_df) > 1 else "",
                    'error_row': context_df.iloc[1 if len(context_df) > 1 else 0].to_json(),
                    'context_after': context_df.iloc[2].to_json() if len(context_df) > 2 else ""
                })
				
				
# Part 5: Main processing function and CLI
    def process_file(self, file_path: str) -> bool:
        """Main processing function"""
        try:
            # Create lock file
            lock_file = Path(file_path).with_suffix('.lock')
            if lock_file.exists():
                raise RuntimeError("File is already being processed")
            lock_file.touch()

            try:
                # Initialize GPU
                self.initialize_gpu()

                # Create output directory
                output_dir = self.create_directory_structure(file_path)

                # Calculate file hash and size
                file_hash = self.calculate_file_hash(file_path)
                file_size = Path(file_path).stat().st_size / 1024  # Size in KB

                # Load data
                print(f"\nProcessing: {Path(file_path).name}")
                df = self.load_data(file_path)
                total_rows = len(df)

                # Initialize progress bar
                pbar = tqdm(total=100, desc="Progress", ncols=100)
                pbar.update(10)  # Data loading complete

                # Detect timeframe
                timeframe = self.detect_timeframe(df)
                pbar.update(10)  # Timeframe detection complete

                # Validate price data
                errors = self.validate_price_data(df)
                pbar.update(20)  # Price validation complete

                # Validate time series
                warnings = self.validate_time_series(df, timeframe)
                pbar.update(20)  # Time series validation complete

                # Calculate health score and statistics
                health = self.calculate_health_score(df, errors, warnings)
                patterns = self.analyze_patterns(df)
                pbar.update(20)  # Analysis complete

                # Prepare results
                results = {
                    'timestamp': "2025-06-14 18:48:48",
                    'user': "arullr001",
                    'file_hash': file_hash,
                    'file_size': file_size,
                    'timeframe': timeframe,
                    'errors': errors,
                    'warnings': warnings,
                    'health': health,
                    'patterns': patterns
                }

                # Generate reports
                self.generate_reports(df, results, output_dir)
                pbar.update(20)  # Report generation complete

                # Copy original file to raw_data directory
                raw_data_path = output_dir / "raw_data" / Path(file_path).name
                with open(file_path, 'rb') as src, open(raw_data_path, 'wb') as dst:
                    dst.write(src.read())

                pbar.close()
                print(f"\nProcessing complete. Reports saved in: {output_dir}")
                
                return True

            finally:
                # Clean up lock file
                if lock_file.exists():
                    lock_file.unlink()

        except Exception as e:
            print(f"\nError during processing: {str(e)}")
            return False

def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Validation Tool")
    parser.add_argument('file', help='Path to the CSV file to process')
    args = parser.parse_args()

    checker = DataChecker()
    checker.process_file(args.file)

if __name__ == "__main__":
    main()
	
	

