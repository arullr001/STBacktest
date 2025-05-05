import requests
import csv
import datetime
import time  # For sleep if needed
import os

# --- Original Script Starts Here ---
params = {
'resolution': "1m",
'symbol': "BTCUSD",
'start': "1712745270",
'end': "1712746220"
}
response = requests.get("https://cdn.india.deltaex.org/v2/history/candles", params=params)
historical_data = response.json()
print("Original API response:")
print(historical_data)
# --- Original Script Ends Here ---

# --- Added Functions Start Here ---

def setup_data_directory():
    """
    Create a 'data' directory if it doesn't exist.
    Returns the path to the data directory.
    """
    data_dir = os.path.join("D:", "Delta Exchange", "Backtesting Main", "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir

def generate_filename(symbol, resolution, start_date_short, end_date_short):
    """
    Generate a filename for the CSV file including the generation timestamp.
    """
    # Get current UTC timestamp
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    # Create filename with timestamp
    filename = f"{symbol.lower()}_{resolution}_{start_date_short}_to_{end_date_short}_gen{timestamp}.csv"
    
    # Prepend data directory
    data_dir = setup_data_directory()
    return os.path.join(data_dir, filename)


def select_date_range(start_date, end_date, format="%Y-%m-%d %H:%M:%S"):
    """
    Convert date strings to Unix timestamps for API parameters.
    
    Args:
        start_date (str): Start date in format specified by 'format' parameter
        end_date (str): End date in format specified by 'format' parameter
        format (str): Date format string (default: "%Y-%m-%d %H:%M:%S")
        
    Returns:
        tuple: (start_timestamp, end_timestamp) as strings
    """
    try:
        # Parse the date strings to datetime objects
        start_datetime = datetime.datetime.strptime(start_date, format)
        end_datetime = datetime.datetime.strptime(end_date, format)
        
        # Check if dates are in the future
        now = datetime.datetime.now()
        if start_datetime > now or end_datetime > now:
            print("Error: Cannot select future dates for historical data.")
            return None, None
        
        # Convert to Unix timestamps (seconds since epoch)
        start_timestamp = str(int(start_datetime.timestamp()))
        end_timestamp = str(int(end_datetime.timestamp()))
        
        print(f"Converted date range:")
        print(f"  {start_date} → Unix timestamp: {start_timestamp}")
        print(f"  {end_date} → Unix timestamp: {end_timestamp}")
        
        return start_timestamp, end_timestamp
    
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        print(f"Please ensure dates are in format: {format}")
        return None, None

def get_historical_data_by_date_range(symbol, resolution, start_date, end_date, date_format="%Y-%m-%d %H:%M:%S"):
    """
    Fetch historical data for a specific date range.
    
    Args:
        symbol (str): Trading pair symbol (e.g., "BTCUSD")
        resolution (str): Candle resolution (e.g., "1m", "1h", "1d")
        start_date (str): Start date in specified format
        end_date (str): End date in specified format
        date_format (str): Input date format
        
    Returns:
        list: List of candle dictionaries
    """
    # Convert dates to timestamps
    start_ts, end_ts = select_date_range(start_date, end_date, date_format)
    
    if not start_ts or not end_ts:
        print("Failed to convert dates to timestamps. Aborting.")
        return None
    
    # Prepare request parameters
    params = {
        'resolution': resolution,
        'symbol': symbol,
        'start': start_ts,
        'end': end_ts
    }
    
    print(f"Fetching data for {symbol}, {resolution} from {start_date} to {end_date}...")
    
    try:
        response = requests.get("https://cdn.india.deltaex.org/v2/history/candles", params=params)
        response.raise_for_status()  # Raise exception for non-200 status codes
        data = response.json()
        
        # Extract candle data from response
        candles = get_candle_data(data)
        print(f"Successfully retrieved {len(candles)} candles.")
        return candles
        
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        return None
    except ValueError as e:
        print(f"JSON parsing error: {e}")
        return None
        
        
def get_candle_data(json_response):
    """
    Extract candle data from API response.
    
    Args:
        json_response (dict): JSON response from API
        
    Returns:
        list: List of candle dictionaries
    """
    # Check if the response is already a list
    if isinstance(json_response, list):
        return json_response
    # Check if response is a dict with 'result' key containing a list
    elif isinstance(json_response, dict) and 'result' in json_response and isinstance(json_response['result'], list):
        return json_response['result']
    else:
        print("Warning: Unknown response format. Returning empty list.")
        return []

def filter_candles_by_date_range(candles, start_date, end_date, date_format="%Y-%m-%d %H:%M:%S"):
    """
    Filter candles to keep only those within specified date range.
    
    Args:
        candles (list): List of candle dictionaries
        start_date (str): Start date in specified format
        end_date (str): End date in specified format
        date_format (str): Date format string
        
    Returns:
        list: Filtered candles within date range
    """
    if not candles:
        return []
        
    try:
        # Convert date strings to timestamps for comparison
        start_ts = int(datetime.datetime.strptime(start_date, date_format).timestamp())
        end_ts = int(datetime.datetime.strptime(end_date, date_format).timestamp())
        
        # Filter candles within range
        filtered_candles = [
            candle for candle in candles 
            if 'time' in candle and start_ts <= candle['time'] <= end_ts
        ]
        
        print(f"Filtered {len(filtered_candles)} candles within date range "
              f"({len(candles) - len(filtered_candles)} excluded)")
        
        return filtered_candles
        
    except ValueError as e:
        print(f"Error filtering by date range: {e}")
        return candles  # Return original data on error

def format_candle_data(candles, include_date_time=True):
    """
    Format candle data for CSV export.
    
    Args:
        candles (list): List of candle dictionaries
        include_date_time (bool): Whether to split time into date and time columns
        
    Returns:
        list: Formatted candle data with human-readable date/time
    """
    formatted_candles = []
    
    for candle in candles:
        # Create a new dict for the formatted candle
        formatted_candle = {}
        
        # Convert Unix timestamp to readable date and time
        if 'time' in candle:
            dt = datetime.datetime.fromtimestamp(candle['time'])
            
            if include_date_time:
                # Split into separate date and time columns
                formatted_candle['date'] = dt.strftime('%Y-%m-%d')
                formatted_candle['time'] = dt.strftime('%H:%M:%S')
            else:
                # Keep as a single datetime column
                formatted_candle['datetime'] = dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Add OHLC values with shorter column names
        if 'open' in candle:
            formatted_candle['O'] = candle['open']
        if 'high' in candle:
            formatted_candle['H'] = candle['high']
        if 'low' in candle:
            formatted_candle['L'] = candle['low']
        if 'close' in candle:
            formatted_candle['C'] = candle['close']
            
        # Add volume if present
        if 'volume' in candle:
            formatted_candle['volume'] = candle['volume']
            
        formatted_candles.append(formatted_candle)
        
    return formatted_candles
    
    
def save_to_csv(data, filename="btc_candles.csv"):
    """
    Save candle data to CSV file.
    
    Args:
        data (list): List of formatted candle dictionaries
        filename (str): Output filename
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not data:
        print("No data to save.")
        return False
        
    try:
        # Get column headers from first dictionary
        fieldnames = data[0].keys()
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
            
        print(f"Successfully saved {len(data)} candles to {filename}")
        return True
        
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return False


def interactive_date_selection():
    """
    Interactive function to get date range from user input.
    
    Returns:
        tuple: (start_date, end_date, symbol, resolution)
    """
    print("\n=== Historical Candle Data Selection ===")
    
    # Display current UTC time and user
    current_utc = datetime.datetime.utcnow()
    print(f"Current Date and Time (UTC): {current_utc.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current User's Login: arullr001")
    
    # Get symbol
    symbol = input("Enter trading pair symbol (default: BTCUSD): ").strip() or "BTCUSD"
    
    # Get resolution
    resolution_options = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "3d", "1w", "2w", "1M"]
    print("\nAvailable resolutions:")
    print("  Short-term: 1m, 5m, 15m, 30m")
    print("  Medium-term: 1h, 2h, 4h, 6h, 12h")
    print("  Long-term: 1d, 3d, 1w, 2w, 1M")
    print("For large date ranges (months/year), consider using 1h or 1d to reduce data volume")
    resolution = input(f"Enter resolution (default: 1d): ").strip() or "1d"
    
    # Validate resolution
    if resolution not in resolution_options:
        print(f"Warning: '{resolution}' is not a standard resolution. Proceeding anyway.")
    
    # Get date format preference
    date_format = "%Y-%m-%d %H:%M:%S"  # Default format
    format_choice = input("Use format 'YYYY-MM-DD'? (Y/n): ").strip().lower()
    
    if format_choice == 'n':
        date_format = "%d/%m/%Y %H:%M:%S"
        date_input_format = "%d/%m/%Y"
        date_display_format = "DD/MM/YYYY"
    else:
        date_format = "%Y-%m-%d %H:%M:%S"
        date_input_format = "%Y-%m-%d"
        date_display_format = "YYYY-MM-DD"
    
    # Get yesterday's date at 23:59:59
    today = datetime.datetime.utcnow().date()
    yesterday = today - datetime.timedelta(days=1)
    yesterday_end = datetime.datetime.combine(yesterday, datetime.time(23, 59, 59))

    # Offer preset date ranges with date information
    print("\nPreset date ranges:")
    print(f"  1) Past 7 days from yesterday ({(yesterday - datetime.timedelta(days=7)).strftime('%Y-%m-%d')} to {yesterday.strftime('%Y-%m-%d')})")
    print(f"  2) Past 30 days from yesterday ({(yesterday - datetime.timedelta(days=30)).strftime('%Y-%m-%d')} to {yesterday.strftime('%Y-%m-%d')})")
    print(f"  3) Past 90 days from yesterday ({(yesterday - datetime.timedelta(days=90)).strftime('%Y-%m-%d')} to {yesterday.strftime('%Y-%m-%d')})")
    print(f"  4) Past 6 months from yesterday ({(yesterday - datetime.timedelta(days=180)).strftime('%Y-%m-%d')} to {yesterday.strftime('%Y-%m-%d')})")
    print(f"  5) Past 1 year from yesterday ({(yesterday - datetime.timedelta(days=365)).strftime('%Y-%m-%d')} to {yesterday.strftime('%Y-%m-%d')})")
    print("  6) Custom range")
    
    preset_choice = input("Select a preset or enter custom dates (default: 5): ").strip() or "5"
    
    if preset_choice == "1":  # Last 7 days
        start_date_obj = yesterday - datetime.timedelta(days=7)
        end_date_obj = yesterday
    elif preset_choice == "2":  # Last 30 days
        start_date_obj = yesterday - datetime.timedelta(days=30)
        end_date_obj = yesterday
    elif preset_choice == "3":  # Last 90 days
        start_date_obj = yesterday - datetime.timedelta(days=90)
        end_date_obj = yesterday
    elif preset_choice == "4":  # Last 6 months
        start_date_obj = yesterday - datetime.timedelta(days=180)
        end_date_obj = yesterday
    elif preset_choice == "5":  # Last 1 year
        start_date_obj = yesterday - datetime.timedelta(days=365)
        end_date_obj = yesterday
    else:  # Custom range
        # Set default dates
        default_start = "2024-04-01"
        default_end = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        
        print("\nCustom Date Range Selection:")
        print(f"Default start date: {default_start}")
        print(f"Default end date: {default_end}")
        print(f"Enter dates in {date_display_format} format (press Enter to use defaults)")
        
        # Get start date with default
        start_input = input(f"Enter start date (default: {default_start}): ").strip()
        start_date_str = start_input if start_input else default_start
        
        # Get end date with default
        end_input = input(f"Enter end date (default: {default_end}): ").strip()
        end_date_str = end_input if end_input else default_end
        
        try:
            # Parse input dates
            if date_format.startswith("%Y"):
                start_date_obj = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
                end_date_obj = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
            else:
                start_date_obj = datetime.datetime.strptime(start_date_str, "%d/%m/%Y").date()
                end_date_obj = datetime.datetime.strptime(end_date_str, "%d/%m/%Y").date()
            
            # Validate dates
            if start_date_obj > today or end_date_obj > today:
                print("Error: Cannot select future dates for historical data.")
                return None, None, None, None, None
                
            if end_date_obj < start_date_obj:
                print("Error: End date cannot be before start date.")
                return None, None, None, None, None
                
        except ValueError as e:
            print(f"Error parsing dates: {e}")
            print("Using default dates instead.")
            # Use default dates if there's an error
            start_date_obj = datetime.datetime.strptime(default_start, "%Y-%m-%d").date()
            end_date_obj = datetime.datetime.strptime(default_end, "%Y-%m-%d").date()
    
    # Format the selected dates with time components
    if date_format.startswith("%Y"):
        start_date = f"{start_date_obj.strftime('%Y-%m-%d')} 00:00:00"
        end_date = f"{end_date_obj.strftime('%Y-%m-%d')} 23:59:59"
    else:
        start_date = f"{start_date_obj.strftime('%d/%m/%Y')} 00:00:00"
        end_date = f"{end_date_obj.strftime('%d/%m/%Y')} 23:59:59"
    
    # Calculate and show date range info
    days_in_range = (end_date_obj - start_date_obj).days + 1
    
    # Recommendation for resolution based on range
    if days_in_range > 180 and resolution in ["1m", "5m", "15m"]:
        print(f"\nWarning: You've selected {resolution} resolution for a {days_in_range}-day period.")
        print("This may result in a very large dataset and long processing time.")
        print("Consider using a higher resolution (1h, 4h, 1d) for this time range.")
        confirm_res = input(f"Continue with {resolution} resolution? (y/N): ").strip().lower()
        if confirm_res != 'y':
            resolution = input("Enter new resolution (recommended: 1h or 1d): ").strip() or "1d"
    
    print(f"\nSelected parameters:")
    print(f"  Symbol: {symbol}")
    print(f"  Resolution: {resolution}")
    print(f"  Date range: {start_date} to {end_date} ({days_in_range} days)")
    
    confirm = input("\nProceed with these parameters? (Y/n): ").strip().lower()
    if confirm == 'n':
        print("Operation cancelled.")
        return None, None, None, None, None
    
    return start_date, end_date, symbol, resolution, date_format  
    
    
def process_date_chunks(symbol, resolution, start_date, end_date, date_format, chunk_days=3):
    """
    Process a large date range by breaking it into smaller chunks.
    
    Args:
        symbol (str): Trading pair symbol
        resolution (str): Candle resolution
        start_date (str): Start date string
        end_date (str): End date string
        date_format (str): Date format string
        chunk_days (int): Number of days per chunk
        
    Returns:
        list: Combined candles from all chunks
    """
    # Parse start and end dates
    start_dt = datetime.datetime.strptime(start_date, date_format)
    end_dt = datetime.datetime.strptime(end_date, date_format)
    
    # Calculate total days
    total_days = (end_dt - start_dt).days + 1
    print(f"Fetching {total_days} days of data in chunks of {chunk_days} days each...")
    
    # Initialize empty list for all candles
    all_candles = []
    
    # Process in chunks
    current_start = start_dt
    chunk_num = 1
    
    while current_start <= end_dt:
        # Calculate chunk end (either chunk_days later or end_date, whichever comes first)
        chunk_end = min(current_start + datetime.timedelta(days=chunk_days-1), end_dt)
        
        # Format dates for API call
        chunk_start_str = current_start.strftime(date_format)
        chunk_end_str = chunk_end.strftime(date_format)
        
        print(f"\nFetching chunk {chunk_num}: {chunk_start_str} to {chunk_end_str}")
        
        # Get data for this chunk
        chunk_candles = get_historical_data_by_date_range(
            symbol, resolution, chunk_start_str, chunk_end_str, date_format
        )
        
        if chunk_candles:
            print(f"Retrieved {len(chunk_candles)} candles for this chunk")
            all_candles.extend(chunk_candles)
        else:
            print(f"Warning: No data retrieved for chunk {chunk_num}")
        
        # Move to next chunk
        current_start = chunk_end + datetime.timedelta(days=1)
        chunk_num += 1
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Sort combined results by timestamp
    all_candles.sort(key=lambda x: x['time'])
    
    # Remove potential duplicates (by timestamp)
    unique_candles = []
    seen_timestamps = set()
    
    for candle in all_candles:
        if candle['time'] not in seen_timestamps:
            unique_candles.append(candle)
            seen_timestamps.add(candle['time'])
    
    print(f"\nTotal data retrieved: {len(all_candles)} candles")
    print(f"After removing duplicates: {len(unique_candles)} unique candles")
    
    return unique_candles

def process_historical_data(use_interactive=True):
    """
    Main function to retrieve, process and save historical data.
    
    Args:
        use_interactive (bool): Whether to use interactive mode for date selection
        
    Returns:
        bool: Success status
    """
    # Extract candles from the original response
    original_candles = get_candle_data(historical_data)
    print(f"\nExtracted {len(original_candles)} candles from original response")
    
    # Choose between interactive or preset date range
    if use_interactive:
        # Get parameters through interactive prompt
        start_date, end_date, symbol, resolution, date_format = interactive_date_selection()
        
        if not start_date:  # User cancelled
            return False
            
        # Alert about large date ranges
        start_dt = datetime.datetime.strptime(start_date, date_format)
        end_dt = datetime.datetime.strptime(end_date, date_format)
        date_range_days = (end_dt - start_dt).days + 1
        
        if date_range_days > 30:
            print(f"\nNote: You've selected a large date range ({date_range_days} days).")
            print("This will be processed in smaller chunks to handle API limitations.")
            print("The process may take some time. Please be patient.")
            confirm = input("Continue with this large date range? (Y/n): ").strip().lower()
            if confirm == 'n':
                return False
        
        # Process data in chunks for large date ranges
        if date_range_days > 3:
            candles = process_date_chunks(symbol, resolution, start_date, end_date, date_format)
        else:
            # For small ranges, fetch directly
            candles = get_historical_data_by_date_range(
                symbol, resolution, start_date, end_date, date_format
            )
        
        if not candles:
            print("Failed to retrieve data for the selected date range.")
            # Fall back to original data
            print("Falling back to original data...")
            candles = original_candles
    else:
        # Use preset date range for 1 year of data
        end_date_obj = datetime.datetime.now()
        start_date_obj = end_date_obj - datetime.timedelta(days=365)  # 1 year ago
        
        start_date = start_date_obj.strftime("%Y-%m-%d") + " 00:00:00"
        end_date = end_date_obj.strftime("%Y-%m-%d") + " 23:59:00"
        symbol = "BTCUSD"
        resolution = "1d"  # Daily resolution for a year of data
        
        print(f"\nUsing preset date range for 1 year: {start_date} to {end_date}")
        
        # Process year of data in chunks
        candles = process_date_chunks(
            symbol, resolution, start_date, end_date, "%Y-%m-%d %H:%M:%S"
        )
        
        if not candles:
            print("Failed to retrieve data for the preset year range.")
            # Fall back to original data
            print("Falling back to original data...")
            candles = original_candles
    
    # Format the candle data with date/time columns
    formatted_candles = format_candle_data(candles)
    
    # Print a sample of the formatted data
    print("\nSample of formatted candle data:")
    for candle in formatted_candles[:3] if formatted_candles else []:  # Show first 3 candles
        print(candle)
    
    # Generate filename with symbol, resolution, date range and generation timestamp
    start_date_short = start_date.split()[0].replace("-", "")
    end_date_short = end_date.split()[0].replace("-", "")
    filename = generate_filename(symbol, resolution, start_date_short, end_date_short)
    
    # Save the formatted data to CSV
    success = save_to_csv(formatted_candles, filename)
    
    if success and len(formatted_candles) > 0:
        print(f"\nData Summary:")
        print(f"  Period: {start_date} to {end_date}")
        print(f"  Symbol: {symbol}, Resolution: {resolution}")
        print(f"  Total Candles: {len(formatted_candles)}")
        print(f"  First Candle: {formatted_candles[0]['date']} {formatted_candles[0]['time']}")
        print(f"  Last Candle: {formatted_candles[-1]['date']} {formatted_candles[-1]['time']}")
    
    return success

# --- Execute Data Processing ---
print("\n=== Starting Historical Data Processing ===")
process_historical_data(use_interactive=True)  # Set to False to use full year preset