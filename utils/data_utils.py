# utils/data_utils.py
# Contains functions for loading, preprocessing, and preparing sequence data.

import logging
import os
import sys
import pandas as pd
import numpy as np

# Setup logger for this module
logger = logging.getLogger(__name__)
# Basic handler if run standalone or not configured elsewhere
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # logger.warning("Basic logging handler added in data_utils.") # Optional warning

# --- Data Loading ---

def load_data(pair: str, timeframe_str: str, base_data_dir: str) -> pd.DataFrame | None:
    """
    Loads historical price data from a CSV file based on pair and timeframe string.

    Args:
        pair (str): Currency pair symbol (e.g., "EURUSD").
        timeframe_str (str): Timeframe string (e.g., "M5", "H1").
        base_data_dir (str): The root directory containing pair subdirectories.
                             Example: "G:/Alpha1.1/data"

    Returns:
        pd.DataFrame | None: DataFrame with 'time', 'open', 'high', 'low', 'close',
                              'tick_volume' (if available), or None if loading fails.
    """
    # Construct expected filename format: PAIR_TF.csv inside PAIR subdirectory
    # Assumes data files follow a consistent naming pattern. Adjust if necessary.
    filename = os.path.join(base_data_dir, pair, f"{pair}_{timeframe_str}.csv")
    logger.debug(f"Attempting to load data from: {filename}")

    try:
        df = pd.read_csv(filename)

        # --- Essential Column Checks ---
        if 'time' not in df.columns:
            logger.error(f"'time' column not found in {filename}")
            return None
        # Convert time column, handle potential errors
        try:
            df['time'] = pd.to_datetime(df['time'])
        except Exception as time_e:
            logger.error(f"Failed to parse 'time' column in {filename}: {time_e}")
            return None

        required_price_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_price_cols):
            logger.error(f"Missing required price columns {required_price_cols} in {filename}. Found: {df.columns.tolist()}")
            return None

        # Ensure price columns are numeric
        for col in required_price_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Drop rows if essential price data became NaN after conversion
        df.dropna(subset=required_price_cols, inplace=True)
        if df.empty:
            logger.warning(f"No valid price data remained after to_numeric conversion in {filename}")
            return None

        # Handle optional 'tick_volume' conversion
        if 'tick_volume' in df.columns:
             df['tick_volume'] = pd.to_numeric(df['tick_volume'], errors='coerce') # Coerce keeps column even if conversion fails for some rows
        else:
             logger.warning(f"'tick_volume' column not found in {filename}. Proceeding without it.")


        logger.info(f"Successfully loaded data from {filename}. Shape: {df.shape}")
        return df

    except FileNotFoundError:
        logger.error(f"Data file not found: {filename}")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred loading {filename}: {e}")
        return None

# --- Data Preprocessing ---

def preprocess_data(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Basic preprocessing for loaded data (e.g., handling missing values).

    Args:
        df (pd.DataFrame | None): Input DataFrame, potentially from load_data.

    Returns:
        pd.DataFrame | None: Preprocessed DataFrame, or None if input is None.
    """
    if df is None:
        logger.warning("preprocess_data received None input.")
        return None

    logger.debug(f"Preprocessing data with initial shape: {df.shape}")

    # Example: Forward fill missing values in typical OHLCV columns
    # Only fill columns that actually exist in the DataFrame
    cols_to_ffill = ['open', 'high', 'low', 'close', 'tick_volume']
    existing_cols_to_fill = [col for col in cols_to_ffill if col in df.columns]

    if existing_cols_to_fill:
        # Count NaNs before filling for logging purposes
        nans_before = df[existing_cols_to_fill].isnull().sum().sum()
        if nans_before > 0:
            df[existing_cols_to_fill] = df[existing_cols_to_fill].ffill()
            nans_after = df[existing_cols_to_fill].isnull().sum().sum()
            logger.debug(f"Forward-filled NaNs. Before: {nans_before}, After: {nans_after}")
        else:
            logger.debug("No NaNs found in specified columns to forward-fill.")

        # Drop any remaining NaNs (e.g., at the very beginning of the file)
        # Only check critical price columns for dropping
        critical_cols = ['open', 'high', 'low', 'close']
        initial_len = len(df)
        df.dropna(subset=critical_cols, inplace=True)
        len_after_drop = len(df)
        if initial_len != len_after_drop:
            logger.debug(f"Dropped {initial_len - len_after_drop} rows with NaNs in critical columns.")

    # Add other preprocessing steps here if needed (e.g., outlier detection/capping)

    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    return df


# --- Sequence Preparation for Deep Learning Models ---

def create_sequences(df: pd.DataFrame, target_column: str, sequence_length: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Transforms a DataFrame of features and a target column into sequences
    suitable for LSTM/CNN models.

    Args:
        df (pd.DataFrame): DataFrame containing features and the target column.
                           Index should ideally be time-based but isn't used directly here.
        target_column (str): The name of the column containing the target variable (e.g., 'target').
        sequence_length (int): The number of time steps (bars) in each sequence (lookback period).

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]:
          - X: NumPy array of sequences with shape (num_samples, sequence_length, num_features).
          - y: NumPy array of targets with shape (num_samples,).
          - Returns (None, None) if errors occur.
    """
    logger.debug(f"Creating sequences with length {sequence_length} from data shape {df.shape}")
    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in DataFrame columns: {df.columns.tolist()}")
        return None, None
    if sequence_length <= 0:
        logger.error(f"Sequence length must be positive, got {sequence_length}")
        return None, None
    if len(df) <= sequence_length:
        # Changed check slightly: Need len > seq_len to create at least one sequence ending at last point
        logger.error(f"Data length ({len(df)}) must be greater than sequence length ({sequence_length}) to create valid sequences.")
        return None, None

    # Separate features and target
    feature_columns = df.columns.drop(target_column)
    X_data = df[feature_columns]
    y_data = df[target_column]

    # Convert to NumPy arrays for efficiency
    # Check for non-numeric types BEFORE converting to numpy
    non_numeric_cols = X_data.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols:
        logger.warning(f"Non-numeric columns found in features, excluding them: {non_numeric_cols}")
        X_data = X_data.select_dtypes(include=np.number)
        if X_data.empty:
             logger.error("No numeric feature columns remain after excluding non-numeric types.")
             return None, None
        logger.debug(f"Using numeric feature columns: {X_data.columns.tolist()}")


    X_values = X_data.values
    y_values = y_data.values

    X_sequences, y_sequences = [], []

    # Iterate through the data to create overlapping sequences
    # The loop should go up to the point where the last sequence ENDS at the last available data point
    # The target for sequence X[i : i+seq_len] is y[i+seq_len - 1]
    for i in range(len(X_values) - sequence_length + 1):
        seq_x = X_values[i : i + sequence_length]
        seq_y = y_values[i + sequence_length - 1] # Target corresponds to the last element in the sequence window

        X_sequences.append(seq_x)
        y_sequences.append(seq_y)

    if not X_sequences: # Should not happen if checks above pass, but good safety check
        logger.error("No sequences were generated.")
        return None, None

    X_out = np.array(X_sequences)
    y_out = np.array(y_sequences)

    logger.info(f"Sequence creation complete. X shape: {X_out.shape}, y shape: {y_out.shape}")
    # Example: Input (1000 rows, 15 features), seq_len=50 -> Output X:(951, 50, 15), y:(951,)
    return X_out, y_out


# --- Example Usage / Test Block ---
if __name__ == '__main__':
    # Setup basic logging for testing this module directly
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    logger.info("--- Testing data_utils.py Standalone ---")

    # Define test parameters
    test_pair = "EURUSD"
    test_tf = "M5"
    # Construct path relative to this file's location (utils/) -> ../data
    test_base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    logger.info(f"Using base data directory: {test_base_dir}")

    # Test load_data
    logger.info(f"\n--- Testing load_data ({test_pair} {test_tf}) ---")
    df_loaded = load_data(test_pair, test_tf, test_base_dir)

    if df_loaded is not None:
        logger.info("load_data successful.")
        print(df_loaded.head())
        print(df_loaded.info())

        # Test preprocess_data
        logger.info("\n--- Testing preprocess_data ---")
        df_preprocessed = preprocess_data(df_loaded.copy()) # Use copy

        if df_preprocessed is not None:
            logger.info("preprocess_data successful.")
            print(df_preprocessed.head()) # See if NaNs at start are handled
            print(df_preprocessed.info())

            # Test create_sequences
            logger.info("\n--- Testing create_sequences ---")
            # Add a dummy target column for testing
            df_preprocessed['target'] = np.random.randint(0, 2, size=len(df_preprocessed))
            test_seq_len = 60 # Example sequence length

            X_seq, y_seq = create_sequences(df_preprocessed, 'target', test_seq_len)

            if X_seq is not None and y_seq is not None:
                logger.info("create_sequences successful.")
                logger.info(f"Generated X shape: {X_seq.shape}") # (samples, seq_len, features)
                logger.info(f"Generated y shape: {y_seq.shape}") # (samples,)
                # Optionally print a sample sequence/target
                # print("Sample X sequence (first one):\n", X_seq[0])
                # print("Sample y target (first one):", y_seq[0])
            else:
                logger.error("create_sequences failed.")
        else:
            logger.error("preprocess_data failed.")
    else:
        logger.error("load_data failed. Cannot proceed with further tests.")

    logger.info("\n--- data_utils.py Standalone Test Complete ---")