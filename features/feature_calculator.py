# features/feature_calculator.py
# Added Stoch, BBW, Swings, BOS, OBs+FVG, Zones.
# Corrected previous syntax errors.

import logging
import pandas as pd
import numpy as np
try:
    import pandas_ta as ta
except ImportError:
    print("ERROR: pandas_ta library not found. Please install it (`pip install pandas_ta`)")
    import sys
    sys.exit(1)

import os
import sys
project_root_fc = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root_fc not in sys.path:
    sys.path.insert(0, project_root_fc)
try:
    from config.config_loader import get_config
except ImportError as e:
    print(f"WARNING: Could not import get_config in feature_calculator. Using defaults.")
    # Define a fallback if get_config fails
    def get_config(key, default=None): return default

logger = logging.getLogger(__name__)

# --- Helper: ATR Calculation ---
def calculate_atr(df, period=14):
    """ Calculates ATR and returns Series. """
    # Ensure required columns exist and are numeric
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        logger.error("ATR calculation requires 'high', 'low', 'close' columns.")
        return pd.Series(index=df.index, dtype=float) # Return empty/NaN series
    try:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
        atr = tr.ewm(span=period, adjust=False).mean()
        logger.debug(f"Calculated ATR({period}).")
        return atr
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return pd.Series(index=df.index, dtype=float) # Return empty/NaN series on error


# --- Helper: Swing Point Calculation ---
def add_swing_points(df_calc):
    n_swing = 5 # Default lookback
    n_swing_cfg_str = None # For error message
    try:
        n_swing_cfg_str = get_config("SMC_SWING_LOOKBACK", 5)
        n_swing = int(n_swing_cfg_str)
    except ValueError:
        logger.warning(f"Invalid SMC_SWING_LOOKBACK value '{n_swing_cfg_str}'. Using default {n_swing}.")
        n_swing = 5
    except Exception as cfg_e:
        logger.warning(f"Could not get SMC_SWING_LOOKBACK from config: {cfg_e}. Using default {n_swing}.")
        n_swing = 5

    logger.debug(f"Calculating Swing Points with n={n_swing}...")
    window = n_swing * 2 + 1
    high_col = 'high'; low_col = 'low'
    if high_col not in df_calc.columns or low_col not in df_calc.columns: logger.error("High/Low missing for swing points."); df_calc['swing_point'] = 0; return df_calc
    # Calculate rolling max/min - handle potential all-NaN windows
    rolling_high = df_calc[high_col].rolling(window, center=True, closed='both', min_periods=1).max()
    rolling_low = df_calc[low_col].rolling(window, center=True, closed='both', min_periods=1).min()
    # Check where current high/low equals the window max/min
    df_calc['is_potential_sh'] = df_calc[high_col] == rolling_high
    df_calc['is_potential_sl'] = df_calc[low_col] == rolling_low
    # Assign swing point value
    df_calc['swing_point'] = 0
    df_calc.loc[df_calc['is_potential_sh'] & ~df_calc['is_potential_sl'], 'swing_point'] = 1
    df_calc.loc[df_calc['is_potential_sl'] & ~df_calc['is_potential_sh'], 'swing_point'] = -1
    df_calc.drop(columns=['is_potential_sh', 'is_potential_sl'], inplace=True)
    logger.debug("Swing points calculated.")
    return df_calc

# --- Helper: BOS Calculation ---
def add_bos(df_calc):
    logger.debug("Calculating Break of Structure (BOS)...")
    if 'swing_point' not in df_calc.columns or 'high' not in df_calc.columns or 'low' not in df_calc.columns: logger.error("Required columns missing for BOS."); df_calc['bos'] = 0; return df_calc
    last_sh_price = np.nan; last_sl_price = np.nan; bos_values = [0] * len(df_calc)
    # Use default index if none exists for itertuples
    df_iter = df_calc.reset_index() if df_calc.index.name is None else df_calc
    for i, row in enumerate(df_iter.itertuples(index=False)): # Use index=False if index isn't needed
        try:
            # Access by attribute name
            current_high = row.high; current_low = row.low; current_swing = row.swing_point; current_bos = 0
            is_bullish_bos = False; is_bearish_bos = False
            if not pd.isna(last_sh_price) and current_high > last_sh_price: is_bullish_bos = True
            if not pd.isna(last_sl_price) and current_low < last_sl_price: is_bearish_bos = True
            if is_bullish_bos: current_bos = 1
            if is_bearish_bos: current_bos = -1
            bos_values[i] = current_bos
            if current_swing == 1: last_sh_price = current_high
            elif current_swing == -1: last_sl_price = current_low
        except AttributeError as ae: logger.error(f"Attribute error accessing row data at index {i} for BOS: {ae}. Row: {row}"); bos_values[i] = 0; continue
        except Exception as e: logger.error(f"General error at index {i} for BOS: {e}"); bos_values[i] = 0; continue
    df_calc['bos'] = bos_values; logger.debug("BOS calculation finished.")
    return df_calc

# --- Helper: Order Block Calculation (with FVG check) ---
def add_order_blocks(df_calc, digits=5):
    """ Identifies simple OBs (candle before BOS) and checks for associated FVG. """
    logger.debug("Calculating Order Blocks and checking for FVGs...")
    required_cols = ['bos', 'open', 'high', 'low', 'close']
    if not all(col in df_calc.columns for col in required_cols):
        logger.error(f"Required columns missing for OB/FVG calc: {required_cols}")
        df_calc['bull_ob_high']=np.nan; df_calc['bull_ob_low']=np.nan
        df_calc['bear_ob_high']=np.nan; df_calc['bear_ob_low']=np.nan
        df_calc['bull_ob_has_fvg']=False; df_calc['bear_ob_has_fvg']=False
        return df_calc

    # Initialize columns
    df_calc['bull_ob_h_new'] = np.nan; df_calc['bull_ob_l_new'] = np.nan; df_calc['bull_ob_has_fvg_new'] = False
    df_calc['bear_ob_h_new'] = np.nan; df_calc['bear_ob_l_new'] = np.nan; df_calc['bear_ob_has_fvg_new'] = False

    # Create shifted columns for FVG check (vectorized approach attempt) - More efficient than loop lookup
    df_calc['low_plus_2'] = df_calc['low'].shift(-2)
    df_calc['high_plus_2'] = df_calc['high'].shift(-2)
    df_calc['high_prev'] = df_calc['high'].shift(1)
    df_calc['low_prev'] = df_calc['low'].shift(1)
    df_calc['is_prev_bearish'] = df_calc['close'].shift(1) < df_calc['open'].shift(1)
    df_calc['is_prev_bullish'] = df_calc['close'].shift(1) > df_calc['open'].shift(1)

    # Conditions for setting OB based on BOS and previous candle type
    bull_ob_condition = (df_calc['bos'] == 1) & df_calc['is_prev_bearish']
    bear_ob_condition = (df_calc['bos'] == -1) & df_calc['is_prev_bullish']

    # Conditions for FVG
    bull_fvg_condition = bull_ob_condition & (df_calc['low_plus_2'] > df_calc['high_prev'])
    bear_fvg_condition = bear_ob_condition & (df_calc['high_plus_2'] < df_calc['low_prev'])

    # Set OB levels and FVG flag where conditions met
    df_calc.loc[bull_ob_condition, 'bull_ob_h_new'] = df_calc['high_prev']
    df_calc.loc[bull_ob_condition, 'bull_ob_l_new'] = df_calc['low_prev']
    df_calc.loc[bull_fvg_condition, 'bull_ob_has_fvg_new'] = True

    df_calc.loc[bear_ob_condition, 'bear_ob_h_new'] = df_calc['high_prev']
    df_calc.loc[bear_ob_condition, 'bear_ob_l_new'] = df_calc['low_prev']
    df_calc.loc[bear_fvg_condition, 'bear_ob_has_fvg_new'] = True

    # Invalidate competing OB type when one is confirmed
    df_calc.loc[bull_ob_condition, ['bear_ob_h_new', 'bear_ob_l_new', 'bear_ob_has_fvg_new']] = np.nan, np.nan, False
    df_calc.loc[bear_ob_condition, ['bull_ob_h_new', 'bull_ob_l_new', 'bull_ob_has_fvg_new']] = np.nan, np.nan, False

    # Forward fill OB levels and FVG status
    ob_cols_new = ['bull_ob_h_new', 'bull_ob_l_new', 'bull_ob_has_fvg_new',
                   'bear_ob_h_new', 'bear_ob_l_new', 'bear_ob_has_fvg_new']
    df_calc[ob_cols_new] = df_calc[ob_cols_new].ffill()

    # Simple Mitigation Logic
    bull_mitigated = df_calc['low'] <= df_calc['bull_ob_h_new'];
    df_calc.loc[bull_mitigated, ['bull_ob_h_new', 'bull_ob_l_new', 'bull_ob_has_fvg_new']] = np.nan, np.nan, False
    bear_mitigated = df_calc['high'] >= df_calc['bear_ob_l_new'];
    df_calc.loc[bear_mitigated, ['bear_ob_h_new', 'bear_ob_l_new', 'bear_ob_has_fvg_new']] = np.nan, np.nan, False
    # Forward fill again after mitigation
    df_calc[ob_cols_new] = df_calc[ob_cols_new].ffill()

    # Rename final columns and set final types
    df_calc.rename(columns={ 'bull_ob_h_new': 'bull_ob_high', 'bull_ob_l_new': 'bull_ob_low', 'bull_ob_has_fvg_new': 'bull_ob_has_fvg', 'bear_ob_h_new': 'bear_ob_high', 'bear_ob_l_new': 'bear_ob_low', 'bear_ob_has_fvg_new': 'bear_ob_has_fvg'}, inplace=True)
    df_calc['bull_ob_has_fvg'] = df_calc['bull_ob_has_fvg'].fillna(False).astype(bool)
    df_calc['bear_ob_has_fvg'] = df_calc['bear_ob_has_fvg'].fillna(False).astype(bool)

    # Drop helper columns
    df_calc.drop(columns=['low_plus_2', 'high_plus_2', 'high_prev', 'low_prev', 'is_prev_bearish', 'is_prev_bullish'], inplace=True, errors='ignore')

    logger.debug("Order Block + FVG check calculation finished.")
    return df_calc

# --- Helper: Horizontal Zone Detection ---
def add_horizontal_zones(df_calc, atr_period=14, window_size=20, min_rejections=2, tolerance_pct=0.001, min_zone_width_pct=0.003, max_zone_width_atr_mult=1.5):
    """ Detects horizontal consolidation zones based on refined rules. """
    logger.debug(f"Calculating Horizontal Zones (N={window_size}, Rej={min_rejections}, Tol={tolerance_pct*100:.3f}%)...")
    high_col = 'high'; low_col = 'low'; close_col = 'close'; atr_col = f'atrr_{atr_period}'
    if not all(c in df_calc.columns for c in [high_col, low_col, close_col, atr_col]): logger.error(f"Required columns missing for zone detection."); df_calc['zone_top'] = np.nan; df_calc['zone_bottom'] = np.nan; return df_calc

    atr = df_calc[atr_col]
    # Calculate rolling boundaries looking back N bars (use closed='left' or adjust window+shift)
    rolling_high = df_calc[high_col].rolling(window=window_size, closed='left').max()
    rolling_low = df_calc[low_col].rolling(window=window_size, closed='left').min()
    zone_width = rolling_high - rolling_low

    # Apply width filters
    max_width_limit = max_zone_width_atr_mult * atr; min_width_limit = min_zone_width_pct * df_calc[close_col]
    passes_max_width = zone_width <= max_width_limit; passes_min_width = zone_width > min_width_limit
    passes_width_filter = passes_max_width & passes_min_width

    # Initialize output columns
    df_calc['zone_top'] = np.nan; df_calc['zone_bottom'] = np.nan
    potential_zone_indices = df_calc.index[passes_width_filter & passes_width_filter.shift(-1, fill_value=False)] # Check end of potential zone

    logger.debug(f"Found {len(potential_zone_indices)} potential zones based on width criteria.")
    valid_zone_count = 0

    # Iterate only through potential zone end points to check rejections in the preceding window
    for idx in potential_zone_indices:
        try:
            window_end_loc = df_calc.index.get_loc(idx); window_start_loc = window_end_loc - window_size
            if window_start_loc < 0: continue
            segment = df_calc.iloc[window_start_loc : window_end_loc]
            if segment.empty: continue

            zone_top_val = segment[high_col].max(); zone_bottom_val = segment[low_col].min()
            price_for_tol = df_calc.loc[idx, close_col] # Use price at detection point for tolerance
            if pd.isna(price_for_tol) or price_for_tol == 0: tolerance_val = 0.0001
            else: tolerance_val = price_for_tol * tolerance_pct

            top_rejects = (abs(segment[high_col] - zone_top_val) <= tolerance_val).sum()
            bottom_rejects = (abs(segment[low_col] - zone_bottom_val) <= tolerance_val).sum()

            if top_rejects >= min_rejections and bottom_rejects >= min_rejections:
                # Mark zone on the bar *after* the window where it was detected
                df_calc.loc[idx, 'zone_top'] = zone_top_val
                df_calc.loc[idx, 'zone_bottom'] = zone_bottom_val
                valid_zone_count += 1
        except Exception as e: logger.warning(f"Error processing potential zone ending before {idx}: {e}"); continue

    logger.debug(f"Confirmed {valid_zone_count} zones based on rejections.")
    zone_cols = ['zone_top', 'zone_bottom']; df_calc[zone_cols] = df_calc[zone_cols].ffill() # Forward fill zones
    logger.debug("Horizontal Zone detection finished.")
    return df_calc


# --- Main Feature Calculation Function ---
def calculate_features(df: pd.DataFrame) -> pd.DataFrame | None:
    """ Calculates features including TA indicators, Swings, BOS, OBs (with FVG), and Horizontal Zones. """
    logger.debug(f"Calculating all features for DataFrame with shape {df.shape}")
    required_cols_orig_case = ['open', 'high', 'low', 'close']
    df_calc = df.copy()
    df_calc.columns = [col.lower() for col in df_calc.columns]
    required_cols = [col.lower() for col in required_cols_orig_case]
    if not all(col in df_calc.columns for col in required_cols): logger.error(f"Input missing required columns."); return None

    try:
        open_col, high_col, low_col, close_col = 'open', 'high', 'low', 'close'

        # 1. Basic Price Features & ATR
        df_calc['returns'] = df_calc[close_col].pct_change()
        df_calc['range'] = df_calc[high_col] - df_calc[low_col]
        range_denom = df_calc['range'].replace(0, np.nan); df_calc['close_ratio'] = (df_calc[close_col] - df_calc[low_col]) / range_denom; df_calc['close_ratio'] = df_calc['close_ratio'].fillna(0.5)

        # --- Get ATR Period --- <<< CORRECTED SYNTAX >>>
        atr_period_for_calc = 14; atr_period_from_config = None
        try:
            atr_period_from_config = get_config("SL_ATR_PERIOD", 14)
            if atr_period_from_config is not None: atr_period_for_calc = int(atr_period_from_config)
        except ValueError: logger.warning(f"Invalid SL_ATR_PERIOD '{atr_period_from_config}'. Using default {atr_period_for_calc}."); atr_period_for_calc = 14
        except Exception as e: logger.warning(f"Error getting SL_ATR_PERIOD: {e}. Using default {atr_period_for_calc}."); atr_period_for_calc = 14
        logger.debug(f"Using ATR Period for calculation: {atr_period_for_calc}")

        # Calculate ATR using helper and assign the correct column name
        atr_col_name = f'atrr_{atr_period_for_calc}' # e.g., atrr_14
        df_calc[atr_col_name] = calculate_atr(df_calc, period=atr_period_for_calc)
        logger.debug(f"Calculated {atr_col_name} using helper.")

        # 2. Standard Indicators
        logger.debug("Calculating standard indicators (RSI, EMA)..."); df_calc.ta.rsi(close=close_col, length=14, append=True, col_names='rsi_14'); df_calc.ta.ema(close=close_col, length=20, append=True, col_names='ema_20'); df_calc.ta.ema(close=close_col, length=50, append=True, col_names='ema_50')
        logger.debug("Calculating Bollinger Bands..."); df_calc.ta.bbands(close=close_col, length=20, std=2, append=True)
        bb_middle_col = 'bbm_20_2.0'; bb_upper_col = 'bbu_20_2.0'; bb_lower_col = 'bbl_20_2.0'
        if all(c in df_calc.columns for c in [bb_upper_col, bb_lower_col, bb_middle_col]):
             bb_middle_denom = df_calc[bb_middle_col].replace(0, np.nan); bbw_col_name = 'bbw_20_2.0'; df_calc[bbw_col_name] = (df_calc[bb_upper_col] - df_calc[bb_lower_col]) / bb_middle_denom; df_calc.loc[:, bbw_col_name] = df_calc[bbw_col_name].fillna(0); logger.debug("Calculated BBW.")
        else: logger.warning(f"Could not calculate BBW..."); df_calc['bbw_20_2.0'] = 0.0
        logger.debug("Calculating MACD..."); df_calc.ta.macd(close=close_col, fast=12, slow=26, signal=9, append=True)
        logger.debug("Calculating Stochastic Oscillator..."); df_calc.ta.stoch(high=high_col, low=low_col, close=close_col, k=14, d=3, smooth_k=3, append=True)

        # 3. SMC Features
        df_calc = add_swing_points(df_calc)
        df_calc = add_bos(df_calc)
        df_calc = add_order_blocks(df_calc) # Now includes FVG check

        # 4. Horizontal Zone Detection
        df_calc = add_horizontal_zones(df_calc, atr_period=atr_period_for_calc)

        # 5. Final processing
        df_calc.columns = [col.lower() for col in df_calc.columns] # Ensure final columns are lowercase

        initial_len = len(df_calc)
        # --- DropNA Subset Logic ---
        dropna_subset = ['ema_50', f'atrr_{atr_period_for_calc}', 'macds_12_26_9'] # Use correct slow EMA period (50)
        dropna_subset = [col.lower() for col in dropna_subset] # Ensure list is lowercase
        actual_subset = [col for col in dropna_subset if col in df_calc.columns]
        if actual_subset:
            logger.debug(f"Attempting dropna using subset: {actual_subset}")
            df_calc.dropna(subset=actual_subset, inplace=True)
        else:
            logger.warning("Could not find any columns in subset for dropna, skipping dropna.")
        # --- End DropNA Subset ---

        final_len = len(df_calc)
        if initial_len > final_len: logger.debug(f"Dropped {initial_len - final_len} total rows due to NaNs in essential subset.")
        elif final_len == 0 and initial_len > 0 : logger.warning("DataFrame became empty after dropna based on subset!") # Check if empty after dropna

        logger.info(f"Features calculated successfully. Output shape: {df_calc.shape}")
        if df_calc.empty:
             logger.error("Returning empty DataFrame from calculate_features!")
        return df_calc

    except Exception as e:
        logger.exception(f"An error occurred during feature calculation: {e}")
        return None

# --- Test Block ---
# (Remains the same)
if __name__ == '__main__':
    import os
    logging.basicConfig(level=logging.INFO)
    logger.info("--- Testing Feature Calculator Standalone (incl. Zones) ---")
    sample_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'EURUSD', 'EURUSD_M5_2024-01-01_to_2024-03-31.csv')
    if os.path.exists(sample_data_path):
          df_raw = pd.read_csv(sample_data_path, parse_dates=['time'], index_col='time')
          logger.info(f"Loaded sample data shape: {df_raw.shape}")
          df_features = calculate_features(df_raw.head(1000))
          if df_features is not None and not df_features.empty: # Check if empty
               logger.info("Features calculated for sample data.")
               print(df_features.tail())
               print("\nColumns:")
               print(df_features.columns.tolist())
               zone_cols = ['zone_top', 'zone_bottom']
               fvg_cols = ['bull_ob_has_fvg', 'bear_ob_has_fvg'] # Check FVG cols too
               if all(c in df_features.columns for c in fvg_cols):
                   print("\nFVG Flag Counts (Sample):")
                   print(f" Bullish OBs with FVG: {df_features['bull_ob_has_fvg'].sum()}")
                   print(f" Bearish OBs with FVG: {df_features['bear_ob_has_fvg'].sum()}")
               if all(c in df_features.columns for c in zone_cols):
                    print("\nZone Columns (Non-NaN Counts):"); print(df_features[zone_cols].notna().sum())
                    print("\nLast few zone values:"); print(df_features[zone_cols].dropna().tail())
               logger.info(f"Final shape: {df_features.shape}")
          else: logger.error("Feature calculation failed or returned empty for sample data.")
    else: logger.warning(f"Sample data file not found at {sample_data_path}. Cannot run standalone test.")