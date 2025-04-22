# G:/AlphaLive/trading_engine/engine.py
# Modified for H1 Trend, Risk Mgmt integration, Time Filter, Dupe Trade Filter
# Includes fix for MIN_BARS_FOR_CALCULATION NameError
# Includes fix for FutureWarning ffill inplace
# CORRECTED: Timezone handling before pd.merge_asof

import time
import logging
import pandas as pd
import numpy as np
import os
import sys
import MetaTrader5 as mt5

# --- Setup Path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Add handler if none exists
    handler = logging.StreamHandler(); formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter); logger.addHandler(handler); logger.setLevel(logging.INFO)
    logger.info("Basic logging handler added in engine.")


# --- Import Components ---
try:
    from trading_engine.mt5_connector import (
        get_latest_bars, get_symbol_info, get_account_info,
        get_tick_info, is_mt5_connected, TIMEFRAME_MAP,
        get_open_positions # <-- Import new function
    )
    from models.regime_predictor import load_xgboost_binary_model, make_xgboost_binary_prediction
    # IMPORTANT: Ensure this path is correct for your feature calculator
    from features.feature_calculator import calculate_features
    from risk_management.risk_manager import calculate_trade_parameters
    from trading_engine.trade_executor import execute_trade # Modified executor
    from shared_state import get_state, set_state
    from config.config_loader import get_config
    # Use pandas_ta directly for H1 EMA calculation here
    import pandas_ta as ta

    # Optional: Import telegram alert function if used
    try:
        # Adjust path if needed (e.g., if bot_handler is directly in telegram_interface)
        from telegram_interface.bot_handler import schedule_telegram_message
        # Create a wrapper for consistent alert calls
        def send_telegram_alert(message):
             chat_id = get_config("ALERT_CHAT_ID")
             if chat_id:
                 try:
                     # Ensure chat_id is integer
                     schedule_telegram_message(int(chat_id), message)
                 except Exception as tg_e:
                     logger.error(f"Failed to schedule telegram alert: {tg_e}")
             else:
                 logger.warning("Telegram ALERT_CHAT_ID not configured.")

    except ImportError:
        # Dummy placeholder if not implemented or optional
        logger.warning("Telegram interface not found. Alerts will only be logged.")
        def send_telegram_alert(message):
            logger.info(f"TELEGRAM_ALERT_PLACEHOLDER: {message}")

except ImportError as e:
    logger.critical(f"Engine failed import components: {e}", exc_info=True); sys.exit(1)

# --- Engine Configuration ---
try:
    SYMBOLS_TO_TRADE = get_config('SYMBOLS_TO_TRADE', 'EURUSD').split(',') # Default to EURUSD if missing
    PRIMARY_TIMEFRAME_STR = get_config('MT5_TIMEFRAME', 'M15').upper() # TF for main features/signals
    BARS_FOR_PRIMARY_FEATURES = int(get_config('BARS_FOR_FEATURES', 250)) # How many M15 bars
    MIN_BARS_FOR_CALCULATION = int(get_config('MIN_BARS_FOR_CALCULATION', 200)) # Min bars for calc logic

    # H1 Trend Config
    CALCULATE_H1_TREND = get_config('CALCULATE_H1_TREND', 'True').lower() == 'true' # Make it configurable
    H1_TIMEFRAME_STR = "H1"
    H1_MT5_TIMEFRAME = TIMEFRAME_MAP.get(H1_TIMEFRAME_STR, mt5.TIMEFRAME_H1) # Get MT5 constant
    H1_EMA_PERIOD = int(get_config('H1_EMA_PERIOD', 50)) # Configurable H1 EMA period
    BARS_FOR_H1_EMA = H1_EMA_PERIOD + int(get_config('H1_EMA_BUFFER', 50)) # Fetch H1 EMA period + buffer

    # Time Filter Config
    ENTRY_START_HOUR_UTC = int(get_config('ENTRY_START_HOUR_UTC', 0)) # Default 0 (start of day)
    ENTRY_END_HOUR_UTC = int(get_config('ENTRY_END_HOUR_UTC', 23)) # Default 23 (end of day)
    logger.info(f"Time Filter Active: Allowing entries between {ENTRY_START_HOUR_UTC:02d}:00 and {ENTRY_END_HOUR_UTC:02d}:59 UTC")

    LOOP_SLEEP_SECONDS = int(get_config('LOOP_SLEEP_SECONDS', 60)) # Check frequency
    CONFIDENCE_THRESHOLD = float(get_config("MIN_PREDICTION_CONFIDENCE", 0.80))
    ATR_PERIOD = int(get_config("SL_ATR_PERIOD", 14)) # Ensure consistent ATR period for SL calc later
    ALERT_ON_EXECUTION = get_config('ALERT_ON_EXECUTION', 'True').lower() == 'true'
    ALERT_ON_FAILURE = get_config('ALERT_ON_FAILURE', 'True').lower() == 'true'

except ValueError as ve: logger.critical(f"Invalid numeric config: {ve}. Exiting."); sys.exit(1)
except Exception as cfg_e: logger.critical(f"Error reading engine config: {cfg_e}. Exiting."); sys.exit(1)

if PRIMARY_TIMEFRAME_STR not in TIMEFRAME_MAP: logger.critical(f"Invalid primary timeframe '{PRIMARY_TIMEFRAME_STR}'. Exiting."); sys.exit(1)


# --- Load Model ---
model = None
trained_feature_list_lower = []
MODEL_EXPECTS_H1_TREND = False # Default
try:
    MODEL_PATH_CONFIG = get_config("XGB_MODEL_PATH")
    if not MODEL_PATH_CONFIG: raise ValueError("XGB_MODEL_PATH not set in config.")
    # Construct absolute path robustly
    if not os.path.isabs(MODEL_PATH_CONFIG):
        MODEL_FULL_PATH = os.path.join(project_root, MODEL_PATH_CONFIG)
    else:
        MODEL_FULL_PATH = MODEL_PATH_CONFIG

    logger.info(f"Engine loading model from: {MODEL_FULL_PATH}")
    if not os.path.exists(MODEL_FULL_PATH):
        raise FileNotFoundError(f"Model file not found at specified path: {MODEL_FULL_PATH}")

    # Pass full path directly to loader
    model, trained_feature_list = load_xgboost_binary_model(MODEL_FULL_PATH)
    if model is None or trained_feature_list is None:
        # Error logged within load_xgboost_binary_model
        raise ValueError(f"Failed to load XGBoost model/features list from {MODEL_FULL_PATH}")
    trained_feature_list_lower = [f.lower() for f in trained_feature_list]
    MODEL_EXPECTS_H1_TREND = 'h1_trend' in trained_feature_list_lower
    logger.info(f"Model loaded. Expects {len(trained_feature_list_lower)} features. Expects 'h1_trend': {MODEL_EXPECTS_H1_TREND}")
    # logger.debug(f"DEBUG: Full expected feature list: {trained_feature_list_lower}")
    if MODEL_EXPECTS_H1_TREND and not CALCULATE_H1_TREND: logger.warning("Model expects 'h1_trend' but CALCULATE_H1_TREND is disabled!")

except Exception as model_e: logger.critical(f"Model loading failed: {model_e}. Exiting.", exc_info=True); sys.exit(1)


# --- Main Trading Loop ---
def run_trading_loop():
    global model, trained_feature_list_lower, MODEL_EXPECTS_H1_TREND

    logger.info(f"🚀 Starting live trading loop for {SYMBOLS_TO_TRADE} on {PRIMARY_TIMEFRAME_STR}...")
    atr_col_name_check = f'atrr_{ATR_PERIOD}' # Expected ATR column name from feature calculator

    while True:
        # --- Check Control Flags & Connection ---
        if not get_state("is_running", True): logger.info("Stop request received, exiting loop."); break
        if get_state("is_paused", False): logger.info("Trading loop paused..."); time.sleep(max(10, int(LOOP_SLEEP_SECONDS / 2))); continue
        if not is_mt5_connected(): logger.error("MT5 Connection lost! Waiting..."); time.sleep(30); continue

        loop_start_time = time.time()

        # --- Iterate Symbols ---
        for symbol in SYMBOLS_TO_TRADE:
            symbol_start_time = time.time()
            logger.info(f"--- Processing {symbol} ---")

            try:
                # === Step 1a: Get Primary TF Data ===
                logger.debug(f"Fetching {BARS_FOR_PRIMARY_FEATURES} bars ({PRIMARY_TIMEFRAME_STR}) for {symbol}...")
                primary_bars_df = get_latest_bars(symbol, PRIMARY_TIMEFRAME_STR, BARS_FOR_PRIMARY_FEATURES)
                # Check minimum bars AFTER fetching, before feature calculation
                if primary_bars_df is None or primary_bars_df.empty: logger.warning(f"Could not fetch {PRIMARY_TIMEFRAME_STR} bars {symbol}. Skip."); continue
                if len(primary_bars_df) < MIN_BARS_FOR_CALCULATION: logger.warning(f"Insufficient {PRIMARY_TIMEFRAME_STR} bars ({len(primary_bars_df)}) {symbol}. Need {MIN_BARS_FOR_CALCULATION}. Skip."); continue

                # === Step 1b: Get H1 Data (If Needed) ===
                h1_ema_series = None
                if CALCULATE_H1_TREND or MODEL_EXPECTS_H1_TREND:
                    logger.debug(f"Fetching {BARS_FOR_H1_EMA} bars ({H1_TIMEFRAME_STR}) for {symbol}...")
                    h1_bars_df = get_latest_bars(symbol, H1_TIMEFRAME_STR, BARS_FOR_H1_EMA)
                    if h1_bars_df is None or h1_bars_df.empty or len(h1_bars_df) < H1_EMA_PERIOD: logger.warning(f"Insufficient H1 bars ({len(h1_bars_df) if h1_bars_df is not None else 'None'}) for H1 EMA {symbol}.")
                    else:
                        # === Step 1c: Calculate H1 EMA ===
                        logger.debug(f"Calculating H1 EMA({H1_EMA_PERIOD}) for {symbol}...")
                        try:
                            h1_bars_df.columns = [col.lower() for col in h1_bars_df.columns]
                            if 'close' not in h1_bars_df.columns: raise ValueError("H1 bars missing 'close'")
                            h1_ema_col = f'h1_ema_{H1_EMA_PERIOD}'
                            # Use pandas_ta directly here
                            h1_bars_df.ta.ema(close='close', length=H1_EMA_PERIOD, append=True, col_names=h1_ema_col)
                            # Select only the EMA column and drop NaNs
                            h1_ema_series = h1_bars_df[[h1_ema_col]].dropna()
                            if h1_ema_series.empty: logger.warning(f"H1 EMA calc resulted in empty Series {symbol}.")
                            else: logger.debug(f"H1 EMA calculated {symbol}.")
                        except Exception as h1_calc_e: logger.error(f"Failed H1 EMA calc {symbol}: {h1_calc_e}"); h1_ema_series = None

                # === Step 2: Calculate Primary TF Features ===
                logger.debug(f"Calculating {PRIMARY_TIMEFRAME_STR} features for {symbol}...")
                # Assumes feature_calculator handles its own data needs and returns DataFrame with DatetimeIndex
                features_primary_df = calculate_features(primary_bars_df.copy()) # Pass a copy
                if features_primary_df is None or features_primary_df.empty: logger.warning(f"{PRIMARY_TIMEFRAME_STR} Feature calc failed {symbol}. Skip."); continue

                # === Step 3: Merge H1 EMA and Calculate h1_trend ===
                features_final_df = features_primary_df # Start with primary
                h1_ema_col = f'h1_ema_{H1_EMA_PERIOD}' # Define here for use below

                if CALCULATE_H1_TREND and h1_ema_series is not None and not h1_ema_series.empty:
                    logger.debug(f"Merging H1 EMA and calculating h1_trend for {symbol}...")
                    try:
                        # --- >>> START: Standardize Timezones before merging <<< ---
                        # Ensure features_final_df (from feature_calculator) is UTC aware
                        if features_final_df.index.tz is None:
                            logger.debug(f"Localizing features_final_df index to UTC for {symbol}")
                            features_final_df.index = features_final_df.index.tz_localize('UTC')
                        elif str(features_final_df.index.tz) != 'UTC':
                             logger.warning(f"Converting features_final_df index from {features_final_df.index.tz} to UTC for {symbol}")
                             features_final_df.index = features_final_df.index.tz_convert('UTC')

                        # Ensure h1_ema_series (from mt5_connector) is UTC aware (should be, but check)
                        if h1_ema_series.index.tz is None:
                             logger.warning(f"Localizing h1_ema_series index to UTC for {symbol} (This was unexpected!)")
                             h1_ema_series.index = h1_ema_series.index.tz_localize('UTC')
                        elif str(h1_ema_series.index.tz) != 'UTC':
                              logger.warning(f"Converting h1_ema_series index from {h1_ema_series.index.tz} to UTC for {symbol}")
                              h1_ema_series.index = h1_ema_series.index.tz_convert('UTC')
                        # --- >>> END: Standardize Timezones before merging <<< ---

                        # Now perform the merge with UTC-aware indices (ensure they are sorted)
                        features_merged_df = pd.merge_asof(
                            features_final_df.sort_index(),
                            h1_ema_series[[h1_ema_col]].sort_index(), # Ensure h1_ema_series is also sorted
                            left_index=True,
                            right_index=True,
                            direction='backward',
                            tolerance=pd.Timedelta('65m') # Adjust tolerance if needed based on TFs
                        )

                        # Fix FutureWarning: Use direct assignment instead of inplace=True
                        # Forward fill the merged EMA column
                        features_merged_df[h1_ema_col] = features_merged_df[h1_ema_col].ffill() # Assign back

                        # Check required columns exist before calculating trend
                        if 'close' not in features_merged_df.columns: raise ValueError("'close' missing after merge");
                        if h1_ema_col not in features_merged_df.columns or features_merged_df[h1_ema_col].isna().all():
                             # If EMA column missing or all NaN after ffill, cannot calculate trend
                             raise ValueError(f"'{h1_ema_col}' missing or all NaN after merge/ffill for {symbol}")

                        # Calculate h1_trend
                        features_merged_df['h1_trend'] = 0 # Default
                        features_merged_df.loc[features_merged_df['close'] > features_merged_df[h1_ema_col], 'h1_trend'] = 1
                        features_merged_df.loc[features_merged_df['close'] < features_merged_df[h1_ema_col], 'h1_trend'] = -1

                        # Drop the temporary EMA column and update features_final_df
                        features_final_df = features_merged_df.drop(columns=[h1_ema_col], errors='ignore')
                        logger.debug(f"h1_trend calculated/merged successfully for {symbol}.")

                    except Exception as merge_e:
                        logger.error(f"Error standardizing timezone or merging H1/calc h1_trend {symbol}: {merge_e}")
                        # Ensure h1_trend default column exists even on merge failure
                        if 'h1_trend' not in features_final_df.columns:
                            logger.debug(f"Adding default h1_trend=0 due to merge error for {symbol}.")
                            features_final_df['h1_trend'] = 0

                # Add default h1_trend column if it wasn't calculated or calculation disabled/failed
                elif 'h1_trend' not in features_final_df.columns:
                     logger.debug(f"Adding default h1_trend=0 for {symbol} (H1 calc disabled or failed).")
                     features_final_df['h1_trend'] = 0

                # Final check for empty DataFrame after potential merge failures
                if features_final_df.empty: logger.warning(f"Final features empty after H1 merge/processing {symbol}. Skipping."); continue

                # Get the latest row of features and its timestamp
                latest_features = features_final_df.iloc[-1]; latest_calc_time = features_final_df.index[-1]
                logger.info(f"Final features ready @ {latest_calc_time}. Has h1_trend: {'h1_trend' in latest_features.index}")

                # === Step 3.5: Apply Time Filter ===
                current_hour_utc = latest_calc_time.hour
                if not (ENTRY_START_HOUR_UTC <= current_hour_utc <= ENTRY_END_HOUR_UTC):
                    logger.info(f"Time filter: Hour {current_hour_utc} UTC outside {ENTRY_START_HOUR_UTC}-{ENTRY_END_HOUR_UTC}. Skip signal check {symbol}.")
                    continue # Skip to next symbol

                # === Step 4: Make Prediction ===
                predicted_proba_up = None;
                missing_req_features = [f for f in trained_feature_list_lower if f not in latest_features.index]
                if not missing_req_features:
                    feature_values_for_model = latest_features[trained_feature_list_lower]
                    if not feature_values_for_model.isna().any():
                         try:
                             predicted_proba_up = make_xgboost_binary_prediction(model, trained_feature_list_lower, latest_features)
                             logger.info(f"Prediction {symbol}: P(Up)={predicted_proba_up:.4f}" if predicted_proba_up is not None else f"Prediction None {symbol}")
                         except Exception as pred_e: logger.error(f"Prediction error {symbol}: {pred_e}")
                    else: logger.warning(f"NaN in model features {symbol} @ {latest_calc_time}. Features with NaN: {feature_values_for_model[feature_values_for_model.isna()].to_dict()}")
                else: logger.warning(f"Required model features missing {missing_req_features} for {symbol} @ {latest_calc_time}.")

                if predicted_proba_up is None: logger.debug("Skipping: no valid prediction."); continue

                # === Step 5: Determine Signal & Check Threshold ===
                trade_type = None; signal_desc = ""
                if predicted_proba_up >= CONFIDENCE_THRESHOLD: trade_type = mt5.ORDER_TYPE_BUY; signal_desc=f"BUY Signal (P={predicted_proba_up:.4f}>={CONFIDENCE_THRESHOLD:.2f})"
                elif predicted_proba_up <= (1.0 - CONFIDENCE_THRESHOLD): trade_type = mt5.ORDER_TYPE_SELL; signal_desc=f"SELL Signal (P={predicted_proba_up:.4f}<={1.0 - CONFIDENCE_THRESHOLD:.2f})"
                if not trade_type: logger.info(f"No signal: {symbol} P={predicted_proba_up:.4f} within confidence bounds."); continue
                logger.info(f"{signal_desc} generated for {symbol}")

                # === Step 5.5: Apply Duplicate Trade Filter ===
                logger.debug(f"Checking for existing positions for {symbol}...")
                open_positions_for_symbol = get_open_positions(symbol=symbol)
                if open_positions_for_symbol: logger.info(f"Position already open for {symbol} (Tickets: {[p.ticket for p in open_positions_for_symbol]}). Skipping new entry."); continue
                logger.debug(f"No open position found for {symbol}. Proceeding...")

                # === Step 6: Calculate Risk Parameters ===
                logger.debug(f"Calculating risk parameters for {symbol} {('BUY' if trade_type == mt5.ORDER_TYPE_BUY else 'SELL')}...")
                symbol_info = get_symbol_info(symbol); account_info = get_account_info(); tick_info = get_tick_info(symbol)
                if not all([symbol_info, account_info, tick_info]): logger.warning(f"Missing broker info for risk calc {symbol}. Skip."); continue
                entry_price = tick_info.ask if trade_type == mt5.ORDER_TYPE_BUY else tick_info.bid
                if entry_price is None or entry_price <= 0: logger.warning(f"Invalid entry price {entry_price} for {symbol}. Skip."); continue
                latest_atr = latest_features.get(atr_col_name_check, np.nan); # Use .get for safety; check ATR column name matches feature_calculator output
                # Check ATR validity specifically if SL method requires it
                sl_method_cfg = get_config('SL_METHOD', 'ATR').upper()
                if sl_method_cfg == 'ATR' and (pd.isna(latest_atr) or latest_atr <=0):
                    logger.warning(f"Invalid ATR ({latest_atr}) for ATR SL method risk calc {symbol}. Skipping trade.")
                    continue

                trade_params = None
                try: trade_params = calculate_trade_parameters(symbol_info, account_info, trade_type, entry_price, latest_atr)
                except Exception as risk_e: logger.error(f"Exception calculating risk parameters for {symbol}: {risk_e}", exc_info=True)

                if not trade_params: logger.warning(f"Risk parameter calculation failed or conditions not met for {symbol}. Skip."); continue
                lot_size = trade_params['lot_size']; stop_loss = trade_params['stop_loss_price']; take_profit = trade_params['take_profit_price']; digits = symbol_info.digits
                logger.info(f"Risk params OK for {symbol}: Lots={lot_size:.2f}, SL={stop_loss:.{digits}f}, TP={take_profit:.{digits}f}")

                # === Step 7: Execute Trade ===
                logger.info(f"Attempting trade execution for {symbol}...")
                trade_desc_short = f"{('BUY' if trade_type == mt5.ORDER_TYPE_BUY else 'SELL')} {symbol}"
                if ALERT_ON_EXECUTION: send_telegram_alert(f"Signal: {trade_desc_short} @ {entry_price:.{digits}f} (P={predicted_proba_up:.2f}) -> EXECUTE")

                trade_result = execute_trade(symbol, trade_type, entry_price, lot_size, stop_loss, take_profit)

                if trade_result and trade_result.get("success"):
                     logger.info(f"EXECUTION SUCCESS {symbol}: {trade_result}")
                     if ALERT_ON_EXECUTION: send_telegram_alert(f"EXECUTED: {trade_desc_short} {lot_size:.2f} @ {trade_result.get('fill_price', entry_price):.{digits}f}. Ticket: {trade_result.get('ticket', 'N/A')}")
                else:
                     logger.error(f"EXECUTION FAILED {symbol}: {trade_result}")
                     if ALERT_ON_FAILURE: send_telegram_alert(f"FAILED EXEC: {trade_desc_short}. Reason: {trade_result.get('error', 'Unknown') if trade_result else 'Unknown'}")

            except Exception as symbol_e:
                 logger.exception(f"Critical unhandled error processing symbol {symbol}: {symbol_e}")
                 set_state("last_error", f"Sym {symbol} Err: {symbol_e}")
                 # Send critical alert if possible
                 if ALERT_ON_FAILURE: send_telegram_alert(f"CRITICAL ENGINE ERROR processing symbol {symbol}: {symbol_e}")

            # --- End Symbol Processing ---
            symbol_end_time = time.time(); logger.info(f"--- Finished check {symbol} ({symbol_end_time - symbol_start_time:.2f}s) ---")

        # --- End Symbol Loop ---
        loop_end_time = time.time(); loop_duration = loop_end_time - loop_start_time; logger.debug(f"Loop cycle done in {loop_duration:.2f}s.")
        sleep_time = max(0.1, LOOP_SLEEP_SECONDS - loop_duration); logger.debug(f"Sleeping {sleep_time:.2f}s..."); time.sleep(sleep_time)

    logger.info("Trading engine loop terminated.")