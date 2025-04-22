# G:/Alpha1.1/run_monthly_backtests.py
# Runs backtests month-by-month with reset balance and collects results.
# FIXED: Corrected arguments passed to run_backtest function call.

import logging
import pandas as pd
import os
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- Setup Path ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Setup Logging ---
# Initialize logger first
logger = logging.getLogger(__name__)
try:
    from utils.logging_config import setup_logging
    # Call setup_logging without the override argument
    setup_logging()
    logger.info("Initialized logging via utils.logging_config.")
except Exception as log_e:
    print(f"Logging setup error: {log_e}. Basic config used.")
    # Setup basic config if util fails
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

# --- Import Backtester Function & Config ---
try:
    from backtester import run_backtest
    from config.config_loader import get_config
except ImportError as e:
    logger.critical(f"Could not import run_backtest or get_config: {e}", exc_info=True)
    sys.exit(1)

# --- Configuration ---
SYMBOL = "EURUSD"
TIMEFRAME = "M5"
START_YEAR = 2024
START_MONTH = 5 
END_YEAR = 2025
END_MONTH = 3

INITIAL_BALANCE = 10000.0
SPREAD_POINTS = 15

# --- !!! IMPORTANT: VERIFY THESE PATHS !!! ---
# --- !!! IMPORTANT: USE THESE PATHS FOR 2024/25 BACKTEST !!! ---
RAW_DATA_FILE_PATH = os.path.join(project_root, "data", "EURUSD", "EURUSD_M5_2024-04-01_to_2025-04-02.csv") # <<< UNCOMMENT/USE THIS
FEATURE_FILE_PATH = os.path.join(project_root, "data", "features_M5_20240501_to_20250331.csv") # <<< UNCOMMENT/USE THIS
# --- !!! PATHS FOR 2023 BACKTEST (Comment out for 2024/25 test) !!! ---
# RAW_DATA_FILE_PATH = os.path.join(project_root, "data", "EURUSD", "EURUSD_M5_2023-01-01_to_2023-12-31.csv") # <<< COMMENT OUT
# FEATURE_FILE_PATH = os.path.join(project_root, "data", "features_M5_20230101_to_20231231.csv") # <<< COMMENT OUT

# Load parameters from .env to pass to backtester where needed
conf_thresh = float(get_config("MIN_PREDICTION_CONFIDENCE", 0.80))
model_file = get_config("XGB_MODEL_PATH")
# SL/TP multipliers are NOT passed, they are read internally by risk_manager


# --- Main Loop ---
if __name__ == "__main__":
    logger.info("Starting Monthly Backtest Runner...")
    logger.info(f"Period: {START_YEAR}-{START_MONTH:02d} to {END_YEAR}-{END_MONTH:02d}")
    logger.info(f"Symbol: {SYMBOL}, Timeframe: {TIMEFRAME}")
    logger.info(f"Initial Balance per month: {INITIAL_BALANCE:,.2f}")
    logger.info(f"Using Raw Data: {RAW_DATA_FILE_PATH}")
    logger.info(f"Using Feature Data: {FEATURE_FILE_PATH}")
    # Log the SL/TP params that *should* be used internally by risk_manager
    sl_mult_cfg = get_config('SL_ATR_MULTIPLIER', 'N/A')
    tp_ratio_cfg = get_config('DEFAULT_TP_RR_RATIO', 'N/A')
    logger.info(f"Using Params: Conf={conf_thresh:.2f}, SL={sl_mult_cfg}*ATR, RR={tp_ratio_cfg} (from .env)")


    if not os.path.exists(RAW_DATA_FILE_PATH): logger.critical(f"Raw data file not found: {RAW_DATA_FILE_PATH}"); sys.exit(1)
    if not os.path.exists(FEATURE_FILE_PATH): logger.critical(f"Feature file not found: {FEATURE_FILE_PATH}"); sys.exit(1)
    if not model_file or not os.path.exists(os.path.join(project_root, model_file)): logger.critical(f"Model file path invalid or not found: {model_file}"); sys.exit(1)

    all_monthly_results = []
    current_date = datetime(START_YEAR, START_MONTH, 1)
    end_date_overall = datetime(END_YEAR, END_MONTH, 1) + relativedelta(months=1)

    while current_date < end_date_overall:
        month_start_str = current_date.strftime('%Y-%m-%d')
        month_end = current_date + relativedelta(months=1) - relativedelta(days=1)
        month_end_str = month_end.strftime('%Y-%m-%d')

        logger.warning(f"\n--- Running Backtest for {current_date.strftime('%Y-%m')} ---")

        try:
            # --- CORRECTED FUNCTION CALL ---
            monthly_result = run_backtest(
                symbol=SYMBOL,
                timeframe_str=TIMEFRAME,
                start_date_str=month_start_str,
                end_date_str=month_end_str,
                data_file_path=RAW_DATA_FILE_PATH,
                feature_file_path=FEATURE_FILE_PATH,
                initial_balance=INITIAL_BALANCE,
                simulated_spread_points=SPREAD_POINTS,
                # Pass only the arguments the function expects:
                confidence_threshold=conf_thresh,
                model_path=model_file
                # SL/TP multipliers are handled internally by risk_manager reading .env
            )

            if monthly_result:
                 monthly_result['Month'] = current_date.strftime('%Y-%m')
                 all_monthly_results.append(monthly_result)
            else:
                 logger.error(f"Backtest failed for {current_date.strftime('%Y-%m')}. Skipping.")

        except Exception as e:
            logger.exception(f"Error running backtest for month {current_date.strftime('%Y-%m')}: {e}")

        current_date += relativedelta(months=1) # Move to the next month

    # --- Display Results ---
    if all_monthly_results:
        results_df = pd.DataFrame(all_monthly_results)
        display_cols = [
            'Month', 'net_pnl_perc', 'max_drawdown_perc', 'total_trades',
            'win_rate_perc', 'profit_factor', 'avg_win_loss_ratio',
            'net_pnl', 'total_cost'
        ]
        for col in display_cols:
             if col not in results_df.columns: results_df[col] = pd.NA
        results_df = results_df[display_cols]
        results_df.set_index('Month', inplace=True)

        print("\n\n" + "="*30 + " Monthly Backtest Summary " + "="*30)
        pd.options.display.float_format = '{:,.2f}'.format
        print(results_df)
        print("="*82)

        summary_filename = os.path.join(project_root, "backtest_results", f"summary_monthly_{START_YEAR}{START_MONTH:02d}_to_{END_YEAR}{END_MONTH:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        try:
             results_df.to_csv(summary_filename)
             logger.info(f"Monthly summary saved to: {summary_filename}")
        except Exception as e: logger.error(f"Failed to save monthly summary CSV: {e}")
    else:
        logger.warning("No monthly results were generated.")

    logger.info("Monthly backtest runner finished.")

