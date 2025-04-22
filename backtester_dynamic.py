# G:/AlphaLive/backtester_dynamic.py
# Runs backtests calculating features dynamically on each bar.

import logging
import pandas as pd
import os
import sys
import datetime
import math
import argparse
from collections import namedtuple
import time
import numpy as np
from typing import Optional, Dict, List, Any

# --- Setup Path ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path: sys.path.insert(0, project_root)
logger = logging.getLogger(__name__)
# Attempt to use configured logging, fallback to basic
if not logger.hasHandlers():
     try: from utils.logging_config import setup_logging; setup_logging()
     except ImportError: logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

# --- Import project components ---
try:
    from config.config_loader import get_config
    from models.regime_predictor import load_xgboost_binary_model, make_xgboost_binary_prediction
    from risk_management.risk_manager import calculate_trade_parameters
    # Import the feature calculation function
    from features.feature_calculator import calculate_features
    import MetaTrader5 as mt5
    try:
        from trading_engine.mt5_connector import initialize_mt5, shutdown_mt5, TIMEFRAME_MAP
    except ImportError:
         logger.error("Could not import from trading_engine.mt5_connector. Defining TIMEFRAME_MAP locally.")
         # Provide a minimal fallback TIMEFRAME_MAP if connector is missing (for standalone testing)
         TIMEFRAME_MAP = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 16385, "H4": 16388, "D1": 16408, "W1": 32769, "MN1": 49153}
         def initialize_mt5(): logger.error("MT5 init failed - connector not found."); return False
         def shutdown_mt5(): pass
except ImportError as e:
    logger.critical(f"Import failed in backtester_dynamic: {e}", exc_info=True)
    sys.exit(1)

# --- Constants ---
# Minimum bars needed for feature calculation (adjust based on longest indicator period + buffer)
REQUIRED_FEATURE_LOOKBACK = 200
# Minimum bars in a window before attempting feature calculation
MIN_BARS_FOR_CALCULATION = 50

# --- Mock MT5 Objects ---
MockSymbolInfo = namedtuple("MockSymbolInfo", ["name", "point", "digits", "spread", "trade_contract_size", "trade_tick_value", "trade_tick_size", "currency_profit", "volume_min", "volume_max", "volume_step"])
MockAccountInfo = namedtuple("MockAccountInfo", ["login", "balance", "equity", "currency", "leverage"])

# --- Helper PnL Function ---
def calculate_simulated_pnl(trade: dict, current_close_price: float, symbol_info: MockSymbolInfo):
    pnl = 0.0
    point = symbol_info.point
    tick_value = symbol_info.trade_tick_value
    tick_size = symbol_info.trade_tick_size
    contract_size = symbol_info.trade_contract_size

    # Basic validation
    if point is None or point <= 0 or tick_size is None or tick_size <= 0 or contract_size is None or contract_size <= 0 or tick_value is None:
        logger.error(f"Invalid symbol properties for PnL calc: {symbol_info}")
        return 0.0

    value_per_point_per_lot = (tick_value / tick_size) * point if tick_size > 0 else 0.0
    if value_per_point_per_lot == 0:
        logger.warning(f"Value per point is zero for {symbol_info.name}, cannot calculate PnL accurately.")
        return 0.0

    price_diff = 0.0
    if trade['type'] == mt5.ORDER_TYPE_BUY:
        price_diff = current_close_price - trade['entry_price']
    elif trade['type'] == mt5.ORDER_TYPE_SELL:
        price_diff = trade['entry_price'] - current_close_price

    points_diff = price_diff / point if point > 0 else 0.0
    pnl = points_diff * value_per_point_per_lot * trade['lots']
    return pnl

# --- Main Backtesting Function Definition ---
def run_backtest_dynamic( # Renamed function
    symbol: str, timeframe_str: str, start_date_str: str, end_date_str: str,
    data_file_path: str, # Raw data path is still needed
    initial_balance: float = 10000.0,
    simulated_spread_points: int = 15,
    entry_start_hour_utc: int = 7,
    entry_end_hour_utc: int = 18,
    model_path: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    # Note: feature_file_path argument removed
    ) -> Dict:
    """ Runs OOS backtest calculating features dynamically. Returns results dictionary."""
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_name = f"XGB_BinDir_OOS_TFilt{entry_start_hour_utc}-{entry_end_hour_utc}_DynFeat" # Indicate dynamic features
    logger.info("="*10 + f" Starting {strategy_name} Backtest: {symbol} ({timeframe_str}) - Run: {run_timestamp} " + "="*10)
    logger.info(f"Period: {start_date_str} to {end_date_str}, Initial Balance: {initial_balance:,.2f}")
    logger.warning("!!! Feature calculation is DYNAMIC - this will be slower than using pre-calculated files !!!")

    # --- Get Configs ---
    conf_thresh = confidence_threshold if confidence_threshold is not None else float(get_config("MIN_PREDICTION_CONFIDENCE", 0.80))
    model_file = model_path if model_path is not None else get_config("XGB_MODEL_PATH")
    atr_period = int(get_config("SL_ATR_PERIOD", 14)) # Needed for finding ATR column name
    logger.info(f"Using Confidence Threshold: {conf_thresh:.2f}")
    if not model_file: logger.error("Model path not specified or found in config."); return {}

    # --- Step 1: Load RAW Data ---
    logger.info(f"Loading RAW price data from: {data_file_path}")
    if not os.path.exists(data_file_path):
        logger.error(f"Raw Data file not found: {data_file_path}")
        return {}
    try:
        # Load the ENTIRE raw data file first
        historical_data_full = pd.read_csv(data_file_path, parse_dates=['time'], index_col='time')
        # Ensure standard column names (lowercase)
        historical_data_full.columns = [col.lower() for col in historical_data_full.columns]
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in historical_data_full.columns for col in required_cols):
            logger.error(f"Raw data file missing required columns: {required_cols}")
            return {}
        # Convert timezone if present, ensure datetime index
        if historical_data_full.index.tz is not None:
            logger.info("Localizing timezone to None for consistency.")
            historical_data_full.index = historical_data_full.index.tz_localize(None)
        historical_data_full.index = pd.to_datetime(historical_data_full.index)

        # Filter for the specific backtest period
        start_dt = pd.Timestamp(datetime.datetime.strptime(start_date_str, '%Y-%m-%d'))
        end_dt = pd.Timestamp(datetime.datetime.strptime(end_date_str, '%Y-%m-%d').replace(hour=23, minute=59, second=59))

        # Select data for the requested period PLUS the lookback needed for features
        start_dt_with_lookback = start_dt - pd.Timedelta(days=math.ceil(REQUIRED_FEATURE_LOOKBACK * (TIMEFRAME_MAP[timeframe_str.upper()]/1440.0) * 1.5 )) # Estimate days needed for lookback, add buffer
        historical_data_for_run = historical_data_full.loc[start_dt_with_lookback:end_dt].copy()

        if historical_data_for_run.empty:
             logger.error("No raw data found for the specified date range including lookback.")
             return {}

        logger.info(f"Loaded {len(historical_data_full)} total raw bars. Using {len(historical_data_for_run)} bars (including lookback) for run.")
        logger.info(f"Backtest period: {start_dt} to {end_dt}")

    except Exception as e:
        logger.exception(f"Error loading or filtering raw CSV: {e}")
        return {}

    # --- Step 2: Initialize Backtest State & Load Model/Props ---
    balance = initial_balance
    equity = initial_balance
    open_positions = []
    trade_history = []
    peak_equity = initial_balance
    max_drawdown = 0.0
    trade_id_counter = 0

    logger.info("Loading XGBoost BINARY model artifacts...")
    # Pass the derived model file path to the loading function
    model_load_path = os.path.join(project_root, model_file) if not os.path.isabs(model_file) else model_file
    model, trained_feature_list = load_xgboost_binary_model(model_load_path) # Use full path

    if model is None or trained_feature_list is None:
        logger.error(f"Failed load XGBoost model or features list from {model_load_path}")
        return {}
    trained_feature_list_lower = [f.lower() for f in trained_feature_list]
    logger.info(f"Model expects {len(trained_feature_list_lower)} features.")

    # --- Load MT5 Props (using live connection temporarily) ---
    logger.warning("Using LIVE MT5 connection for initial symbol properties!")
    mt5_initialized = initialize_mt5()
    if not mt5_initialized: logger.error("MT5 connection failed for props."); return {}
    live_symbol_info_for_props = mt5.symbol_info(symbol)
    shutdown_mt5() # Close connection after getting props
    if not live_symbol_info_for_props:
        logger.error(f"Cannot get live symbol info for {symbol}.")
        return {}
    mock_symbol_info = MockSymbolInfo(
        name=symbol, point=live_symbol_info_for_props.point, digits=live_symbol_info_for_props.digits,
        spread=simulated_spread_points, trade_contract_size=live_symbol_info_for_props.trade_contract_size,
        trade_tick_value=live_symbol_info_for_props.trade_tick_value, trade_tick_size=live_symbol_info_for_props.trade_tick_size,
        currency_profit=live_symbol_info_for_props.currency_profit, volume_min=live_symbol_info_for_props.volume_min,
        volume_max=live_symbol_info_for_props.volume_max, volume_step=live_symbol_info_for_props.volume_step
    )
    point = mock_symbol_info.point
    digits = mock_symbol_info.digits
    atr_col_name_check = f'atrr_{atr_period}' # Define expected ATR column name

    # --- Nested Helper Function to Close Simulated Trades ---
    def close_simulated_trade(pos_to_close, close_price, close_time, reason):
        nonlocal balance, equity
        logger.info(f"Sim Close: ID {pos_to_close['id']} ({('BUY' if pos_to_close['type']==mt5.ORDER_TYPE_BUY else 'SELL')}) due to {reason} @ {close_price:.{digits}f}")
        pnl = calculate_simulated_pnl(pos_to_close, close_price, mock_symbol_info)
        sim_cost = pos_to_close.get('cost', 0.0)
        net_pnl = pnl - sim_cost
        balance += net_pnl
        # Equity is calculated fresh each bar end, balance is cumulative PnL
        logger.debug(f" --> Closed PnL={pnl:.2f}, Cost={sim_cost:.2f}, Net={net_pnl:.2f}, NewBalance={balance:.2f}")
        closed_trade = pos_to_close.copy()
        closed_trade.update({
            'close_time': close_time,
            'close_price': close_price,
            'pnl': pnl,
            'cost': sim_cost,
            'net_pnl': net_pnl,
            'status': f'CLOSED_{reason}'
        })
        trade_history.append(closed_trade)
        if pos_to_close in open_positions:
             open_positions.remove(pos_to_close)

    # --- Step 3: Loop Through Historical Data (Only the requested period) ---
    # Filter the loaded data to only iterate through the actual requested period
    historical_data_period = historical_data_for_run.loc[start_dt:end_dt]
    logger.info(f"Starting simulation loop for requested period: {len(historical_data_period)} bars...")
    total_bars = len(historical_data_period)
    equity_curve = []
    last_log_bar = -501 # Throttle logging

    for i, (current_time, current_bar) in enumerate(historical_data_period.iterrows()):
        current_bar_number = i + 1

        # --- Log Progress ---
        if current_bar_number == 1 or current_bar_number - last_log_bar > 500 or current_bar_number == total_bars:
            current_equity_start_of_bar = balance
            current_floating_pnl_start = 0.0
            for pos in open_positions:
                current_floating_pnl_start += calculate_simulated_pnl(pos, current_bar['close'], mock_symbol_info)
            current_equity_start_of_bar += current_floating_pnl_start
            logger.debug(f" Bar {current_bar_number}/{total_bars} ({current_time}). Bal: {balance:.2f}, Eq: {current_equity_start_of_bar:.2f}, Open: {len(open_positions)}")
            last_log_bar = current_bar_number

        # --- 3a: Check SL/TP hits ---
        # Iterate over a copy for safe removal
        for pos in open_positions[:]:
            pos_sl = pos['sl']
            pos_tp = pos['tp']
            pos_type = pos['type']
            trade_closed = False
            close_price = None
            reason_closed = None

            # Check SL/TP based on current bar's high/low
            if pos_type == mt5.ORDER_TYPE_BUY:
                if current_bar['low'] <= pos_sl:
                    close_price = pos_sl; reason_closed = "SL"; trade_closed = True
                elif current_bar['high'] >= pos_tp:
                    close_price = pos_tp; reason_closed = "TP"; trade_closed = True
            elif pos_type == mt5.ORDER_TYPE_SELL:
                if current_bar['high'] >= pos_sl:
                    close_price = pos_sl; reason_closed = "SL"; trade_closed = True
                elif current_bar['low'] <= pos_tp:
                    close_price = pos_tp; reason_closed = "TP"; trade_closed = True

            if trade_closed:
                close_simulated_trade(pos, close_price, current_time, reason_closed)

        # --- 3b: Calculate Features Dynamically ---
        latest_features_series = None
        current_atr = np.nan # Reset for this bar

        # Find the integer location of the current bar within the larger dataframe containing lookback
        try:
            current_bar_iloc_in_full = historical_data_for_run.index.get_loc(current_time)
        except KeyError:
            logger.error(f"Timestamp {current_time} not found in historical_data_for_run index. Skipping bar.")
            continue

        # Determine window for feature calculation
        start_iloc = max(0, current_bar_iloc_in_full - REQUIRED_FEATURE_LOOKBACK + 1)
        end_iloc = current_bar_iloc_in_full + 1 # Include current bar

        if end_iloc - start_iloc >= MIN_BARS_FOR_CALCULATION:
            raw_data_window = historical_data_for_run.iloc[start_iloc:end_iloc]
            try:
                # Calculate features on this window
                features_df_calculated = calculate_features(raw_data_window.copy()) # Use copy

                if features_df_calculated is not None and not features_df_calculated.empty:
                    # Get the features for the very last bar (which corresponds to current_time)
                    latest_features_series = features_df_calculated.iloc[-1]

                    # Extract ATR value needed for risk manager
                    if atr_col_name_check in latest_features_series.index:
                        atr_val = latest_features_series[atr_col_name_check]
                        if pd.notna(atr_val) and atr_val > 0:
                            current_atr = atr_val
                        # else: logger.debug(f"NaN or zero ATR calculated for {current_time}")
                    # else: logger.warning(f"ATR column '{atr_col_name_check}' missing in calculated features at {current_time}")

                # else: logger.warning(f"Feature calculation returned None or empty for window ending {current_time}")

            except Exception as calc_e:
                logger.error(f"Error calculating features dynamically at {current_time}: {calc_e}", exc_info=False) # Set exc_info=True for full traceback

        # else: logger.debug(f"Not enough historical data ({end_iloc - start_iloc} bars) for feature calculation at {current_time}")

        # --- 3c: Get Prediction (if features calculated) ---
        predicted_proba_up = None
        if latest_features_series is not None:
            # Check if model features exist and are not NaN
            required_model_features_exist = all(f in latest_features_series.index for f in trained_feature_list_lower)
            if required_model_features_exist:
                 feature_values_for_model = latest_features_series[trained_feature_list_lower]
                 if not feature_values_for_model.isna().any():
                      try:
                          predicted_proba_up = make_xgboost_binary_prediction(model, trained_feature_list_lower, latest_features_series)
                      except Exception as pred_e:
                          logger.error(f"Prediction error at {current_time}: {pred_e}")
                 # else: logger.debug(f"NaN found in required model features for {current_time}, skipping prediction.")
            # else: logger.debug(f"Required model features missing in calculated features for {current_time}.")


        # --- 3d: Entry Logic ---
        is_position_open = len(open_positions) > 0
        entry_condition_met = False
        trade_type = None
        entry_hour = current_time.hour
        time_filter_passed = entry_start_hour_utc <= entry_hour <= entry_end_hour_utc

        if time_filter_passed:
            if not is_position_open and predicted_proba_up is not None:
                # Use the confidence threshold loaded earlier
                if predicted_proba_up >= conf_thresh:
                    trade_type = mt5.ORDER_TYPE_BUY
                    entry_condition_met = True
                elif predicted_proba_up <= (1.0 - conf_thresh):
                    trade_type = mt5.ORDER_TYPE_SELL
                    entry_condition_met = True

        # --- Execute Entry ---
        if entry_condition_met:
            # Check ATR validity only if SL method is ATR
            sl_method_used = get_config('SL_METHOD', 'ATR').upper() # Get SL method from config again
            if sl_method_used == 'ATR' and (pd.isna(current_atr) or current_atr <= 0):
                logger.warning(f"Invalid ATR ({current_atr}) for ATR SL method. Skipping entry @ {current_time}")
            else:
                # Simulate entry on the open of the *next* bar
                next_bar_index_iloc = current_bar_number # i is 0-based index in loop, bar_number is 1-based
                if next_bar_index_iloc < len(historical_data_period): # Ensure next bar exists within the period
                     sim_entry_price = historical_data_period.iloc[next_bar_index_iloc]['open']
                     sim_entry_time = historical_data_period.index[next_bar_index_iloc]

                     # Calculate trade parameters using risk manager
                     mock_account_info = MockAccountInfo(login=12345, balance=balance, equity=equity, currency='USD', leverage=100)
                     try:
                         trade_params = calculate_trade_parameters(
                             symbol_info=mock_symbol_info,
                             account_info=mock_account_info,
                             trade_type=trade_type,
                             entry_price=sim_entry_price,
                             latest_atr=current_atr # Pass the calculated ATR
                         )
                     except Exception as risk_e:
                          logger.error(f"Error calling risk manager at {current_time}: {risk_e}")
                          trade_params = None

                     if trade_params:
                         # Calculate simulated cost
                         value_per_point_per_lot = (mock_symbol_info.trade_tick_value / mock_symbol_info.trade_tick_size) * point if mock_symbol_info.trade_tick_size > 0 else 0.0
                         entry_cost = (simulated_spread_points * value_per_point_per_lot * trade_params['lot_size']) if value_per_point_per_lot > 0 else 0.0

                         trade_id_counter += 1
                         new_pos = {
                             'id': trade_id_counter, 'symbol': symbol, 'type': trade_type,
                             'entry_time': sim_entry_time, 'entry_price': sim_entry_price,
                             'lots': trade_params['lot_size'],
                             'sl': trade_params['stop_loss_price'],
                             'tp': trade_params['take_profit_price'],
                             'status': 'OPEN', 'cost': entry_cost, 'pnl': -entry_cost # Initial PnL is negative cost
                         }
                         open_positions.append(new_pos)
                         equity -= entry_cost # Reduce equity by simulated cost
                         logger.info(f"Sim OPEN: ID={trade_id_counter}, {('BUY' if trade_type==mt5.ORDER_TYPE_BUY else 'SELL')} {new_pos['lots']:.2f} lots @ {sim_entry_price:.{digits}f}, SL={new_pos['sl']:.{digits}f}, TP={new_pos['tp']:.{digits}f}. Cost={entry_cost:.2f}. Eq={equity:.2f}")
                         is_position_open = True # Update status immediately
                     else:
                         logger.warning(f"Parameter calculation failed @ {current_time}. Skipping trade.")
                else:
                     logger.info(f"Reached end of period, cannot enter trade on next bar after {current_time}")


        # --- 3h: Update Equity Curve, Drawdown ---
        current_equity_end_of_bar = balance # Start with cash balance
        current_floating_pnl = 0.0
        for pos in open_positions:
            current_floating_pnl += calculate_simulated_pnl(pos, current_bar['close'], mock_symbol_info)

        current_equity_end_of_bar += current_floating_pnl # Add floating PnL
        equity = current_equity_end_of_bar # Update overall equity

        # Calculate Drawdown
        peak_equity = max(peak_equity, equity)
        current_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        max_drawdown = max(max_drawdown, current_drawdown)

        equity_curve.append({ 'time': current_time, 'equity': round(equity, 2)})

    # --- End of loop ---
    logger.info("Simulation loop finished.")
    logger.info("Closing any End-Of-Period open positions...")
    if open_positions:
        final_bar = historical_data_period.iloc[-1]
        final_close_price = final_bar['close']
        final_close_time = final_bar.name
        for pos in open_positions[:]: # Iterate over copy
            close_simulated_trade(pos, final_close_price, final_close_time, "EOP")

    # --- Step 4: Calculate & Return Metrics ---
    logger.info("Calculating final performance metrics...")
    total_trades = len(trade_history)
    net_profit_loss = balance - initial_balance
    gross_profit = 0.0
    gross_loss = 0.0
    winning_trades = 0
    losing_trades = 0
    largest_win = 0.0
    largest_loss = 0.0
    total_cost = 0.0
    be_trades = 0 # Break-even trades
    all_pnls = []

    if total_trades > 0:
        for i, trade in enumerate(trade_history):
            pnl = trade.get('net_pnl', 0.0)
            cost = trade.get('cost', 0.0)
            total_cost += cost
            all_pnls.append(pnl)

            if pnl > 0:
                winning_trades += 1
                gross_profit += pnl
                largest_win = max(largest_win, pnl)
            elif pnl < 0:
                losing_trades += 1
                gross_loss += abs(pnl)
                largest_loss = min(largest_loss, pnl)
            else:
                be_trades += 1

    win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    average_win = (gross_profit / winning_trades) if winning_trades > 0 else 0.0
    average_loss = (gross_loss / losing_trades) if losing_trades > 0 else 0.0
    # Use abs() for average loss when calculating ratio
    risk_reward_ratio = (average_win / average_loss) if average_loss > 0 else float('inf')

    # --- Print Summary Table ---
    print("\n" + "="*20 + f" DYNAMIC Backtest Results: {start_date_str} to {end_date_str} " + "="*20)
    print(f" Symbol: {symbol} ({timeframe_str}), Initial Bal: {initial_balance:,.2f}")
    print(f" Final Balance: {balance:,.2f}, Net P/L: {net_profit_loss:,.2f} ({(net_profit_loss / initial_balance) * 100.0:.2f}%)")
    print(f" Max Drawdown: {max_drawdown:.2%}, Total Costs (Spread): {total_cost:,.2f}")
    print("-" * 70)
    print(f" Total Trades: {total_trades}, Wins: {winning_trades} ({win_rate:.2f}%), Losses: {losing_trades}, BE: {be_trades}")
    avg_loss_str = f"{average_loss:.2f}" if average_loss != 0 else "0.00" # Display avg loss as positive
    print(f" Profit Factor: {profit_factor:.2f}, Avg Win/Loss Ratio: {risk_reward_ratio:.2f} ({average_win:.2f} / {avg_loss_str})")
    print(f" Largest Win: {largest_win:,.2f}, Largest Loss: {largest_loss:,.2f}") # Largest loss is negative
    print("=" * 70)

    # --- Prepare Results Dictionary ---
    results = {
        'start_date': start_date_str, 'end_date': end_date_str, 'symbol': symbol, 'timeframe': timeframe_str,
        'initial_balance': initial_balance, 'final_balance': round(balance, 2),
        'net_pnl': round(net_profit_loss, 2),
        'net_pnl_perc': round((net_profit_loss / initial_balance) * 100.0, 2) if initial_balance else 0,
        'max_drawdown_perc': round(max_drawdown * 100.0, 2),
        'total_trades': total_trades,
        'total_cost': round(total_cost, 2),
        'winning_trades': winning_trades, 'losing_trades': losing_trades,
        'win_rate_perc': round(win_rate, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else np.inf,
        'avg_win_loss_ratio': round(risk_reward_ratio, 2) if risk_reward_ratio != float('inf') else np.inf,
        'average_win': round(average_win, 2), 'average_loss': round(average_loss, 2), # Report actual avg loss (positive)
        'largest_win': round(largest_win, 2), 'largest_loss': round(largest_loss, 2) # Report actual largest loss (negative)
    }

    # --- Save Trade History (Optional) ---
    if trade_history:
         results_dir = os.path.join(project_root, "backtest_results")
         os.makedirs(results_dir, exist_ok=True)
         trades_filename = os.path.join(results_dir, f"trades_{symbol}_{timeframe_str}_{start_date_str}_to_{end_date_str}_{strategy_name}_{run_timestamp}.csv")
         try:
             pd.DataFrame(trade_history).to_csv(trades_filename, index=False)
             logger.info(f"Trade history saved to: {trades_filename}")
         except Exception as save_e:
             logger.error(f"Failed to save trade history: {save_e}")


    logger.info("="*10 + " Dynamic Backtest Function Finished " + "="*10)
    return results
# End of run_backtest_dynamic function

# --- Main Execution Block (if run directly) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OOS Backtest calculating features dynamically.")
    parser.add_argument("-s", "--symbol", required=True, help="Symbol (e.g., EURUSD)")
    parser.add_argument("-tf", "--timeframe", required=True, help="Timeframe (e.g., M5)")
    parser.add_argument("-start", "--startdate", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("-end", "--enddate", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("-d", "--datafile", required=True, help="Path to RAW PRICE DATA CSV")
    # Feature file argument removed
    parser.add_argument("-b", "--balance", type=float, default=10000.0, help="Initial balance")
    parser.add_argument("-spread", "--spreadpoints", type=int, default=15, help="Simulated spread")
    args = parser.parse_args()

    # Load relevant params from config/env
    conf = float(get_config("MIN_PREDICTION_CONFIDENCE", 0.80))
    model_f = get_config("XGB_MODEL_PATH") # Path relative to project root or absolute

    results_dict = run_backtest_dynamic( # Call the new function
        symbol=args.symbol.upper(), timeframe_str=args.timeframe.upper(),
        start_date_str=args.startdate, end_date_str=args.enddate,
        data_file_path=args.datafile, # Still need raw data
        initial_balance=args.balance, simulated_spread_points=args.spreadpoints,
        confidence_threshold=conf, model_path=model_f
    )

    if results_dict:
        print("\n--- Results Dictionary (Dynamic Features) ---")
        import json
        print(json.dumps(results_dict, indent=2, default=str)) # Use json for potentially nested dicts/lists if needed
    else:
        print("\nDynamic backtest failed to produce results.")

    logger.info("Dynamic backtesting script finished.")