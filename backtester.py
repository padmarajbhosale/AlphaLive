# G:/Alpha1.1/backtester.py
# Runs backtests using pre-calculated features.
# FIXED: Removed extra arguments from calculate_trade_parameters call.

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
from typing import Optional, Dict, List, Any # Keep typing import

# --- Setup Path ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path: sys.path.insert(0, project_root)
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
     try: from utils.logging_config import setup_logging; setup_logging()
     except ImportError: logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

# --- Import project components ---
try:
    from config.config_loader import get_config
    from models.regime_predictor import load_xgboost_binary_model, make_xgboost_binary_prediction
    # Import the function directly, it uses config internally
    from risk_management.risk_manager import calculate_trade_parameters
    # Import constants from risk_manager ONLY if needed elsewhere in backtester (e.g. for logging)
    # Otherwise, rely on calculate_trade_parameters using its internal config values
    from risk_management.risk_manager import SL_METHOD # Example if needed
    import MetaTrader5 as mt5
    try: from trading_engine.mt5_connector import initialize_mt5, shutdown_mt5, TIMEFRAME_MAP
    except ImportError:
         logger.error("Could not import from trading_engine.mt5_connector. Defining TIMEFRAME_MAP locally.")
         TIMEFRAME_MAP = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1}
         def initialize_mt5(): logger.error("MT5 init failed - connector not found."); return False
         def shutdown_mt5(): pass
except ImportError as e: logger.critical(f"Import failed in backtester: {e}", exc_info=True); sys.exit(1)

# --- Mock MT5 Objects ---
MockSymbolInfo = namedtuple("MockSymbolInfo", ["name", "point", "digits", "spread", "trade_contract_size", "trade_tick_value", "trade_tick_size", "currency_profit", "volume_min", "volume_max", "volume_step"])
MockAccountInfo = namedtuple("MockAccountInfo", ["login", "balance", "equity", "currency", "leverage"])

# --- Helper PnL Function ---
def calculate_simulated_pnl(trade: dict, current_close_price: float, symbol_info: MockSymbolInfo):
    # ... [Same PnL logic] ...
    pnl = 0.0; point = symbol_info.point; tick_value = symbol_info.trade_tick_value; tick_size = symbol_info.trade_tick_size; contract_size = symbol_info.trade_contract_size
    if point is None or point <= 0 or tick_size is None or tick_size <= 0 or contract_size is None or contract_size <= 0 or tick_value is None: return 0.0
    value_per_point_per_lot = (tick_value / tick_size) * point if tick_size > 0 else 0.0
    if value_per_point_per_lot == 0: return 0.0; price_diff = 0.0
    if trade['type'] == mt5.ORDER_TYPE_BUY: price_diff = current_close_price - trade['entry_price']
    elif trade['type'] == mt5.ORDER_TYPE_SELL: price_diff = trade['entry_price'] - current_close_price
    points_diff = price_diff / point if point > 0 else 0.0; pnl = points_diff * value_per_point_per_lot * trade['lots']
    return pnl

# --- Main Backtesting Function Definition ---
def run_backtest(
    symbol: str, timeframe_str: str, start_date_str: str, end_date_str: str,
    data_file_path: str,
    feature_file_path: str,
    initial_balance: float = 10000.0,
    simulated_spread_points: int = 15,
    entry_start_hour_utc: int = 7,
    entry_end_hour_utc: int = 18,
    model_path: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    # Removed sl_atr_mult, tp_rr_ratio from args as they are loaded from config by risk_manager
    ) -> Dict:
    """ Runs OOS backtest using XGBoost Binary model, pre-calculated features, and Time Filter. Returns results dictionary."""
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_name = f"XGB_BinDir_OOS_TFilt{entry_start_hour_utc}-{entry_end_hour_utc}"
    logger.info("="*10 + f" Starting {strategy_name} Backtest: {symbol} ({timeframe_str}) - Run: {run_timestamp} " + "="*10)
    logger.info(f"Period: {start_date_str} to {end_date_str}, Initial Balance: {initial_balance:,.2f}")

    # --- Get Configs (These will be used by risk_manager internally) ---
    conf_thresh = confidence_threshold if confidence_threshold is not None else float(get_config("MIN_PREDICTION_CONFIDENCE", 0.80))
    # sl_mult = float(get_config('SL_ATR_MULTIPLIER', 1.5)) # Loaded internally by risk_manager
    # tp_ratio = float(get_config('DEFAULT_TP_RR_RATIO', 1.5)) # Loaded internally by risk_manager
    model_file = model_path if model_path is not None else get_config("XGB_MODEL_PATH")
    atr_period = int(get_config("SL_ATR_PERIOD", 14)) # Still needed for finding ATR column
    logger.info(f"Using Confidence Threshold: {conf_thresh:.2f}") # Log threshold used
    if not model_file: logger.error("Model path not specified or found in config."); return {}

    # --- Step 1: Load RAW Data ---
    # ... [Same as before] ...
    logger.info(f"Loading RAW price data from: {data_file_path}")
    if not os.path.exists(data_file_path): logger.error(f"Data file not found: {data_file_path}"); return {}
    try:
        historical_data_full = pd.read_csv(data_file_path, parse_dates=['time'], index_col='time')
        start_dt = datetime.datetime.strptime(start_date_str, '%Y-%m-%d'); end_dt = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
        if historical_data_full.index.tz is not None: historical_data_full.index = historical_data_full.index.tz_localize(None); start_dt=pd.Timestamp(start_dt); end_dt=pd.Timestamp(end_dt)
        else: start_dt = pd.Timestamp(start_dt); end_dt = pd.Timestamp(end_dt)
        historical_data = historical_data_full.loc[start_dt:end_dt];
        if historical_data.empty: logger.error("No raw data in date range."); return {}
        logger.info(f"Filtered raw data: {start_dt} to {end_dt}. Bars: {len(historical_data)}")
    except Exception as e: logger.exception(f"Error loading raw CSV: {e}"); return {}


    # --- Step 2: Initialize Backtest State & Load Model/Props ---
    # ... [Same as before] ...
    balance = initial_balance; equity = initial_balance; open_positions = []; trade_history = []; peak_equity = initial_balance; max_drawdown = 0.0; trade_id_counter = 0
    logger.info("Loading XGBoost BINARY model...");
    model, trained_feature_list = load_xgboost_binary_model(model_file);
    if model is None or trained_feature_list is None: logger.error("Failed load XGBoost model."); return {}
    trained_feature_list_lower = [f.lower() for f in trained_feature_list]

    # --- Load MT5 Props ---
    # ... [Same as before] ...
    logger.warning("Using LIVE MT5 connection for initial symbol properties!");
    mt5_initialized = initialize_mt5();
    if not mt5_initialized: logger.error("MT5 connection failed for props."); return {}
    live_symbol_info_for_props = mt5.symbol_info(symbol); shutdown_mt5()
    if not live_symbol_info_for_props: logger.error(f"Cannot get live symbol info for {symbol}."); return {}
    mock_symbol_info = MockSymbolInfo( name=symbol, point=live_symbol_info_for_props.point, digits=live_symbol_info_for_props.digits, spread=simulated_spread_points, trade_contract_size=live_symbol_info_for_props.trade_contract_size, trade_tick_value=live_symbol_info_for_props.trade_tick_value, trade_tick_size=live_symbol_info_for_props.trade_tick_size, currency_profit=live_symbol_info_for_props.currency_profit, volume_min=live_symbol_info_for_props.volume_min, volume_max=live_symbol_info_for_props.volume_max, volume_step=live_symbol_info_for_props.volume_step)
    point = mock_symbol_info.point; digits = mock_symbol_info.digits;

    # --- Step 2b: Load Pre-calculated Features ---
    # ... [Same as before] ...
    logger.info(f"Loading pre-calculated features from: {feature_file_path}")
    if not os.path.exists(feature_file_path): logger.error(f"Feature file not found: {feature_file_path}."); return {}
    try:
         features_df_full = pd.read_csv(feature_file_path, index_col='time', parse_dates=True)
         features_df_full.columns = [col.lower() for col in features_df_full.columns]
         atr_col_name_check = f'atrr_{atr_period}'
         if atr_col_name_check not in features_df_full.columns: logger.error(f"Required ATR column '{atr_col_name_check}' not found."); return {}
         if features_df_full.index.tz is not None: features_df_full.index = features_df_full.index.tz_localize(None)
         features_df = features_df_full.reindex(historical_data.index)
         if features_df.empty or features_df.isna().all().all(): logger.error("Feature DataFrame empty/all NaNs after alignment."); return {}
         required_check_cols = trained_feature_list_lower + [atr_col_name_check]
         cols_to_check_actually_exist = [c for c in required_check_cols if c in features_df.columns]
         if cols_to_check_actually_exist:
             missing_ratio = features_df[cols_to_check_actually_exist].isna().mean().mean()
             if missing_ratio > 0.9: logger.error(f"Very high ratio of NaNs ({missing_ratio:.1%}) in features."); return {}
             elif missing_ratio > 0.1: logger.warning(f"High ratio of NaNs ({missing_ratio:.1%}) in features.")
         else: logger.warning("Could not find any required columns to check for NaNs.")
    except Exception as e: logger.exception(f"Error loading or aligning features: {e}"); return {}


    # --- Nested Helper Function to Close Simulated Trades ---
    # ... [Same as before] ...
    def close_simulated_trade(pos_to_close, close_price, close_time, reason):
        nonlocal balance, equity; logger.info(f"Sim Close: ID {pos_to_close['id']} due to {reason} @ {close_price:.{digits}f}"); pnl = calculate_simulated_pnl(pos_to_close, close_price, mock_symbol_info); sim_cost = pos_to_close.get('cost', 0.0); net_pnl = pnl - sim_cost; balance += net_pnl; temp_equity = balance; logger.debug(f" --> Closed PnL={pnl:.2f}, Cost={sim_cost:.2f}, Net={net_pnl:.2f}, NewBalance={balance:.2f}"); closed_trade = pos_to_close.copy(); closed_trade.update({ 'close_time': close_time, 'close_price': close_price, 'pnl': pnl, 'cost': sim_cost, 'net_pnl': net_pnl, 'status': f'CLOSED_{reason}' }); trade_history.append(closed_trade);
        if pos_to_close in open_positions: open_positions.remove(pos_to_close)

    # --- Step 3: Loop Through Historical Data ---
    logger.info(f"Starting simulation loop: {len(historical_data)} bars...")
    total_bars = len(historical_data)
    equity_curve = []

    for index, current_bar in historical_data.iterrows():
        current_time = index
        current_bar_number = historical_data.index.get_loc(index) + 1

        # Equity Start
        current_equity_start_of_bar = balance; current_floating_pnl_start = 0.0
        for pos in open_positions: current_floating_pnl_start += calculate_simulated_pnl(pos, current_bar['close'], mock_symbol_info)
        current_equity_start_of_bar += current_floating_pnl_start
        if current_bar_number == 1 or current_bar_number % 500 == 0 or current_bar_number == total_bars: logger.debug(f" Bar {current_bar_number}/{total_bars} ({current_time}). Bal: {balance:.2f}, Eq: {current_equity_start_of_bar:.2f}, Open: {len(open_positions)}")

        # 3a: Check SL/TP hits
        # ... [Same as before] ...
        for pos in open_positions[:]:
            pos_sl = pos['sl']; pos_tp = pos['tp']; pos_type = pos['type']; trade_closed = False; close_price = None; reason_closed = None
            if pos_type == mt5.ORDER_TYPE_BUY:
                if current_bar['low'] <= pos_sl: close_price = pos_sl; reason_closed = "SL"; trade_closed = True
                elif current_bar['high'] >= pos_tp: close_price = pos_tp; reason_closed = "TP"; trade_closed = True
            elif pos_type == mt5.ORDER_TYPE_SELL:
                if current_bar['high'] >= pos_sl: close_price = pos_sl; reason_closed = "SL"; trade_closed = True
                elif current_bar['low'] <= pos_tp: close_price = pos_tp; reason_closed = "TP"; trade_closed = True
            if trade_closed: close_simulated_trade(pos, close_price, current_time, reason_closed)

        # 3b: Get Prediction
        # ... [Same as before] ...
        predicted_proba_up = None; current_atr = np.nan; latest_features_series = None
        if current_time in features_df.index:
             latest_features_series = features_df.loc[current_time]
             required_model_features_exist = all(f in latest_features_series.index for f in trained_feature_list_lower)
             atr_col_name = f'atrr_{atr_period}'
             required_atr_exists = atr_col_name in latest_features_series.index
             if required_model_features_exist and required_atr_exists and not latest_features_series[trained_feature_list_lower + [atr_col_name]].isna().any():
                 predicted_proba_up = make_xgboost_binary_prediction(model, trained_feature_list_lower, latest_features_series)
                 atr_val = latest_features_series[atr_col_name]
                 current_atr = atr_val if pd.notna(atr_val) and atr_val > 0 else np.nan

        # 3c: Entry Logic
        # ... [Same as before] ...
        is_position_open = len(open_positions) > 0
        entry_condition_met = False; trade_type = None
        entry_hour = current_time.hour
        time_filter_passed = entry_start_hour_utc <= entry_hour <= entry_end_hour_utc

        if time_filter_passed:
            if not is_position_open and predicted_proba_up is not None:
                if predicted_proba_up >= conf_thresh: trade_type = mt5.ORDER_TYPE_BUY; entry_condition_met = True
                elif predicted_proba_up <= (1.0 - conf_thresh): trade_type = mt5.ORDER_TYPE_SELL; entry_condition_met = True

        # Execute Entry
        if entry_condition_met:
            if SL_METHOD == 'ATR' and (pd.isna(current_atr) or current_atr <= 0): logger.warning(f"Invalid ATR ({current_atr}). Skipping entry @ {current_time}")
            else:
                next_bar_index_num = current_bar_number
                if next_bar_index_num < total_bars:
                     sim_entry_price = historical_data.iloc[next_bar_index_num]['open']; sim_entry_time = historical_data.index[next_bar_index_num];
                     mock_account_info = MockAccountInfo(login=12345, balance=balance, equity=equity, currency='USD', leverage=100)
                     # --- CORRECTED CALL --- Remove extra keyword args
                     trade_params = calculate_trade_parameters(mock_symbol_info, mock_account_info, trade_type, sim_entry_price, current_atr)
                     if trade_params:
                         value_per_point_per_lot = (mock_symbol_info.trade_tick_value / mock_symbol_info.trade_tick_size) * point if mock_symbol_info.trade_tick_size > 0 else 0.0
                         entry_cost = (simulated_spread_points * value_per_point_per_lot * trade_params['lot_size']) if value_per_point_per_lot > 0 else 0.0; trade_id_counter += 1
                         new_pos = { 'id': trade_id_counter, 'symbol': symbol, 'type': trade_type, 'entry_time': sim_entry_time, 'entry_price': sim_entry_price, 'lots': trade_params['lot_size'], 'sl': trade_params['stop_loss_price'], 'tp': trade_params['take_profit_price'], 'status': 'OPEN', 'cost': entry_cost, 'pnl': -entry_cost }
                         open_positions.append(new_pos); equity -= entry_cost; logger.info(f"Sim OPEN: ID={trade_id_counter}, {('BUY' if trade_type==mt5.ORDER_TYPE_BUY else 'SELL')} {new_pos['lots']:.2f} lots @ {sim_entry_price:.{digits}f}, SL={new_pos['sl']:.{digits}f}, TP={new_pos['tp']:.{digits}f}. Cost={entry_cost:.2f}. Eq={equity:.2f}"); is_position_open = True
                     else: logger.warning(f"Parameter calculation failed @ {current_time}.")

        # 3h: Update Equity Curve, Drawdown
        # ... [Same as before] ...
        current_equity_end_of_bar = balance; current_floating_pnl = 0.0
        for pos in open_positions: current_floating_pnl += calculate_simulated_pnl(pos, current_bar['close'], mock_symbol_info)
        current_equity_end_of_bar += current_floating_pnl; equity = current_equity_end_of_bar
        peak_equity = max(peak_equity, equity); current_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0; max_drawdown = max(max_drawdown, current_drawdown)
        equity_curve.append({ 'time': current_time, 'equity': round(equity, 2)})

    # --- End of loop ---
    logger.info("Closing End-Of-Period open positions...");
    if open_positions:
        final_bar = historical_data.iloc[-1]; final_close_price = final_bar['close']; final_close_time = final_bar.name
        for pos in open_positions[:]: close_simulated_trade(pos, final_close_price, final_close_time, "EOP")

    # --- Step 4: Calculate & Return Metrics ---
    # ... [Same calculation logic] ...
    logger.info("Calculating performance metrics...");
    total_trades = len(trade_history); net_profit_loss = balance - initial_balance; gross_profit = 0.0; gross_loss = 0.0; winning_trades = 0; losing_trades = 0; largest_win = 0.0; largest_loss = 0.0; total_cost = 0.0; be_trades = 0; all_pnls = []
    if total_trades > 0:
        for i, trade in enumerate(trade_history):
            pnl = trade.get('net_pnl', 0.0); cost = trade.get('cost', 0.0); total_cost += cost; all_pnls.append(pnl);
            if pnl > 0: winning_trades += 1; gross_profit += pnl;
            elif pnl < 0: losing_trades += 1; gross_loss += abs(pnl);
            else: be_trades += 1
        largest_win = max(all_pnls) if all_pnls else 0.0; largest_loss = min(all_pnls) if all_pnls else 0.0
    win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0; profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf'); average_win = (gross_profit / winning_trades) if winning_trades > 0 else 0.0; average_loss = (gross_loss / losing_trades) if losing_trades > 0 else 0.0; risk_reward_ratio = (average_win / average_loss) if average_loss > 0 else float('inf')

    # Print Results
    print("\n" + "="*20 + f" Backtest Results: {start_date_str} to {end_date_str} " + "="*20); print(f" Symbol: {symbol} ({timeframe_str}), Initial Bal: {initial_balance:,.2f}"); print(f" Final Balance: {balance:,.2f}, Net P/L: {net_profit_loss:,.2f} ({(net_profit_loss / initial_balance) * 100.0:.2f}%)"); print(f" Max Drawdown: {max_drawdown:.2%}, Total Costs: {total_cost:,.2f}"); print("-" * 70); print(f" Total Trades: {total_trades}, Wins: {winning_trades} ({win_rate:.2f}%), Losses: {losing_trades}, BE: {be_trades}"); avg_loss_str = f"{average_loss:.2f}" if average_loss != 0 else "0.00"; print(f" Profit Factor: {profit_factor:.2f}, Avg Win/Loss: {risk_reward_ratio:.2f} ({average_win:.2f} / {avg_loss_str})"); print(f" Largest Win: {largest_win:,.2f}, Largest Loss: {largest_loss:,.2f}"); print("="*70)

    # --- RETURN Results Dictionary ---
    results = {
        'start_date': start_date_str, 'end_date': end_date_str, 'symbol': symbol, 'timeframe': timeframe_str,
        'initial_balance': initial_balance, 'final_balance': round(balance, 2),
        'net_pnl': round(net_profit_loss, 2), 'net_pnl_perc': round((net_profit_loss / initial_balance) * 100.0, 2) if initial_balance else 0,
        'max_drawdown_perc': round(max_drawdown * 100.0, 2), 'total_trades': total_trades,
        'total_cost': round(total_cost, 2), 'winning_trades': winning_trades, 'losing_trades': losing_trades,
        'win_rate_perc': round(win_rate, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else np.inf,
        'avg_win_loss_ratio': round(risk_reward_ratio, 2) if risk_reward_ratio != float('inf') else np.inf,
        'average_win': round(average_win, 2), 'average_loss': round(average_loss, 2),
        'largest_win': round(largest_win, 2), 'largest_loss': round(largest_loss, 2)
    }
    logger.info("="*10 + " Backtest Function Finished " + "="*10)
    return results
# End of run_backtest function

# --- Main Execution Block (if run directly) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OOS Backtest using trained XGBoost Binary model and pre-calculated features.")
    parser.add_argument("-s", "--symbol", required=True, help="Symbol")
    parser.add_argument("-tf", "--timeframe", required=True, help=f"Timeframe")
    parser.add_argument("-start", "--startdate", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("-end", "--enddate", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("-d", "--datafile", required=True, help="Path to RAW PRICE DATA CSV")
    parser.add_argument("-ff", "--featurefile", required=True, help="Path to PRE-CALCULATED FEATURE CSV")
    parser.add_argument("-b", "--balance", type=float, default=10000.0, help="Initial balance")
    parser.add_argument("-spread", "--spreadpoints", type=int, default=15, help="Simulated spread")
    args = parser.parse_args()

    # Load params from config/env for direct run
    conf = float(get_config("MIN_PREDICTION_CONFIDENCE", 0.80))
    # Note: SL/TP multipliers are NOT passed here, run_backtest loads them internally via risk_manager
    model_f = get_config("XGB_MODEL_PATH")

    results_dict = run_backtest(
        symbol=args.symbol.upper(), timeframe_str=args.timeframe.upper(),
        start_date_str=args.startdate, end_date_str=args.enddate,
        data_file_path=args.datafile, feature_file_path=args.featurefile,
        initial_balance=args.balance, simulated_spread_points=args.spreadpoints,
        confidence_threshold=conf, model_path=model_f # Pass confidence and model path
    )
    if results_dict:
        print("\n--- Results Dictionary ---")
        import json
        print(json.dumps(results_dict, indent=2, default=str))
    else: print("\nBacktest failed to produce results.")
    logger.info("Backtesting script finished.")

