# G:/AlphaLive/trading_engine/mt5_connector.py
# Handles connection and basic interaction with MetaTrader 5
# Includes TIMEFRAME_MAP, functions for getting bars, symbol/account/tick info, open positions, and placing market orders.
# CORRECTED: AttributeError for mt5.Position type hint in get_open_positions.

import MetaTrader5 as mt5
import logging
import time
import pandas as pd
from collections import namedtuple
# Make sure List, Any, Optional, Tuple are imported from typing
from typing import List, Dict, Any, Optional, Tuple # <-- Ensure Any is here
import math # For volume step calculation

# --- Setup Path & Config ---
import os
import sys
try:
    # Assumes this script is in trading_engine, project root is parent
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from config.config_loader import get_config
except ImportError:
    print(f"WARNING: Could not import get_config relative path in mt5_connector. Using os.getenv.")
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env') # Check parent dir
    if not os.path.exists(env_path):
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env') # Check current dir
    load_dotenv(dotenv_path=env_path if os.path.exists(env_path) else None)
    def get_config(key, default=None): return os.getenv(key, default)

logger = logging.getLogger(__name__)
# Add basic handler if logger has no handlers (useful if run standalone)
if not logger.hasHandlers():
    handler = logging.StreamHandler(); formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter); logger.addHandler(handler); logger.setLevel(logging.INFO)
    logger.info("Basic logging handler added in mt5_connector.")


# --- MT5 Timeframe Mapping ---
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1,
}

def initialize_mt5() -> bool:
    """Initializes the MetaTrader 5 connection with retries."""
    logger.info("Initializing MetaTrader 5 connection...")
    login_str = get_config("MT5_LOGIN")
    password = get_config("MT5_PASSWORD")
    server = get_config("MT5_SERVER")

    if not login_str or not password or not server:
        logger.error("MT5 credentials/server missing in configuration.")
        return False
    try:
        login = int(login_str)
    except ValueError:
        logger.error(f"Invalid MT5_LOGIN format: '{login_str}'. Must be an integer.")
        return False

    max_retries = 3
    retry_delay = 5 # seconds
    for attempt in range(max_retries):
        logger.info(f"Attempting MT5 initialization... (Attempt {attempt + 1}/{max_retries})")
        try:
            if not mt5.initialize():
                 logger.warning(f"MT5 initialize() failed. Error: {mt5.last_error()}. Retrying...")
                 time.sleep(retry_delay)
                 continue

            term_info_init = mt5.terminal_info()
            if not term_info_init:
                 logger.warning(f"MT5 initialize() OK but terminal_info None. Error: {mt5.last_error()}. Retrying...")
                 mt5.shutdown()
                 time.sleep(retry_delay)
                 continue

            if not mt5.login(login=login, password=password, server=server):
                logger.warning(f"MT5 explicit login failed. Error: {mt5.last_error()}. Retrying...")
                mt5.shutdown()
                time.sleep(retry_delay)
                continue

            term_info = mt5.terminal_info()
            acc_info = mt5.account_info()

            if term_info and acc_info:
                logger.info(f"MT5 connection initialized. Build: {term_info.build}, Account: {acc_info.login}, Bal: {acc_info.balance} {acc_info.currency}")
                return True
            else:
                 logger.error(f"MT5 init/login OK but get info failed. Error: {mt5.last_error()}")
                 mt5.shutdown()
                 if attempt < max_retries - 1: logger.info(f"Retrying in {retry_delay}s..."); time.sleep(retry_delay)
                 else: logger.error("MT5 init failed after retries."); return False
        except Exception as e:
            logger.exception(f"Unexpected error during MT5 initialization attempt {attempt + 1}: {e}")
            try: mt5.shutdown()
            except: pass
            if attempt < max_retries - 1: logger.info(f"Retrying in {retry_delay}s..."); time.sleep(retry_delay)
            else: logger.error("MT5 init failed after unexpected errors."); return False
    return False

def shutdown_mt5():
    """Shuts down the MetaTrader 5 connection."""
    logger.info("Shutting down MetaTrader 5 connection...")
    try:
        mt5.shutdown()
        logger.info("MT5 connection shut down.")
    except Exception as e:
        logger.error(f"Exception during MT5 shutdown: {e}")

def is_mt5_connected() -> bool:
    """Checks if the MT5 terminal connection is active."""
    try:
        # Checking terminal_info is a reasonable proxy for an active connection
        term_info = mt5.terminal_info()
        return term_info is not None
    except Exception as e:
        logger.error(f"Exception checking MT5 connection: {e}")
        return False

def get_symbol_info(symbol: str) -> Optional[mt5.SymbolInfo]:
    """Gets live symbol information from MT5."""
    if not is_mt5_connected(): logger.warning("MT5 not connected."); return None
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            if mt5.symbol_select(symbol, True):
                logger.debug(f"Selected {symbol} in MarketWatch, retrying info...")
                time.sleep(0.1)
                symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Failed get symbol info for {symbol}. Error: {mt5.last_error()}")
                return None
        # logger.debug(f"Symbol info for {symbol} retrieved.")
        return symbol_info
    except Exception as e: logger.exception(f"Exception get symbol info {symbol}: {e}"); return None

def get_account_info() -> Optional[mt5.AccountInfo]:
    """Gets live account information from MT5."""
    if not is_mt5_connected(): logger.warning("MT5 not connected."); return None
    try:
        account_info = mt5.account_info()
        if account_info is None: logger.error(f"Failed get account info. Error: {mt5.last_error()}"); return None
        return account_info
    except Exception as e: logger.exception(f"Exception get account info: {e}"); return None

def get_tick_info(symbol: str) -> Optional[mt5.Tick]:
    """Gets the latest tick data for a symbol."""
    if not is_mt5_connected(): logger.warning("MT5 not connected."); return None
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
             if mt5.symbol_select(symbol, True):
                 logger.debug(f"Selected {symbol}, retrying tick...")
                 time.sleep(0.1); tick = mt5.symbol_info_tick(symbol)
             if tick is None: logger.error(f"Failed get tick info for {symbol}. Error: {mt5.last_error()}"); return None
        if tick.bid <= 0 or tick.ask <= 0: logger.warning(f"Tick info for {symbol} has invalid prices: Bid={tick.bid}, Ask={tick.ask}")
        # logger.debug(f"Tick info for {symbol}: Bid={tick.bid}, Ask={tick.ask}")
        return tick
    except Exception as e: logger.exception(f"Exception get tick info {symbol}: {e}"); return None

def get_latest_bars(symbol: str, timeframe_str: str, count: int = 200) -> Optional[pd.DataFrame]:
    """Gets the latest N closed bars for a symbol and timeframe from MT5."""
    tf_upper = timeframe_str.upper(); mt5_timeframe = TIMEFRAME_MAP.get(tf_upper)
    if mt5_timeframe is None: logger.error(f"Invalid timeframe: {timeframe_str}"); return None
    if not is_mt5_connected(): logger.warning(f"MT5 not connected."); return None
    try:
        # logger.debug(f"Requesting {count} bars for {symbol} on {tf_upper}...")
        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
        if rates is None:
            if mt5.symbol_select(symbol, True): logger.warning(f"Selected {symbol}, retrying copy_rates..."); time.sleep(0.1); rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            if rates is None: logger.error(f"Failed fetch rates {symbol}/{tf_upper}. Error: {mt5.last_error()}"); return None
        if len(rates) == 0: logger.warning(f"No rates data returned for {symbol}/{tf_upper}."); return pd.DataFrame()

        # logger.debug(f"Retrieved {len(rates)} bars for {symbol}/{tf_upper}.")
        rates_df = pd.DataFrame(rates); rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s', utc=True)
        rates_df.set_index('time', inplace=True); rates_df.columns = [col.lower() for col in rates_df.columns]
        required_cols=['open','high','low','close']; optional_cols=['tick_volume','real_volume','spread']
        present_cols = [c for c in required_cols if c in rates_df.columns]
        if not all(c in present_cols for c in required_cols): logger.error(f"Missing OHLC columns in {symbol} data!"); return None
        final_cols = present_cols + [c for c in optional_cols if c in rates_df.columns]
        return rates_df[final_cols].sort_index(ascending=True)
    except Exception as e: logger.exception(f"Error fetching data {symbol}/{tf_upper}: {e}"); return None

# --- Function with the CORRECTED Type Hint ---
# VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
def get_open_positions(symbol: Optional[str] = None) -> List[Any]: # <-- FIXED: Changed List[mt5.Position] to List[Any]
    """Gets open positions, optionally filtered by symbol."""
    if not is_mt5_connected(): logger.warning("MT5 not connected, cannot get positions."); return []
    try:
        if symbol: positions = mt5.positions_get(symbol=symbol)
        else: positions = mt5.positions_get() # Get all positions if symbol is None

        if positions is None:
             err_code = mt5.last_error()[0]
             # Check for code 10024 which often just means 'no positions found'
             if err_code == 10024:
                  logger.debug(f"No open positions found for {symbol if symbol else 'account'}.")
             else:
                  logger.error(f"Failed to get positions for {symbol if symbol else 'account'}. Error: {mt5.last_error()}")
             return [] # Return empty list on error or no positions

        # MT5 returns tuple, convert to list
        # The objects inside are position objects, but we type hint as Any
        logger.debug(f"Found {len(positions)} open position(s) for {symbol if symbol else 'account'}.")
        return list(positions)
    except Exception as e:
        logger.exception(f"Exception getting positions: {e}")
        return []
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def place_market_order(symbol: str, volume: float, order_type: int,
                       sl_price: Optional[float] = None, tp_price: Optional[float] = None,
                       magic_number: int = 12345, comment: str = "AlphaLive_Trade") -> Optional[mt5.OrderSendResult]:
    """ Places a market order with optional SL and TP using MT5. Returns OrderSendResult object."""
    if not is_mt5_connected(): logger.error("MT5 not connected. Cannot place order."); return None
    # logger.info(f"Attempting market order: {symbol}, Vol={volume}, Type={order_type}, SL={sl_price}, TP={tp_price}")
    symbol_info = get_symbol_info(symbol)
    if symbol_info is None: logger.error(f"Failed get symbol info {symbol}. Cannot place order."); return None

    volume=round(volume,8); min_volume=symbol_info.volume_min; max_volume=symbol_info.volume_max; step_volume=symbol_info.volume_step; digits=symbol_info.digits
    if step_volume > 0: volume = math.floor(volume/step_volume)*step_volume; volume=round(volume,8)
    if volume < min_volume: logger.warning(f"Order vol {volume}<min {min_volume}. Adjusting."); volume=min_volume
    if volume > max_volume: logger.warning(f"Order vol {volume}>max {max_volume}. Adjusting."); volume=max_volume
    if volume <= 0: logger.error(f"Invalid vol {volume} after adjust. Cannot place order."); return None
    # logger.info(f"Validated/Adjusted order volume: {volume}")

    point=symbol_info.point; price=None; tick=get_tick_info(symbol)
    if tick is None: logger.error(f"Failed get tick {symbol}. Cannot place order."); return None
    if order_type == mt5.ORDER_TYPE_BUY: price = tick.ask
    elif order_type == mt5.ORDER_TYPE_SELL: price = tick.bid
    else: logger.error(f"Invalid order type: {order_type}"); return None
    if price is None or price <= 0: logger.error(f"Invalid market price {price} for {symbol}. Cannot place order."); return None

    sl_final=round(sl_price,digits) if sl_price is not None and sl_price>0 else 0.0
    tp_final=round(tp_price,digits) if tp_price is not None and tp_price>0 else 0.0
    buffer=point*2 # Small buffer for SL/TP validation
    if order_type==mt5.ORDER_TYPE_BUY:
        if sl_final!=0.0 and sl_final>=price-buffer: logger.warning(f"BUY SL {sl_final:.{digits}f} invalid vs price {price:.{digits}f}. SL=0."); sl_final=0.0
        if tp_final!=0.0 and tp_final<=price+buffer: logger.warning(f"BUY TP {tp_final:.{digits}f} invalid vs price {price:.{digits}f}. TP=0."); tp_final=0.0
    elif order_type==mt5.ORDER_TYPE_SELL:
        if sl_final!=0.0 and sl_final<=price+buffer: logger.warning(f"SELL SL {sl_final:.{digits}f} invalid vs price {price:.{digits}f}. SL=0."); sl_final=0.0
        if tp_final!=0.0 and tp_final>=price-buffer: logger.warning(f"SELL TP {tp_final:.{digits}f} invalid vs price {price:.{digits}f}. TP=0."); tp_final=0.0

    request={"action":mt5.TRADE_ACTION_DEAL,"symbol":symbol,"volume":float(volume),"type":order_type,"price":float(price),"sl":float(sl_final),"tp":float(tp_final),"deviation":10,"magic":magic_number,"comment":comment,"type_time":mt5.ORDER_TIME_GTC,"type_filling":mt5.ORDER_FILLING_IOC}

    try: logger.info(f"Sending order request: {request}"); result = mt5.order_send(request)
    except Exception as e: logger.exception(f"Exception during order_send {symbol}: {e}"); return None
    if result is None: logger.error(f"order_send None. Error: {mt5.last_error()}"); return None
    logger.info(f"Order send result: Code={result.retcode}, Deal={result.deal}, Order={result.order}, Comment={result.comment}")
    return result

# --- Example usage ---
if __name__ == "__main__":
    if not logging.getLogger().hasHandlers(): logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    logger.info("Testing MT5 Connector...")
    if initialize_mt5():
        logger.info("Init OK.")
        acc_info=get_account_info(); logger.info(f"Account Info: {acc_info}")
        tick=get_tick_info("EURUSD"); logger.info(f"EURUSD Tick: {tick}")
        positions=get_open_positions("EURUSD"); logger.info(f"EURUSD Positions: {positions}")
        all_positions=get_open_positions(); logger.info(f"All Positions: {all_positions}")
        bars=get_latest_bars("EURUSD", "M1", 5); logger.info(f"EURUSD M1 Bars:\n{bars}")
        shutdown_mt5()
    else: logger.error("Init failed.")
