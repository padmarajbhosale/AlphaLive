# G:/AlphaLive/trading_engine/trade_executor.py
# Modified to accept pre-calculated parameters
# CORRECTED: Added 'import time' and changed filling mode to FOK

import MetaTrader5 as mt5
import logging
import os
import sys
import math # Keep math for rounding if needed
import time # <--- ADDED THIS IMPORT

# --- Setup Path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

# --- Get Config/Defaults ---
try:
    from config.config_loader import get_config
except ImportError:
     # Fallback if config loader not found (e.g., standalone testing)
     def get_config(key, default=None): return os.getenv(key, default)

# Load defaults or config values
MAGIC_NUMBER = int(get_config("MAGIC_NUMBER", 777999)) # Use a configurable magic number
ORDER_COMMENT = get_config("ORDER_COMMENT", "AlphaLive_Bot_v1") # Use a configurable comment
DEFAULT_DEVIATION = int(get_config("MT5_DEVIATION", 10)) # Allowable slippage points

def execute_trade(
    symbol: str,
    trade_type: int, # mt5.ORDER_TYPE_BUY or mt5.ORDER_TYPE_SELL
    entry_price: float, # The price used for order request (current ask/bid)
    volume: float, # Calculated lot size from risk manager
    sl_price: float, # Calculated stop loss price (0.0 if none)
    tp_price: float # Calculated take profit price (0.0 if none)
    ) -> dict: # Return dict indicating success/failure/details
    """
    Executes a market order using pre-calculated parameters from risk manager.

    Args:
        symbol (str): Trading symbol.
        trade_type (int): mt5.ORDER_TYPE_BUY or mt5.ORDER_TYPE_SELL.
        entry_price (float): The current Ask (for BUY) or Bid (for SELL) to use in the request.
        volume (float): The calculated lot size, already validated for min/max/step.
        sl_price (float): The calculated stop loss price, already rounded (0.0 for none).
        tp_price (float): The calculated take profit price, already rounded (0.0 for none).

    Returns:
        dict: Contains 'success' (bool) and other details like 'error', 'ticket', 'deal', 'retcode'.
    """
    try:
        # Log the validated and rounded parameters received
        logger.info(f"Executing trade request: {symbol}, Type={trade_type}, Vol={volume:.2f}, EntryReq={entry_price:.5f}, SL={sl_price:.5f}, TP={tp_price:.5f}")

        # --- Basic Validations (Should have been caught by risk manager, but double check) ---
        if volume <= 0:
            logger.error(f"Invalid volume ({volume}) passed to execute_trade.")
            return {"success": False, "error": "Invalid volume <= 0"}
        if entry_price <= 0:
             logger.error(f"Invalid entry price ({entry_price}) passed to execute_trade.")
             return {"success": False, "error": "Invalid entry price <= 0"}
        # Assume sl_price/tp_price are correctly formatted (0.0 or valid price) by risk manager

        # --- Construct Order Request ---
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume), # Ensure float
            "type": trade_type,
            "price": float(entry_price), # Market price for execution
            "sl": float(sl_price), # Use calculated SL (0.0 if none)
            "tp": float(tp_price), # Use calculated TP (0.0 if none)
            "deviation": DEFAULT_DEVIATION, # Slippage allowed in points
            "magic": MAGIC_NUMBER,
            "comment": ORDER_COMMENT,
            # --- Changed Filling Mode ---
            "type_filling": mt5.ORDER_FILLING_FOK, # Changed from IOC to FOK
            # --- -------------------- ---
            "type_time": mt5.ORDER_TIME_GTC, # Good Till Cancelled
        }

        # --- Send Order ---
        # Ensure MT5 is initialized before sending (should be handled by main.py/engine)
        if not mt5.terminal_info():
             logger.error("MT5 connection lost or not initialized before order send.")
             return {"success": False, "error": "MT5 connection unavailable"}

        logger.info(f"📤 Sending order request: {request}")
        start_time = time.time() # Should work now with 'import time' above
        result = mt5.order_send(request)
        end_time = time.time()
        logger.info(f"MT5 order_send took {end_time - start_time:.3f} seconds.")


        # --- Process Result ---
        if result is None:
            # This case should be rare if MT5 connection is active
            last_error = mt5.last_error()
            logger.error(f"❌ MT5 order_send returned None object. Last Error: {last_error}")
            return {"success": False, "error": f"order_send None: {last_error}"}

        # Check return code for success
        successful_codes = [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED, mt5.TRADE_RETCODE_DONE_PARTIAL]

        if result.retcode not in successful_codes:
            logger.error(f"❌ Order Send Failed: Code={result.retcode}, Comment={result.comment}")
            logger.error(f"Failed Request Details: {request}") # Log the failing request
            # If the specific error is "unsupported filling mode", log advice
            if "unsupported filling mode" in result.comment.lower():
                logger.warning("Suggestion: The 'ORDER_FILLING_FOK' mode is not supported by the broker/symbol. Try 'ORDER_FILLING_RETURN' next.")
            return {"success": False, "error": result.comment, "code": result.retcode}

        # Success (or partial success / placed)
        logger.info(f"✅ Order Send Success/Placed: Code={result.retcode}, Comment={result.comment}, Order={result.order}, Deal={result.deal}")
        deal_ticket = result.deal if hasattr(result, 'deal') else 'N/A'
        order_ticket = result.order if hasattr(result, 'order') else 'N/A'
        fill_price = result.price if hasattr(result, 'price') else entry_price
        fill_volume = result.volume if hasattr(result, 'volume') else volume

        logger.info(f"  --> Deal: {deal_ticket}, Order: {order_ticket}, FillPrice: {fill_price:.{5}f}, FillVolume: {fill_volume:.2f}")

        return {
            "success": True,
            "symbol": symbol,
            "type": trade_type,
            "req_price": entry_price,
            "sl": sl_price,
            "tp": tp_price,
            "req_volume": volume,
            "ticket": order_ticket,
            "deal": deal_ticket,
            "retcode": result.retcode,
            "comment": result.comment,
            "fill_price": fill_price,
            "fill_volume": fill_volume,
            "time_msc": result.time_msc if hasattr(result, 'time_msc') else None,
        }

    except Exception as e:
        # Log the exception with traceback
        logger.exception(f"⚠️ Exception during trade execution for {symbol}: {e}")
        return {"success": False, "error": f"Exception: {str(e)}"}