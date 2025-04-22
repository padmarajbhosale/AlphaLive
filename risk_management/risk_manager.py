# risk_management/risk_manager.py
# Added JPY pair tick_value workaround for lot size calculation + DEBUG logs
# Fixed order of operations for JPY workaround
import logging
import sys
import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import math # For lot size rounding

# --- Imports and Logger Setup ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)
try: from config.config_loader import get_config
except ImportError as e: print(f"FATAL ERROR: Import failed: {e}"); logging.basicConfig(level=logging.ERROR); logging.critical(f"Import failed: {e}", exc_info=True); sys.exit(1)
logger = logging.getLogger(__name__)
log_level_str = get_config('LOG_LEVEL', 'INFO').upper()
try: logger.setLevel(log_level_str)
except ValueError: logger.setLevel(logging.INFO); logger.warning(f"Invalid LOG_LEVEL '{log_level_str}', using INFO.")
if not logging.getLogger().hasHandlers(): handler = logging.StreamHandler(); handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')); logger.addHandler(handler); logger.warning("Basic logging handler added in risk_manager.")

# --- Risk Parameters Load ---
try:
     SL_METHOD = get_config('SL_METHOD', 'ATR').upper(); MIN_CONFIDENCE = float(get_config('MIN_CONFIDENCE', 0.60)); MAX_SPREAD_POINTS = int(get_config('MAX_SPREAD_POINTS', 20))
     RISK_PERCENT_PER_TRADE = float(get_config('RISK_PERCENT_PER_TRADE', 0.01)); DEFAULT_SL_POINTS = int(get_config('DEFAULT_SL_POINTS', 150))
     SL_ATR_MULTIPLIER = float(get_config('SL_ATR_MULTIPLIER', 2.0)); DEFAULT_TP_RR_RATIO = float(get_config('DEFAULT_TP_RR_RATIO', 1.5))
except ValueError as e: logger.error(f"Invalid risk param format: {e}. Using defaults."); SL_METHOD = 'ATR'; MIN_CONFIDENCE = 0.60; MAX_SPREAD_POINTS = 20; RISK_PERCENT_PER_TRADE = 0.01; DEFAULT_SL_POINTS = 150; SL_ATR_MULTIPLIER = 2.0; DEFAULT_TP_RR_RATIO = 1.5
if SL_METHOD not in ['ATR', 'POINTS']: logger.warning(f"Invalid SL_METHOD '{SL_METHOD}'. Defaulting 'ATR'."); SL_METHOD = 'ATR'
logger.info(f"Risk settings: SL Method={SL_METHOD}, Min Conf={MIN_CONFIDENCE:.2f}, Max Spread={MAX_SPREAD_POINTS} pts, Risk={RISK_PERCENT_PER_TRADE:.2%}, SL Pts={DEFAULT_SL_POINTS}, SL ATR Mult={SL_ATR_MULTIPLIER}, TP Ratio={DEFAULT_TP_RR_RATIO}")

# --- Risk Check Function ---
def check_trade_conditions(prediction: int, confidence: float, symbol_info: mt5.SymbolInfo, tick_info: mt5.Tick) -> tuple[bool, str]:
    # ... (remains the same) ...
    symbol_name = symbol_info.name if symbol_info else "Unknown"; logger.debug(f"Checking conditions {symbol_name}...")
    if not all([symbol_info, tick_info]): return False, "Missing market info."
    if confidence < MIN_CONFIDENCE: reason=f"Conf {confidence:.4f} < {MIN_CONFIDENCE:.4f}"; logger.info(f"FAIL {symbol_name}: {reason}"); return False, reason
    logger.debug(f"PASS {symbol_name}: Confidence.")
    spread_points = symbol_info.spread
    if spread_points > MAX_SPREAD_POINTS: reason=f"Spread {spread_points} > {MAX_SPREAD_POINTS} pts"; logger.info(f"FAIL {symbol_name}: {reason}"); return False, reason
    logger.debug(f"PASS {symbol_name}: Spread."); logger.debug("Placeholder: Check max trades...")
    logger.info(f"Conditions MET for {symbol_name}"); return True, "Conditions met"


# --- Parameter Calculation Function (Corrected JPY Workaround Order) --- <<< MODIFIED >>>
def calculate_trade_parameters(
    symbol_info: mt5.SymbolInfo, account_info: mt5.AccountInfo, trade_type: int,
    entry_price: float, latest_atr: float | None
    ) -> dict | None:
    symbol_name = symbol_info.name if symbol_info else "UnknownSymbol"; logger.debug(f"Calculating params for {symbol_name} (SL Method: {SL_METHOD})")
    if not all([symbol_info, account_info]) or entry_price <= 0: logger.error("Missing info/price."); return None
    point = symbol_info.point; digits = symbol_info.digits
    if point <= 0: logger.error(f"Invalid point {point} for {symbol_name}."); return None

    # 1. Calculate SL Distance & Price
    sl_distance_price = 0.0; logger.info(f"Calculating SL using: {SL_METHOD}")
    if SL_METHOD == 'ATR':
        if latest_atr is None or pd.isna(latest_atr) or latest_atr <= 0: logger.error(f"Invalid ATR ({latest_atr}) for SL."); return None
        sl_distance_price = SL_ATR_MULTIPLIER * latest_atr; logger.info(f"Using ATR SL: Dist={SL_ATR_MULTIPLIER}*ATR({latest_atr:.{digits}f})={sl_distance_price:.{digits}f}")
    elif SL_METHOD == 'POINTS':
        sl_distance_points = DEFAULT_SL_POINTS; sl_distance_price = sl_distance_points * point; logger.info(f"Using Points SL: Dist={sl_distance_points} pts={sl_distance_price:.{digits}f}")
    else: logger.error(f"Unknown SL_METHOD '{SL_METHOD}'."); return None
    if sl_distance_price <= 0: logger.error(f"SL distance ({sl_distance_price}) <= 0."); return None
    if trade_type == mt5.ORDER_TYPE_BUY: stop_loss_price = entry_price - sl_distance_price
    elif trade_type == mt5.ORDER_TYPE_SELL: stop_loss_price = entry_price + sl_distance_price
    else: logger.error(f"Invalid trade_type '{trade_type}' for SL."); return None
    stop_loss_price = round(stop_loss_price, digits); logger.info(f"Calculated SL price: {stop_loss_price:.{digits}f}")

    # 2. Calculate Lot Size
    lot_size = None
    try:
        # a. Risk Amount
        balance = account_info.balance; risk_amount = balance * RISK_PERCENT_PER_TRADE
        if risk_amount <= 0: raise ValueError("Risk amount <= 0.")
        account_currency = account_info.currency
        logger.debug(f"LOT_CALC_DEBUG: Balance={balance:.2f}, Risk%={RISK_PERCENT_PER_TRADE:.2%}, RiskAmt={risk_amount:.2f} {account_currency}")

        # b. Value of SL per Lot in Account Currency
        contract_size=symbol_info.trade_contract_size; tick_value=symbol_info.trade_tick_value; tick_size=symbol_info.trade_tick_size
        profit_currency = symbol_info.currency_profit
        logger.debug(f"LOT_CALC_DEBUG: Symbol={symbol_name}, Pt={point}, TickSz={tick_size}, TickVal={tick_value}, ContSz={contract_size}, ProfCcy={profit_currency}, AccCcy={account_currency}")
        if tick_size<=0 or point<=0 or contract_size<=0 or tick_value==0: raise ValueError(f"Invalid symbol props.")

        # Calculate value per point initially using reported tick_value
        value_per_point_per_lot_profit_ccy = (tick_value / tick_size) * point
        logger.debug(f"LOT_CALC_DEBUG: Initial ValuePerPoint_ProfCcy={value_per_point_per_lot_profit_ccy:.5f} {profit_currency}")

        # --- WORKAROUND for potentially incorrect JPY Tick Values --- <<< Check AND Apply >>>
        if profit_currency == 'JPY' and point == 0.001 and contract_size > 0:
             expected_val_per_point = (contract_size / 100000.0) * 100.0 # Approx 100 JPY per point for 100k lot
             # If reported value is drastically different (e.g., < 50% or > 200% of expected)
             if not (expected_val_per_point * 0.5 < value_per_point_per_lot_profit_ccy < expected_val_per_point * 2.0):
                  logger.warning(f"Reported JPY ValuePerPoint ({value_per_point_per_lot_profit_ccy:.5f}) seems incorrect vs expected ({expected_val_per_point:.1f}). OVERRIDING.")
                  value_per_point_per_lot_profit_ccy = expected_val_per_point # <<< APPLY OVERRIDE
                  logger.debug(f"LOT_CALC_DEBUG: OVERRIDDEN ValuePerPoint_ProfCcy={value_per_point_per_lot_profit_ccy:.5f} {profit_currency}")
        # --- End Workaround ---

        sl_distance_points = round(sl_distance_price / point)
        if sl_distance_points <= 0: raise ValueError(f"SL points <= 0 ({sl_distance_points}).")
        logger.debug(f"LOT_CALC_DEBUG: SL_Distance_Points={sl_distance_points}")

        # --- Calculate SL value in Profit Ccy USING the potentially overridden value_per_point --- <<< CALC MOVED HERE >>>
        sl_value_per_lot_profit_ccy = sl_distance_points * value_per_point_per_lot_profit_ccy
        logger.debug(f"LOT_CALC_DEBUG: SL_ValuePerLot_ProfCcy={sl_value_per_lot_profit_ccy:.5f} {profit_currency}")

        # c. Currency Conversion
        sl_value_per_lot_account_ccy = 0.0
        if profit_currency == account_currency:
            sl_value_per_lot_account_ccy = sl_value_per_lot_profit_ccy
            logger.debug(f"LOT_CALC_DEBUG: SlValuePerLot_AccCcy={sl_value_per_lot_account_ccy:.5f} {account_currency} (No conversion)")
        else:
            # ... (Conversion logic remains the same) ...
            logger.info(f"Attempting currency conversion: {profit_currency} -> {account_currency}"); rate = None; pair1 = f"{profit_currency}{account_currency}"; tick1 = mt5.symbol_info_tick(pair1);
            if tick1 and tick1.bid > 0: rate = tick1.bid; sl_value_per_lot_account_ccy = sl_value_per_lot_profit_ccy * rate; logger.info(f"Using {pair1} Bid={rate}.")
            else:
                pair2 = f"{account_currency}{profit_currency}"; tick2 = mt5.symbol_info_tick(pair2)
                if tick2 and tick2.ask > 0: rate = tick2.ask; sl_value_per_lot_account_ccy = sl_value_per_lot_profit_ccy / rate; logger.info(f"Using {pair2} Ask={rate}.")
                else: logger.error(f"Could not find conversion rate pair ({pair1} or {pair2})."); return None
            logger.debug(f"LOT_CALC_DEBUG: Conversion Rate ({pair1 if tick1 and tick1.bid>0 else pair2}) = {rate}, SlValuePerLot_AccCcy={sl_value_per_lot_account_ccy:.5f} {account_currency}")

        if sl_value_per_lot_account_ccy <= 0: raise ValueError("SL value/lot in account ccy <= 0.")

        # d. Calculate Ideal Lot Size
        ideal_lot_size = risk_amount / sl_value_per_lot_account_ccy
        logger.info(f"Calculated ideal lot size: {ideal_lot_size:.8f}")

        # e/f. Normalize & Validate Lot Size
        volume_step=symbol_info.volume_step; volume_min=symbol_info.volume_min; volume_max=symbol_info.volume_max
        logger.debug(f"LOT_CALC_DEBUG: VolStep={volume_step}, VolMin={volume_min}, VolMax={volume_max}")
        if volume_step <= 0: raise ValueError("Symbol volume_step invalid.");
        lot_size = math.floor(ideal_lot_size / volume_step) * volume_step; lot_size = round(lot_size, 8)
        logger.debug(f"LOT_CALC_DEBUG: LotSize after step floor/rounding: {lot_size}")
        if lot_size < volume_min: logger.warning(f"Calculated lot {lot_size:.8f} below min {volume_min}. Adjusting."); lot_size = volume_min
        if lot_size > volume_max: logger.warning(f"Calculated lot {lot_size:.8f} above max {volume_max}. Adjusting."); lot_size = volume_max
        if lot_size <= 0: raise ValueError(f"Final lot size <= 0 ({lot_size}).")

    except ValueError as ve: logger.error(f"Lot size calc ValueError: {ve}"); return None
    except Exception as e: logger.exception(f"Unexpected error calculating lot size: {e}"); return None
    logger.info(f"Final Calculated Lot Size for {symbol_name}: {lot_size}")

    # 3. Calculate Take Profit Price
    tp_distance_price = sl_distance_price * DEFAULT_TP_RR_RATIO
    if trade_type == mt5.ORDER_TYPE_BUY: take_profit_price = entry_price + tp_distance_price
    elif trade_type == mt5.ORDER_TYPE_SELL: take_profit_price = entry_price - tp_distance_price
    else: logger.error(f"Invalid trade_type '{trade_type}' for TP."); return None
    take_profit_price = round(take_profit_price, digits); logger.info(f"Calculated TP price: {take_profit_price:.{digits}f} (RR: {DEFAULT_TP_RR_RATIO})")

    return {'lot_size': lot_size, 'stop_loss_price': stop_loss_price, 'take_profit_price': take_profit_price}

# --- Test Block ---
if __name__ == '__main__':
    pass # Minimal