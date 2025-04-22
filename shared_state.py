# shared_state.py
import threading
import logging

logger = logging.getLogger(__name__)

# === Shared Bot State (Control Flags) ===
bot_state = {
    "is_running": False,
    "is_paused": False,
    "mt5_connected": False,
    "last_error": None,
    "close_all_requested": False,
    "bot_event_loop": None,
    "telegram_bot_instance": None
}

state_lock = threading.Lock()

def get_state(key, default=None):
    with state_lock:
        return bot_state.get(key, default)

def set_state(key, value):
    with state_lock:
        bot_state[key] = value

def get_all_state():
    with state_lock:
        return dict(bot_state)

# === Shared Feature State (Per Symbol) ===
# This is updated in real-time by your data pipeline or MT5 fetch logic
live_feature_state = {}
feature_lock = threading.Lock()

def update_symbol_features(symbol: str, features: dict):
    with feature_lock:
        live_feature_state[symbol] = features
        logger.debug(f"🔄 Updated features for {symbol}: {features}")

def get_latest_features(symbol: str):
    with feature_lock:
        return live_feature_state.get(symbol, {
            'Regime': 0,            # Default to Ranging
            'BOS_Signal': 0,
            'OB_Present': 0,
            'OB_Type_Encoded': 0
        })

def get_symbol_list():
    # You can later fetch this dynamically from MT5 or config
    return ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']

def get_config():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    return {
        'model_path': os.getenv("MODEL_PATH"),
        'scaler_path': os.getenv("SCALER_PATH"),
        'meta_model_path': os.getenv("META_MODEL_PATH"),
        'meta_scaler_path': os.getenv("META_SCALER_PATH"),
        'telegram_token': os.getenv("TELEGRAM_BOT_TOKEN"),
        'chat_id': os.getenv("ALERT_CHAT_ID"),
    }
