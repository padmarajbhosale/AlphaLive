# G:/AlphaLive/main.py
# Main entry point for the live bot

import sys
import os
import time
import logging
import threading
import MetaTrader5 as mt5 # Need this for is_mt5_connected check

# --- Setup Path ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path: sys.path.insert(0, project_root)

# --- Setup Logging ---
try:
    from utils.logging_config import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
except Exception as log_e:
    # Fallback if logging setup fails
    print(f"FATAL ERROR during logging setup: {log_e}")
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    logging.critical(f"Logging setup exception: {log_e}", exc_info=True)
    logger = logging.getLogger(__name__) # Assign after basicConfig
    logger.error("Fell back to basic logging config due to setup error.")

# --- Import Core Components ---
try:
    from shared_state import set_state, get_state
    from trading_engine.mt5_connector import initialize_mt5, shutdown_mt5, is_mt5_connected
    from trading_engine.engine import run_trading_loop
    # Optional: Import Telegram bot thread function if used
    try:
        from telegram_interface.bot_handler import run_bot_in_thread
        TELEGRAM_ENABLED = True
    except ImportError:
        logger.warning("Telegram interface not found or failed to import. Running without Telegram bot.")
        TELEGRAM_ENABLED = False
        # Define a dummy function if Telegram is disabled to avoid errors later if called
        def run_bot_in_thread(): logger.info("Telegram disabled, dummy thread function called.")

except ImportError as e:
    logger.critical(f"Core module import failed: {e}", exc_info=True)
    sys.exit(1)


# --- Main Application Logic ---
def main():
    logger.info("="*10 + " Starting Trading Bot Application " + "="*10)
    # Initialize shared state flags
    set_state("is_running", False)
    set_state("is_paused", False)
    set_state("mt5_connected", False)
    set_state("last_error", None)
    set_state("close_all_requested", False) # Initialize if used by telegram

    mt5_initialized_successfully = False
    telegram_thread = None

    try:
        # --- Initialize MT5 Connection ---
        logger.info("Initializing MT5 connection...")
        logger.info("Please ensure the MetaTrader 5 terminal is running and logged into the LIVE/DEMO account.")
        logger.info("Ensure 'Allow Algo Trading' is enabled in MT5 options.")
        if initialize_mt5(): # Uses function from mt5_connector
            logger.info("MT5 initialization successful.")
            set_state("mt5_connected", True)
            mt5_initialized_successfully = True
        else:
            logger.error("MT5 initialization failed. Please check terminal and configuration. Exiting.")
            set_state("mt5_connected", False)
            # TODO: Optionally send an alert here via another method if possible
            return # Exit if MT5 connection fails

        # --- Start Optional Telegram Bot Thread ---
        if TELEGRAM_ENABLED:
            logger.info("Starting Telegram bot thread...")
            telegram_thread = threading.Thread(target=run_bot_in_thread, name="TelegramBotThread", daemon=True)
            telegram_thread.start()
            # Check if thread started successfully (is_alive() check immediately after start might be unreliable)
            time.sleep(1) # Give thread a moment to start
            if telegram_thread.is_alive():
                 logger.info("Telegram bot thread appears active.")
            else:
                 logger.error("Failed to start Telegram bot thread (check bot_handler logs/token).")
                 # Decide if this is critical
        else:
             logger.info("Telegram bot is disabled.")

        # --- Start Main Trading Engine Loop ---
        logger.info("Starting main trading engine loop...")
        set_state("is_running", True) # Set running flag before starting loop
        # Wrap the engine loop call in its own try/except to catch engine-specific crashes
        try:
            run_trading_loop() # This call will block until the loop exits or an exception occurs
            logger.info("Trading loop function finished normally.") # Only reached if loop breaks normally
        except Exception as engine_e:
             logger.exception(f"CRITICAL ERROR inside trading engine loop: {engine_e}")
             set_state("last_error", f"Critical Engine Loop Error: {engine_e}")
             # Optional: Send alert here too
             # Decide if the whole application should exit on engine crash
             # raise engine_e # Re-raise to trigger main finally block

    except KeyboardInterrupt:
        logger.info("Shutdown requested via KeyboardInterrupt (Ctrl+C).")
        set_state("is_running", False) # Signal threads to stop
    except Exception as main_e:
        # Catch unexpected errors in the main setup/flow
        logger.exception(f"CRITICAL UNHANDLED ERROR in main execution flow: {main_e}")
        set_state("last_error", f"Critical Main Error: {main_e}")
        set_state("is_running", False) # Signal threads to stop
    finally:
        # --- Shutdown Sequence ---
        set_state("is_running", False) # Ensure flag is false
        logger.info("Initiating shutdown sequence...")

        # Wait a moment for loop/threads to potentially react to flag
        time.sleep(2)

        # Shutdown MT5 connection if it was initialized
        if mt5_initialized_successfully:
            logger.info("Attempting to shut down MT5 connection...")
            # Use the checker function before calling shutdown
            if is_mt5_connected():
                shutdown_mt5() # Uses function from mt5_connector
                logger.info("MT5 shutdown command sent.")
            else:
                logger.warning("MT5 connection seems lost or already shut down, skipping shutdown command.")
        else:
            logger.info("MT5 was not initialized, skipping shutdown.")

        # Check Telegram thread status
        if telegram_thread and telegram_thread.is_alive():
            logger.info("Telegram thread is still alive (will exit as daemon). Consider adding graceful shutdown if needed.")

        logger.info("="*10 + " Trading Bot Application Stopped " + "="*10)

# --- Script Entry Point ---
if __name__ == "__main__":
    main()
