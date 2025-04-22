# utils/logging_config.py
import logging
import logging.handlers
import os
import sys

# --- Dynamic Import for Configuration ---
# Ensure the project root (G:/Alpha1.1) is in the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

config_loaded = False
try:
    from config.config_loader import get_config
    config_loaded = True
except ImportError as e:
    print(f"WARNING: Could not import get_config from config.config_loader in logging_config.py. Error: {e}")
    print("Logging will use default settings.")
    # Define defaults if config cannot be loaded
    LOG_LEVEL = 'INFO'
    LOG_DIRECTORY = './logs'
    LOG_FILENAME = 'bot.log'
else:
   # Load config if import was successful
   LOG_LEVEL = get_config('LOG_LEVEL', 'INFO').upper()
   LOG_DIRECTORY = get_config('LOG_DIRECTORY', './logs')
   # Define log filename (can be made configurable too if needed)
   LOG_FILENAME = 'bot.log'


# Construct full log file path relative to project root
log_file_path = os.path.join(project_root, LOG_DIRECTORY, LOG_FILENAME)

def setup_logging():
    """Configures the root logger for the application."""

    try:
        # Ensure the log directory exists
        log_dir_abs = os.path.join(project_root, LOG_DIRECTORY)
        os.makedirs(log_dir_abs, exist_ok=True)
        # print(f"Checked/Created log directory: {log_dir_abs}") # Optional: for verification

        # --- Define Log Format ---
        log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(log_format, datefmt=date_format)

        # --- Get Root Logger ---
        # Configure the root logger - other loggers will inherit this
        root_logger = logging.getLogger()

        # --- Set Level ---
        # Set the threshold for the logger. Messages below this level will be ignored.
        level = logging.getLevelName(LOG_LEVEL) # Get numeric level from name
        if not isinstance(level, int):
            print(f"Invalid LOG_LEVEL '{LOG_LEVEL}' in config. Defaulting to INFO.")
            level = logging.INFO
        root_logger.setLevel(level)

        # --- Clear Existing Handlers (Avoid Duplicates) ---
        if root_logger.hasHandlers():
            # print("Root logger already has handlers. Clearing them to reconfigure.")
            # Be careful clearing handlers if other libraries might also configure logging
            # For a self-contained app, clearing is usually safe on initial setup.
            for handler in root_logger.handlers[:]: # Iterate over a copy
               root_logger.removeHandler(handler)
               handler.close() # Close handlers before removing


        # --- Console Handler ---
        console_handler = logging.StreamHandler(sys.stdout) # Log to standard output
        console_handler.setLevel(level) # Console shows messages at this level or higher
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # --- Rotating File Handler ---
        # Rotates log file when it reaches a certain size.
        # maxBytes=5*1024*1024 means 5 MB per file
        # backupCount=3 means keep bot.log, bot.log.1, bot.log.2
        max_bytes = 5 * 1024 * 1024 # 5 MB
        backup_count = 2 # Keep 2 backups + the current file
        # Use 'w' mode to start fresh log on each run? Or 'a' to append? 'a' is typical.
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path, mode='a', maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
        )
        file_handler.setLevel(level) # File logs messages at this level or higher
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging configured: Level={LOG_LEVEL}, Path={log_file_path}")

    except Exception as e:
        # Fallback basic config if setup fails catastrophically
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        logging.exception(f"!!! FAILED TO CONFIGURE CUSTOM LOGGING: {e}. Falling back to basic config. !!!")


# --- Example Usage Section (for testing this file directly) ---
if __name__ == "__main__":
    print("\n--- Setting up and Testing Logging ---")
    setup_logging() # Configure logging

    # Get a logger instance specific to this test module
    test_logger = logging.getLogger(__name__) # Use module name

    print("\n--- Logging Test Messages (Check console AND log file) ---")
    # These will log using the root logger configuration we set up
    test_logger.debug("This is a debug message. (Should only appear in console/file if LOG_LEVEL=DEBUG)")
    test_logger.info("This is an info message.")
    test_logger.warning("This is a warning message.")
    test_logger.error("This is an error message.")
    test_logger.critical("This is a critical message.")

    print("\n--- Testing Exception Logging ---")
    try:
        result = 1 / 0 # Simulate an error
    except ZeroDivisionError:
        # logger.exception automatically includes traceback information
        test_logger.exception("An error occurred (testing exception logging).")

    print("\n--- Logging Test Complete ---")
    print(f"Check the console output above.")
    print(f"Also check the log file located at: {log_file_path}")
    print(f"If LOG_LEVEL is INFO (default), you should see INFO, WARNING, ERROR, CRITICAL messages there.")