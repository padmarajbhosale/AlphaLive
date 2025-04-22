# config/config_loader.py
import os
from dotenv import load_dotenv
import logging

# --- Configure Logging (Basic setup for now) ---
# We might move this to a dedicated logging utility later
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Environment Variables ---
# Determine the path to the .env file (expected in the project root)
# This code assumes config_loader.py is in G:/Alpha1.1/config/
# It goes up one directory (__file__ -> config) then up another (-> G:/Alpha1.1)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(project_root, '.env')

# Load the .env file if it exists
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logging.info(f"Configuration loaded successfully from: {dotenv_path}")
else:
    # If .env is not found, the script will rely on system environment variables
    logging.warning(f".env file not found at {dotenv_path}. Relying on system environment variables.")
    # You might want to raise an error here if the .env file is mandatory

# --- Function to Access Configuration ---
def get_config(variable_name: str, default: str = None) -> str | None:
    """
    Gets a configuration variable from environment variables loaded from .env or the system.

    Args:
        variable_name: The name of the environment variable (e.g., 'DB_SERVER').
        default: The value to return if the variable is not found (defaults to None).

    Returns:
        The value of the environment variable as a string, or the default value.
        Returns None if not found and no default is provided. Logs a warning in that case.
    """
    value = os.getenv(variable_name, default)
    if value is None and default is None:
        # Only log a warning if it's truly missing and no fallback was given
        logging.warning(f"Configuration variable '{variable_name}' not found in environment variables or .env file.")
    return value

# --- Example Usage Section (for testing this file directly) ---
if __name__ == '__main__':
    # This block executes only when you run `python config/config_loader.py`
    print("\n--- Testing Configuration Loading ---")

    # Test retrieving mandatory variables
    db_server = get_config('DB_SERVER')
    mt5_login = get_config('MT5_LOGIN')
    log_dir = get_config('LOG_DIRECTORY')
    log_level = get_config('LOG_LEVEL')
    db_driver = get_config('DB_DRIVER')

    print(f"DB Server: {db_server}")
    print(f"MT5 Login: {mt5_login}")
    print(f"Log Directory: {log_dir}")
    print(f"Log Level: {log_level}")
    print(f"DB Driver: {db_driver}")


    # Test retrieving a non-existent variable (should return None and log warning)
    print("\nTesting non-existent variable:")
    non_existent = get_config('THIS_VAR_DOES_NOT_EXIST')
    print(f"Non Existent Var: {non_existent}")

    # Test retrieving a non-existent variable with a default value
    print("\nTesting non-existent variable with default:")
    non_existent_default = get_config('THIS_VAR_ALSO_DOES_NOT_EXIST', 'fallback_value')
    print(f"Non Existent Var with Default: {non_existent_default}") # Should be 'fallback_value'

    print("\n--- Test Complete ---") 
