# models/regime_predictor.py
# Loads and predicts using Keras or XGBoost models.
# Added functions for BINARY XGBoost model.
# FIXED: SyntaxError in model loading functions.

import logging
import os
import sys
import joblib
import numpy as np
import pandas as pd

# Import ML libs conditionally
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# --- Setup Path and Imports ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config.config_loader import get_config
except ImportError as e:
    print(f"FATAL ERROR: Could not import get_config in regime_predictor.py. Error: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)

# --- Configuration ---
KERAS_MODEL_FILENAME = get_config("MODEL_FILENAME", "model.keras")
KERAS_SCALER_FILENAME = get_config("SCALER_FILENAME", "scaler.pkl")
KERAS_MODEL_PATH = os.path.join(project_root, "models", KERAS_MODEL_FILENAME)
KERAS_SCALER_PATH = os.path.join(project_root, "models", KERAS_SCALER_FILENAME)
KERAS_FEATURES_PATH = KERAS_SCALER_PATH.replace(".pkl", "_features.joblib")

XGB_REGIME_MODEL_FILENAME = get_config("XGB_MODEL_PATH", "models/model_xgboost_regime.ubj")
XGB_REGIME_MODEL_PATH = os.path.join(project_root, XGB_REGIME_MODEL_FILENAME)
XGB_REGIME_LE_PATH = XGB_REGIME_MODEL_PATH.replace(".ubj", "_le.pkl")
XGB_REGIME_FEATURES_PATH = XGB_REGIME_MODEL_PATH.replace(".ubj", "_features.joblib")

XGB_BINARY_MODEL_FILENAME = get_config("XGB_MODEL_PATH", "models/model_xgboost_direction.ubj")
XGB_BINARY_MODEL_PATH = os.path.join(project_root, XGB_BINARY_MODEL_FILENAME)
XGB_BINARY_FEATURES_PATH = XGB_BINARY_MODEL_PATH.replace(".ubj", "_features.joblib")


# --- Global cache ---
_keras_model = None; _keras_scaler = None; _keras_features = None
_xgb_regime_model = None; _xgb_regime_le = None; _xgb_regime_features = None
_xgb_binary_model = None; _xgb_binary_features = None


# --- Keras Functions ---
def load_keras_regime_model(force_reload=False):
    global _keras_model, _keras_scaler, _keras_features
    if _keras_model is not None and _keras_scaler is not None and _keras_features is not None and not force_reload: return _keras_model, _keras_scaler, _keras_features
    if not TF_AVAILABLE: logger.critical("TF/Keras not installed."); return None, None, None
    logger.info(f"Attempting to load Keras artifacts..."); logger.info(f" Keras Model: {KERAS_MODEL_PATH}"); logger.info(f" Keras Scaler: {KERAS_SCALER_PATH}"); logger.info(f" Keras Features: {KERAS_FEATURES_PATH}");
    try:
        model = load_model(KERAS_MODEL_PATH) if os.path.exists(KERAS_MODEL_PATH) else None; scaler = joblib.load(KERAS_SCALER_PATH) if os.path.exists(KERAS_SCALER_PATH) else None; features = joblib.load(KERAS_FEATURES_PATH) if os.path.exists(KERAS_FEATURES_PATH) else None
        if model and scaler and features and isinstance(features, list): _keras_model, _keras_scaler, _keras_features = model, scaler, features; logger.info("Keras artifacts loaded successfully."); return model, scaler, features
        else: logger.error(f"Failed to load all Keras artifacts."); return None, None, None # Simplified error
    except Exception as e: logger.exception(f"Error loading Keras artifacts: {e}"); return None, None, None

def make_keras_regime_prediction(model, scaler, trained_feature_list, latest_features_series: pd.Series):
    # ... (Keep Keras prediction function as before) ...
    if model is None or scaler is None or trained_feature_list is None: return None, None
    if latest_features_series is None or latest_features_series.empty: return None, None
    try:
        missing_features = [f for f in trained_feature_list if f not in latest_features_series.index];
        if missing_features: return None, None
        features_ordered = latest_features_series[trained_feature_list].values.reshape(1, -1)
        if np.isnan(features_ordered).any(): return None, None
        if not np.issubdtype(features_ordered.dtype, np.number): features_ordered = features_ordered.astype(np.float64)
        scaled_features = scaler.transform(features_ordered); probabilities = model.predict(scaled_features, verbose=0)[0]
        if len(probabilities) == 3: predicted_class = np.argmax(probabilities); confidence = probabilities[predicted_class]; return predicted_class, float(confidence)
        elif len(probabilities) == 1: predicted_class = (probabilities > 0.5).astype(int)[0]; confidence=probabilities[0] if predicted_class==1 else 1.0-probabilities[0]; return predicted_class, float(confidence)
        else: logger.error(f"Keras model output shape unexpected: {len(probabilities)}"); return None, None
    except Exception as e: logger.exception(f"Error during Keras prediction: {e}"); return None, None


# --- XGBoost Regime Functions ---
def load_xgboost_regime_model(force_reload=False):
    """ Loads the XGBoost REGIME model, LabelEncoder, and feature list. """
    global _xgb_regime_model, _xgb_regime_le, _xgb_regime_features
    if _xgb_regime_model is not None and _xgb_regime_le is not None and _xgb_regime_features is not None and not force_reload: return _xgb_regime_model, _xgb_regime_le, _xgb_regime_features
    if not XGB_AVAILABLE: logger.critical("XGBoost not installed."); return None, None, None
    logger.info(f"Attempting to load XGBoost REGIME artifacts..."); logger.info(f" XGB Model: {XGB_REGIME_MODEL_PATH}"); logger.info(f" XGB LE: {XGB_REGIME_LE_PATH}"); logger.info(f" XGB Features: {XGB_REGIME_FEATURES_PATH}");
    try:
        model = None; le = None; features = None # Initialize
        model_loaded_ok = False
        if os.path.exists(XGB_REGIME_MODEL_PATH): model = xgb.XGBClassifier(); model.load_model(XGB_REGIME_MODEL_PATH); model_loaded_ok = hasattr(model, 'predict')
        else: logger.error(f"XGBoost REGIME model file not found: {XGB_REGIME_MODEL_PATH}")
        if os.path.exists(XGB_REGIME_LE_PATH): le = joblib.load(XGB_REGIME_LE_PATH)
        else: logger.error(f"XGBoost REGIME LabelEncoder file not found: {XGB_REGIME_LE_PATH}")
        if os.path.exists(XGB_REGIME_FEATURES_PATH): features = joblib.load(XGB_REGIME_FEATURES_PATH)
        else: logger.error(f"XGBoost REGIME Feature List file not found: {XGB_REGIME_FEATURES_PATH}")

        if model_loaded_ok and le and features and isinstance(features, list):
            _xgb_regime_model, _xgb_regime_le, _xgb_regime_features = model, le, features
            logger.info("XGBoost REGIME artifacts loaded successfully.")
            return model, le, features
        else:
             # <<< SYNTAX FIXED HERE >>>
             logger.error(f"Failed to load all XGBoost REGIME artifacts. Check logs above.")
             if model and not model_loaded_ok: # Check if model object exists but seems invalid
                  logger.error("Loaded XGB REGIME model object appears invalid.")
             return None, None, None
    except Exception as e: logger.exception(f"Error loading XGBoost REGIME artifacts: {e}"); return None, None, None

def make_xgboost_regime_prediction(model, le, trained_feature_list, latest_features_series: pd.Series):
    # ... (Keep XGBoost Regime prediction function as before) ...
    if model is None or le is None or trained_feature_list is None: return None, None
    if latest_features_series is None or latest_features_series.empty: return None, None
    try:
        missing_features = [f for f in trained_feature_list if f not in latest_features_series.index];
        if missing_features: logger.debug(f"Missing features for XGBoost Regime pred: {missing_features}."); return None, None
        features_ordered = latest_features_series[trained_feature_list].astype(float).values.reshape(1, -1) # Ensure float
        if np.isnan(features_ordered).any(): return None, None
        probabilities = model.predict_proba(features_ordered)[0]
        if len(probabilities) == len(le.classes_): predicted_class_index = np.argmax(probabilities); confidence = probabilities[predicted_class_index]; return predicted_class_index, float(confidence)
        else: logger.error(f"XGBoost model output shape unexpected."); return None, None
    except Exception as e: logger.exception(f"Error during XGBoost Regime prediction: {e}"); return None, None


# --- XGBoost Binary Direction Functions ---
def load_xgboost_binary_model(force_reload=False):
    """ Loads the XGBoost binary direction model and feature list. """
    global _xgb_binary_model, _xgb_binary_features
    if _xgb_binary_model is not None and _xgb_binary_features is not None and not force_reload: return _xgb_binary_model, _xgb_binary_features
    if not XGB_AVAILABLE: logger.critical("XGBoost not installed."); return None, None
    logger.info(f"Attempting to load XGBoost BINARY artifacts..."); logger.info(f" XGB Model: {XGB_BINARY_MODEL_PATH}"); logger.info(f" XGB Features: {XGB_BINARY_FEATURES_PATH}");
    try:
        model = None; features = None # Initialize
        model_loaded_ok = False
        if os.path.exists(XGB_BINARY_MODEL_PATH): model = xgb.XGBClassifier(); model.load_model(XGB_BINARY_MODEL_PATH); model_loaded_ok = hasattr(model, 'predict_proba')
        else: logger.error(f"XGBoost BINARY model file not found: {XGB_BINARY_MODEL_PATH}")
        if os.path.exists(XGB_BINARY_FEATURES_PATH): features = joblib.load(XGB_BINARY_FEATURES_PATH)
        else: logger.error(f"XGBoost BINARY Feature List file not found: {XGB_BINARY_FEATURES_PATH}")

        if model_loaded_ok and features and isinstance(features, list):
            _xgb_binary_model, _xgb_binary_features = model, features
            logger.info("XGBoost BINARY artifacts loaded successfully.")
            return model, features
        else:
             # <<< SYNTAX FIXED HERE >>>
             logger.error(f"Failed to load all XGBoost BINARY artifacts. Check logs above.")
             if model and not model_loaded_ok: # Check if model object exists but seems invalid
                  logger.error("Loaded XGB BINARY model object appears invalid.")
             return None, None
    except Exception as e: logger.exception(f"Error loading XGBoost BINARY artifacts: {e}"); return None, None

def make_xgboost_binary_prediction(model, trained_feature_list, latest_features_series: pd.Series):
    """ Returns probability of class 1 ('Up'). """
    if model is None or trained_feature_list is None: return None
    if latest_features_series is None or latest_features_series.empty: return None
    try:
        missing_features = [f for f in trained_feature_list if f not in latest_features_series.index];
        if missing_features: logger.debug(f"Missing features for XGBoost Binary: {missing_features}."); return None
        features_ordered = latest_features_series[trained_feature_list].astype(float).values.reshape(1, -1)
        if np.isnan(features_ordered).any(): logger.warning(f"NaN values found in features before XGBoost prediction."); return None
        probabilities = model.predict_proba(features_ordered)[0]
        if len(probabilities) == 2: return float(probabilities[1]) # Return probability of class 1
        else: logger.error(f"XGBoost binary model output shape unexpected: {len(probabilities)}"); return None
    except Exception as e: logger.exception(f"Error during XGBoost binary prediction: {e}"); return None