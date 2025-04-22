# models/xgboost_trainer.py
# UPDATED: To train on the binary 'target_direction' variable.
# WORKAROUND: Commented out early_stopping_rounds due to persistent TypeError.

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import os
import sys
import logging

# --- Setup Path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Setup Logging ---
try:
    from utils.logging_config import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Initialized logging via utils.logging_config.")
except Exception as log_e:
    print(f"Logging setup error: {log_e}. Basic config used.")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    logger = logging.getLogger(__name__)

# --- Import Config Loader ---
try:
    from config.config_loader import get_config
except ImportError as e:
    logger.critical(f"FATAL ERROR: Could not import get_config. Error: {e}")
    sys.exit(1)

# --- Configuration ---
logger.info("Loading configuration for XGBoost training...")
INPUT_FEATURE_FILE = get_config("TRAINING_FEATURE_FILE", "data/EURUSD_M5_model_ready_features.csv")
XGB_MODEL_PATH = get_config("XGB_MODEL_PATH", "models/model_xgboost_direction.ubj")
FEATURES_PATH = XGB_MODEL_PATH.replace(".ubj", "_features.joblib")

TEST_SIZE = float(get_config("TRAIN_TEST_SIZE", 0.2))
RANDOM_SEED = int(get_config("TRAIN_RANDOM_SEED", 42))

# --- XGBoost Hyperparameters ---
N_ESTIMATORS = int(get_config("XGB_N_ESTIMATORS", 100))
MAX_DEPTH = int(get_config("XGB_MAX_DEPTH", 3))
LEARNING_RATE = float(get_config("XGB_LEARNING_RATE", 0.1))
# EARLY_STOPPING_ROUNDS = int(get_config("XGB_EARLY_STOPPING", 10)) # Keep variable if needed later, but don't pass to fit for now

logger.info(f"Input Feature File: {INPUT_FEATURE_FILE}")
logger.info(f"XGBoost Model Output Path: {XGB_MODEL_PATH}")
logger.info(f"Features List Output Path: {FEATURES_PATH}")
logger.info(f"XGBoost Params: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, learning_rate={LEARNING_RATE}") # Removed early stopping from log


# --- Main Training Logic ---
def train_xgboost_binary():
    # 1. Load Data
    logger.info(f"📥 Loading data from: {INPUT_FEATURE_FILE}")
    try:
        df = pd.read_csv(INPUT_FEATURE_FILE, index_col='time', parse_dates=True)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError: logger.error(f"Input file not found: {INPUT_FEATURE_FILE}"); return
    except Exception as e: logger.exception(f"Error loading data: {e}"); return

    # 2. Prepare Features (X) and Target (y)
    logger.info("Preparing features and target...")
    target_col = 'target_direction'
    if target_col not in df.columns: logger.error(f"'{target_col}' column not found."); return

    y = df[target_col]
    X = df.drop(columns=[target_col, 'Return_Label'], errors='ignore')

    # Ensure features are numeric
    logger.info(f"Initial features for XGBoost: {X.columns.tolist()}")
    cols_before = set(X.columns); X = X.apply(pd.to_numeric, errors='coerce'); X.dropna(axis=1, how='any', inplace=True)
    cols_after = set(X.columns); dropped_cols = cols_before - cols_after
    if dropped_cols: logger.warning(f"Dropped non-numeric or NaN columns from features: {list(dropped_cols)}")
    logger.info(f"Final features for XGBoost training: {X.columns.tolist()}")
    if X.empty: logger.error("Feature DataFrame 'X' is empty after processing."); return

    # 3. Train/Test Split
    logger.info(f"Splitting data (Test size: {TEST_SIZE}, Random state: {RANDOM_SEED})")
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        logger.info(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
        logger.info(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")
    except Exception as e: logger.exception(f"Error during train/test split: {e}"); return

    # 4. Initialize and Train XGBoost Model
    logger.info("Initializing XGBoost Classifier for BINARY classification...")
    try: scale_pos_weight = sum(y_train == 0) / sum(y_train == 1); logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
    except ZeroDivisionError: logger.warning("Using default scale_pos_weight=1."); scale_pos_weight = 1

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    logger.info("🚀 Training XGBoost model...")
    try:
        # <<< WORKAROUND: Removed early_stopping_rounds from fit() call >>>
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            # early_stopping_rounds=EARLY_STOPPING_ROUNDS, # Temporarily Commented Out
            verbose=False # Set to True to see round-by-round progress if needed
        )
        logger.info("Model training complete.")
        # Note: Best iteration info might not be available without early stopping

    except Exception as e:
        logger.exception(f"Error during XGBoost model fitting: {e}")
        return

    # 5. Evaluate Model
    logger.info("\n📊 Evaluating model on test set...")
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred_class = (y_pred_proba > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred_class)
        auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred_class, target_names=['NotUp(0)', 'Up(1)'])

        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test AUC: {auc:.4f}")
        print("\n📊 Classification Report on Test Set:")
        print(report)
    except Exception as e: logger.exception(f"Error during model evaluation: {e}"); return

    # 6. Save Model and Feature List
    logger.info("💾 Saving artifacts...")
    try:
        os.makedirs(os.path.dirname(XGB_MODEL_PATH), exist_ok=True)
        model.save_model(XGB_MODEL_PATH)
        logger.info(f"✅ XGBoost model saved: {XGB_MODEL_PATH}")
        feature_list = X.columns.tolist()
        joblib.dump(feature_list, FEATURES_PATH)
        logger.info(f"✅ Trained feature list saved: {FEATURES_PATH}")
    except Exception as e: logger.exception(f"Error saving model artifacts: {e}")

    logger.info("XGBoost training process finished.")


if __name__ == "__main__":
    train_xgboost_binary()