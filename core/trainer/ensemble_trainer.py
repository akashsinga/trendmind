#core/trainer/ensemble_trainer.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from core.config import DAILY_PROCESSED_PATH, DATA_DIR, TARGET_COLUMN, MODEL_DIR
from core.features.feature_engineer import create_features
from core.utils.load_multiple_bhavcopies import load_multiple_bhavcopies

# Paths
MODEL_PATH = os.path.join(MODEL_DIR, "ensemble_model.pkl")
IMPORTANCE_CSV = "outputs/ensemble_feature_importance.csv"

def run_ensemble_training():
    print("[INFO] Loading data...")
    df = load_multiple_bhavcopies(DATA_DIR)
    print(f"[INFO] Loaded {len(df)} rows of raw bhavcopy data")

    print("[INFO] Creating features...")
    features = create_features(df, predict_mode=False)
    print(f"[INFO] Processed dataset has {len(features)} rows")
    
    os.makedirs(os.path.dirname(DAILY_PROCESSED_PATH), exist_ok=True)
    features.to_csv(DAILY_PROCESSED_PATH, index=False)
    print(f"[INFO] Saved processed data to {DAILY_PROCESSED_PATH}")
    
    df = features

    if df.empty:
        print("[ERROR] Feature generation failed. No data to train on.")
        return

    X = df.drop(columns=["date", "symbol", TARGET_COLUMN], errors="ignore")
    y = df[TARGET_COLUMN]

    print("[INFO] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Drop rows with NaNs from training data
    nan_mask = X_train.notna().all(axis=1)
    X_train = X_train[nan_mask]
    y_train = y_train[nan_mask]

    print("[INFO] Training base models...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lgbm = LGBMClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("lgbm", lgbm), ("lr", lr)],
        voting="soft"
    )

    print("[INFO] Fitting ensemble model...")
    ensemble.fit(X_train, y_train)

    # Evaluate on test set
    # Drop NaNs from test set
    nan_mask_test = X_test.notna().all(axis=1)
    X_test_clean = X_test[nan_mask_test]
    y_test_clean = y_test[nan_mask_test]

    # Predict and evaluate
    y_pred = ensemble.predict(X_test_clean)

    print("\n[RESULTS]")
    print(f"Accuracy:  {accuracy_score(y_test_clean, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test_clean, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test_clean, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test_clean, y_pred):.4f}")

    print(f"\n[INFO] Saving model to {MODEL_PATH}")
    joblib.dump(ensemble, MODEL_PATH)

    # Retrieve fitted base models
    rf_fitted = ensemble.named_estimators_["rf"]
    lgbm_fitted = ensemble.named_estimators_["lgbm"]

    feature_names = list(X.columns)

    print("\n[INFO] Feature Importances from RandomForest:")
    rf_importances = rf_fitted.feature_importances_
    for name, imp in sorted(zip(feature_names, rf_importances), key=lambda x: -x[1]):
        print(f"{name:<30} {imp:.4f}")

    print("\n[INFO] Feature Importances from LightGBM:")
    lgbm_importances = lgbm_fitted.feature_importances_ / lgbm_fitted.feature_importances_.sum()
    for name, imp in sorted(zip(feature_names, lgbm_importances), key=lambda x: -x[1]):
        print(f"{name:<30} {imp:.4f}")

    # Save importances to CSV
    os.makedirs(os.path.dirname(IMPORTANCE_CSV), exist_ok=True)
    pd.DataFrame({
        "feature": feature_names,
        "rf_importance": rf_importances,
        "lgbm_importance": lgbm_importances
    }).to_csv(IMPORTANCE_CSV, index=False)

    print("\n[SUCCESS] Ensemble training and evaluation complete.")

if __name__ == "__main__":
    run_ensemble_training()
