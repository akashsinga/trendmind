# weekly_train.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from features.create_weekly_features import create_weekly_features

MODEL_PATH = "models/weekly_model.pkl"
DATA_PATH = "data/weekly_processed.csv"


def train_weekly_model():
    print("[INFO] Loading weekly aggregated data...")
    df = pd.read_csv(DATA_PATH)

    print("[INFO] Creating features...")
    df = create_weekly_features(df)

    if df.empty:
        print("[ERROR] No data after feature creation.")
        return

    X = df.drop(columns=["symbol", "week", "target"])
    y = df["target"]

    print("[INFO] Splitting data for training...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("[INFO] Training RandomForest model on weekly features...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n[RESULTS]")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")

    print("\n[DEBUG] Prediction distribution:")
    print(pd.Series(y_pred).value_counts())

    print("[INFO] Saving model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[SUCCESS] Weekly model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_weekly_model()
