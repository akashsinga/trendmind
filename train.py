# train.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

MODEL_PATH = "models/random_forest_model.pkl"
DATA_PATH = "data/processed_data.csv"

def train_model(data_path=DATA_PATH):
    print("[INFO] Loading processed dataset...")
    df = pd.read_csv(data_path)

    if df.empty:
        print("[ERROR] Processed dataset is empty.")
        return

    # Drop non-feature columns
    print("[INFO] Preparing training data...")
    X = df.drop(columns=["symbol", "date", "target"])
    y = df["target"]

    print("[INFO] Splitting dataset (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("[INFO] Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("[INFO] Evaluating model...")
    y_pred = model.predict(X_test)

    print("[RESULTS]")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    
    print("\n[INFO] Feature Importances:")
    for feature, importance in sorted(zip(X.columns, model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"{feature:<30} {importance:.4f}")


    # Save the model
    print("[INFO] Saving model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[SUCCESS] Model saved to {MODEL_PATH}")

    return model

if __name__ == "__main__":
    train_model()
