# model/train_model.py
"""
Train a simple classifier on the Heart Disease dataset and save it to model/heart_model.joblib.

Expected CSV at: data/heart.csv
- Must contain a 'target' column (1 = disease present, 0 = absent)
- Common feature columns: age, sex, cp, trestbps, chol, fbs, restecg,
  thalach, exang, oldpeak, slope, ca, thal
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_PATH = os.getenv("DATA_PATH", "data/heart.csv")
OUT_PATH = os.getenv("OUT_PATH", "model/heart_model.joblib")

# Read data
df = pd.read_csv(DATA_PATH)

# Features and target
FEATURES = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal"]
TARGET = "target"

X = df[FEATURES].copy()
y = df[TARGET].copy()

# Treat some as categorical (encoded as integers in dataset)
cat_features = ["cp", "restecg", "slope", "thal"]
num_features = [f for f in FEATURES if f not in cat_features]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ]
)

clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf.fit(X_train, y_train)
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Validation accuracy: {acc:.4f}")

# Save a bundle with model and feature order (for the API)
bundle = {"model": clf, "features": FEATURES}
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
joblib.dump(bundle, OUT_PATH)
print(f"Saved model to {OUT_PATH}")
