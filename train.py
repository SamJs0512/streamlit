import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
df = pd.read_csv("bodyPerformance.csv")
df.columns = df.columns.str.strip()  # Clean column names

# 2. Feature selection
features = [
    "age", "gender", "height_cm", "weight_kg", "body fat_%",
    "diastolic", "systolic", "gripForce", 
    "sit and bend forward_cm", "sit-ups counts", "broad jump_cm"
]

X = df[features].copy()
y = df["class"]

# 3. Encode gender
X = pd.get_dummies(X, columns=["gender"], drop_first=True)

# 4. Feature Engineering
X["BMI"] = X["weight_kg"] / ((X["height_cm"] / 100) ** 2)
X["BP_ratio"] = X["systolic"] / (X["diastolic"] + 0.1)
X["age_grip"] = X["age"] * X["gripForce"]
X["weight_height_ratio"] = X["weight_kg"] / X["height_cm"]
X["strength_weight"] = X["gripForce"] / X["weight_kg"]
X["bodyfat_BMI"] = X["body fat_%"] * X["BMI"]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Train Model with smaller size
model = RandomForestClassifier(
    n_estimators=200,   # reduced from 500
    max_depth=10,       # reduced from 12
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
print(f"✅ Training Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# 8. Save model + columns with compression
bundle = {
    "model": model,
    "columns": list(X.columns)
}
joblib.dump(bundle, "fitness_classifier.pkl", compress=3, protocol=5)  # compress level 3
print("✅ NEW 'fitness_classifier.pkl' saved. Should now be <25MB")
