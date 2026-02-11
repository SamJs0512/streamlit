import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =============================
# 1. Load Dataset
# =============================
df = pd.read_csv("bodyPerformance.csv")
df.columns = df.columns.str.strip()

# Fix column naming inconsistencies
df = df.rename(columns={
    "body fat_%": "body_fat_pct",
    "gripForce": "gripforce",
    "sit and bend forward_cm": "sit_and_bend_forward_cm",
    "sit-ups counts": "sit-ups_counts",
    "broad jump_cm": "broad_jump_cm"
})

# =============================
# 2. Data Cleaning
# =============================
num_cols = [
    "age", "height_cm", "weight_kg", "body_fat_pct",
    "diastolic", "systolic", "gripforce",
    "sit_and_bend_forward_cm", "sit-ups_counts", "broad_jump_cm"
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].fillna(df[col].median(), inplace=True)

df["class"] = df["class"].astype(str)

# =============================
# 3. Feature Selection
# =============================
features = num_cols + ["gender"]
X = df[features].copy()
y = df["class"]

# Encode gender
X = pd.get_dummies(X, columns=["gender"], drop_first=True)

# =============================
# 4. Feature Engineering
# =============================
X["bmi"] = X["weight_kg"] / ((X["height_cm"] / 100) ** 2)
X["bp_ratio"] = X["systolic"] / (X["diastolic"] + 0.1)
X["age_grip"] = X["age"] * X["gripforce"]
X["strength_weight"] = X["gripforce"] / X["weight_kg"]
X["bodyfat_bmi"] = X["body_fat_pct"] * X["bmi"]

# =============================
# 5. Train-Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# 6. Train Final Tuned Model
# =============================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =============================
# 7. Evaluation
# =============================
y_pred = model.predict(X_test)

print(f"✅ Final Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =============================
# 8. Save Model for Streamlit
# =============================
bundle = {
    "model": model,
    "columns": list(X.columns)
}

joblib.dump(bundle, "fitness_classifier.pkl", compress=3)

print("✅ fitness_classifier.pkl saved successfully!")
