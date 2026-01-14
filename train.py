import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("bodyPerformance.csv")

# rename columns if needed
df = df.rename(columns={
    "body fat_%": "body_fat_pct"
})

# -----------------------------
# 2. Define features and target
# -----------------------------
feature_cols = [
    "age", "gender", "height_cm", "weight_kg",
    "body_fat_pct", "diastolic", "systolic", "gripForce"
]

X = df[feature_cols].copy()
y = df["class"]

# one-hot encode gender
X = pd.get_dummies(X, columns=["gender"], drop_first=True)

# -----------------------------
# 3. Train smaller Random Forest
# -----------------------------
model = RandomForestClassifier(
    n_estimators=80,   # smaller model
    max_depth=8,       # shallower trees
    min_samples_leaf=5, 
    random_state=42
)
model.fit(X, y)

# -----------------------------
# 4. Save model + columns bundle
# -----------------------------
bundle = {
    "model": model,
    "columns": list(X.columns)
}

joblib.dump(bundle, "fitness_classifier_compact.pkl", compress=3)
print("✅ Model retrained & saved as fitness_classifier_compact.pkl")
