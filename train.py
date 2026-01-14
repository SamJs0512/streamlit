import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset
df = pd.read_csv("bodyPerformance.csv")
df = df.rename(columns={"body fat_%": "body_fat_pct"})

# 2. Features and target
feature_cols = [
    "age", "gender", "height_cm", "weight_kg",
    "body_fat_pct", "diastolic", "systolic", "gripForce"
]
X = df[feature_cols].copy()
y = df["class"]

# One-hot encode gender
X = pd.get_dummies(X, columns=["gender"], drop_first=True)

# 3. Train Random Forest
model = RandomForestClassifier(
    n_estimators=80,   # smaller to shrink file
    max_depth=8,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X, y)

# 4. Save model + columns
bundle = {"model": model, "columns": list(X.columns)}
joblib.dump(bundle, "fitness_classifier_compact.pkl", compress=3)

print("✅ Model retrained & saved")
