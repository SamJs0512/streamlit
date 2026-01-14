import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("bodyPerformance.csv")

# Rename columns if needed
df = df.rename(columns={"body fat_%": "body_fat_pct"})

# Features
feature_cols = [
    "age", "gender", "height_cm", "weight_kg",
    "body_fat_pct", "diastolic", "systolic", "gripForce"
]

X = df[feature_cols].copy()
y = df["class"]

# One-hot encode gender
X = pd.get_dummies(X, columns=["gender"], drop_first=True)

# Train model (smaller to shrink file)
model = RandomForestClassifier(
    n_estimators=80,  # fewer trees
    max_depth=8,      # shallower
    random_state=42
)
model.fit(X, y)

# Save bundle with columns
bundle = {
    "model": model,
    "columns": list(X.columns)
}

joblib.dump(bundle, "fitness_classifier_compact.pkl", compress=3)

print("✅ Model retrained & saved in current sklearn version")
