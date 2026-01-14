import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# load dataset
df = pd.read_csv("bodyPerformance.csv")

# rename columns to safe names if needed
df = df.rename(columns={
    "body fat_%": "body_fat_pct"
})

# features available in dataset
feature_cols = [
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "body_fat_pct",
    "diastolic",
    "systolic",
    "gripForce"
]

X = df[feature_cols].copy()
y = df["class"]

# one-hot encode gender
X = pd.get_dummies(X, columns=["gender"], drop_first=True)

# TRAIN MODEL
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42
)
model.fit(X, y)

# bundle with feature names
bundle = {
    "model": model,
    "columns": list(X.columns)
}

# save with compression
joblib.dump(bundle, "fitness_classifier_compact.pkl", compress=3)

print("Saved fitness_classifier_compact.pkl")

