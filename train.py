import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("bodyPerformance.csv")

df = df.rename(columns={
    "height": "height_cm",
    "weight": "weight_kg",
    "body fat_%": "body_fat_pct"
})

feature_cols = [
    "age", "gender", "height_cm", "weight_kg",
    "body_fat_pct", "diastolic", "systolic", "gripForce"
]

X = df[feature_cols]
y = df["class"]

X = pd.get_dummies(X, columns=["gender"], drop_first=True)

model = RandomForestClassifier(
    n_estimators=120,
    max_depth=8,
    random_state=42
)
model.fit(X, y)

bundle = {
    "model": model,
    "columns": list(X.columns)
}

joblib.dump(bundle, "fitness_classifier_compact.pkl", compress=3)

print("retrained & saved")
