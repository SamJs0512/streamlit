# train.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# 1. Load data
# ===============================
df = pd.read_csv("bodyPerformance.csv")
df.columns = df.columns.str.strip()

# ===============================
# 2. Features
# ===============================
base_features = [
    "age", "gender", "height_cm", "weight_kg", "body fat_%",
    "diastolic", "systolic", "gripForce",
    "sit and bend forward_cm", "sit-ups counts", "broad jump_cm"
]

X = df[base_features].copy()
y = df["class"]

# ===============================
# 3. Encode gender
# ===============================
X["gender_M"] = (X["gender"] == "M").astype(int)
X.drop(columns=["gender"], inplace=True)

# ===============================
# 4. Feature engineering
# ===============================
X["BMI"] = X["weight_kg"] / ((X["height_cm"] / 100) ** 2)
X["BP_ratio"] = X["systolic"] / (X["diastolic"] + 0.1)
X["age_grip"] = X["age"] * X["gripForce"]
X["weight_height_ratio"] = X["weight_kg"] / X["height_cm"]
X["strength_weight"] = X["gripForce"] / X["weight_kg"]
X["bodyfat_BMI"] = X["body fat_%"] * X["BMI"]

# ===============================
# 5. Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 6. Train
# ===============================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
model.fit(X_train, y_train)

# ===============================
# 7. Evaluate
# ===============================
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# ===============================
# 8. Save bundle
# ===============================
joblib.dump(
    {
        "model": model,
        "columns": X.columns.tolist()
    },
    "fitness_classifier.pkl"
)

print("✅ Model saved")
