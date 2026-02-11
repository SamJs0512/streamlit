import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
df = pd.read_csv("bodyPerformance.csv")

# Clean column names to handle spaces or special characters consistently
df.columns = df.columns.str.strip()

# 2. Feature selection
# We include the physical test results as they are crucial for the 'class' prediction
features = [
    "age", "gender", "height_cm", "weight_kg", "body fat_%",
    "diastolic", "systolic", "gripForce", 
    "sit and bend forward_cm", "sit-ups counts", "broad jump_cm"
]

X = df[features].copy()
y = df["class"]

# 3. Encode gender (Female: 0, Male: 1)
X = pd.get_dummies(X, columns=["gender"], drop_first=True)

# 4. Derived features (Feature Engineering)
X["BMI"] = X["weight_kg"] / ((X["height_cm"] / 100) ** 2)
X["BP_ratio"] = X["systolic"] / (X["diastolic"] + 0.1) # Added small constant to avoid div by zero
X["age_grip"] = X["age"] * X["gripForce"]
X["weight_height_ratio"] = X["weight_kg"] / X["height_cm"]
X["strength_weight"] = X["gripForce"] / X["weight_kg"]
X["bodyfat_BMI"] = X["body fat_%"] * X["BMI"]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Train HistGradientBoostingClassifier
# This model is excellent for this dataset size and type
model = HistGradientBoostingClassifier(
    max_iter=1000,
    max_depth=12,
    learning_rate=0.05,
    random_state=42,
    early_stopping=True,
    l2_regularization=0.5
)
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Save model + columns
bundle = {
    "model": model,
    "columns": list(X.columns)
}
joblib.dump(bundle, "fitness_classifier.pkl")
print("✅ Model saved successfully as 'fitness_classifier.pkl'")