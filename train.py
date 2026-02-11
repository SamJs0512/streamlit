import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# 1. Load data
# ===============================
df = pd.read_csv("bodyPerformance.csv")

# ===============================
# 2. Clean data
# ===============================
df = df.rename(columns={"body fat_%": "body_fat_pct"})
df = df.dropna()

# ===============================
# 3. Feature selection
# ===============================
features = [
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "body_fat_pct",
    "diastolic",
    "systolic",
    "gripForce"
]

X = df[features]
y = df["class"]

# Encode gender
X = pd.get_dummies(X, columns=["gender"], drop_first=True)

# ===============================
# 4. Train-test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 5. Train model
# ===============================
model = LogisticRegression(
    max_iter=2000,
    multi_class="multinomial"
)

model.fit(X_train, y_train)

# ===============================
# 6. Evaluation
# ===============================
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ===============================
# 7. Save bundle
# ===============================
bundle = {
    "model": model,
    "columns": list(X.columns)
}

joblib.dump(bundle, "fitness_classifier.pkl")

print("✅ Model saved successfully")

