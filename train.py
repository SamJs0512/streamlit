import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# 1. Read CSV into DataFrame
# ===============================
df = pd.read_csv("bodyPerformance.csv")

# ===============================
# 2. Summary statistics
# ===============================
print("\n--- Summary Statistics ---")
print(df.describe())

# ===============================
# 3. Understand variable types
# ===============================
print("\n--- Data Types ---")
print(df.dtypes)

# ===============================
# 4. Check for missing data
# ===============================
print("\n--- Missing Values ---")
print(df.isnull().sum())

# ===============================
# 5. Clean data
# ===============================
# Rename problematic column
df = df.rename(columns={"body fat_%": "body_fat_pct"})

# Drop rows with missing values (safe for this dataset)
df = df.dropna()

# ===============================
# 6. Feature selection
# (keep model small + meaningful)
# ===============================
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

X = df[feature_cols]
y = df["class"]

# One-hot encode gender
X = pd.get_dummies(X, columns=["gender"], drop_first=True)

# ===============================
# 7. Train-test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 8. Initialise & train model
# (compact to reduce pkl size)
# ===============================
model = RandomForestClassifier(
    n_estimators=80,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# 9. Model evaluation
# ===============================
y_pred = model.predict(X_test)

print("\n--- Model Accuracy ---")
print(accuracy_score(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# ===============================
# 10. Save model + feature columns
# ===============================
bundle = {
    "model": model,
    "columns": list(X.columns)
}

joblib.dump(bundle, "fitness_classifier_compact.pkl", compress=3)

print("\n✅ Model saved as fitness_classifier_compact.pkl")
