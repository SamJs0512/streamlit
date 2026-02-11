import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
df = pd.read_csv("bodyPerformance.csv")

# 2. Clean data
df = df.rename(columns={"body fat_%": "body_fat_pct"})
df = df.dropna()  # Remove missing values

# 3. Feature selection
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

# Encode categorical features
X = pd.get_dummies(X, columns=["gender"], drop_first=True)

# 4. Feature scaling (important for numeric features)
numeric_features = ["age", "height_cm", "weight_kg", "body_fat_pct",
                    "diastolic", "systolic", "gripForce"]

scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Train Random Forest model
model = RandomForestClassifier(
    n_estimators=200,   # Number of trees
    max_depth=10,       # Limit depth to prevent overfitting
    random_state=42
)
model.fit(X_train, y_train)

# 7. Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Save bundle (model + columns + scaler)
bundle = {
    "model": model,
    "columns": list(X.columns),
    "scaler": scaler
}

joblib.dump(bundle, "fitness_classifier.pkl")
print("✅ Random Forest model saved successfully")
