import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("/STREAMLIT/bodyPerformance.csv")

feature_cols = [
    "age", "gender", "height_cm", "weight_kg", "body fat_%",
    "diastolic", "systolic",
    "gripForce", "sit_and_bend_forward", "sit_ups", "broad_jump"
]

X = df[feature_cols]
y = df["class"]

# one-hot encode gender
X = pd.get_dummies(X, columns=["gender"])

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# save model
joblib.dump(model, "fitness_classifier.pkl")

print("Model saved as fitness_classifier.pkl")