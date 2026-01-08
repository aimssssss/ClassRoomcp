import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# CREATE DATASET (NIGERIAN CLASSROOM CONDITIONS)
# -----------------------------

data = {
    "Temperature": [30, 32, 28, 35, 29, 31, 34, 27, 33, 36,
                    26, 28, 30, 31, 29, 35, 34, 32, 27, 33],
    
    "Humidity": [60, 65, 55, 70, 58, 62, 68, 50, 66, 72,
                 48, 54, 59, 63, 57, 71, 69, 64, 52, 67],
    
    "CO2": [700, 850, 600, 1100, 650, 800, 1000, 550, 900, 1200,
            500, 620, 720, 780, 660, 1150, 1050, 820, 580, 950],
    
    "Light": [300, 280, 350, 200, 320, 290, 220, 360, 260, 180,
              380, 340, 310, 295, 325, 210, 230, 285, 355, 250],
    
    # 0 = Uncomfortable, 1 = Neutral, 2 = Comfortable
    "Comfort": [1, 0, 2, 0, 2, 1, 0, 2, 1, 0,
                2, 2, 1, 1, 2, 0, 0, 1, 2, 1]
}

df = pd.DataFrame(data)

print("Dataset created successfully")
print(df.head())

# -----------------------------
# MACHINE LEARNING MODEL
# -----------------------------

X = df[["Temperature", "Humidity", "CO2", "Light"]]
y = df["Comfort"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# MODEL EVALUATION
# -----------------------------

predictions = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# -----------------------------
# PREDICT A NEW CLASSROOM
# -----------------------------

new_classroom = [[33, 66, 950, 260]]  # Typical Nigerian classroom
prediction = model.predict(new_classroom)

labels = {0: "Uncomfortable", 1: "Neutral", 2: "Comfortable"}
print("\nPredicted Comfort Level:", labels[prediction[0]])
import joblib
joblib.dump(model, 'classroom_comfort_model.pkl')
print("Success! Brain file created.")