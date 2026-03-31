import pandas as pd
import numpy as np

# Models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# If not installed, install using: pip install xgboost
from xgboost import XGBClassifier

# -----------------------------
# 1. Load Data
# -----------------------------
train_df = pd.read_excel("Training.xlsx")
test_df = pd.read_excel("Testing.xlsx")

print("Training Data Shape:", train_df.shape)
print("Testing Data Shape:", test_df.shape)

# -----------------------------
# 2. Preprocessing
# -----------------------------
target_col = "prognosis"

# Encode target labels
le = LabelEncoder()
train_df[target_col] = le.fit_transform(train_df[target_col])
test_df[target_col] = le.transform(test_df[target_col])

# Split features and target
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 3. Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

results = {}

# -----------------------------
# 4. Training & Evaluation
# -----------------------------
for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# -----------------------------
# 5. Best Model Selection
# -----------------------------
best_model_name = max(results, key=results.get)
print("\n✅ Best Model:", best_model_name)
print("Best Accuracy:", results[best_model_name])

# -----------------------------
# 6. Predict Function
# -----------------------------
def predict_disease(sample):
    best_model = models[best_model_name]
    sample_scaled = scaler.transform([sample])
    prediction = best_model.predict(sample_scaled)
    
    disease = le.inverse_transform(prediction)
    return disease[0]

# Example usage (replace with real values)
# sample_input = X_test[0]
# print("Predicted Disease:", predict_disease(sample_input))