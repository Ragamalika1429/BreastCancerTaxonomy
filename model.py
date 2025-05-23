import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

# Load the dataset with selected features
def load_data():
    data = pd.read_csv('data/breast_cancer_data.csv')
    data = data.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    selected_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
    X = data[selected_features]
    y = data['diagnosis']
    print("Selected features:", X.columns.tolist())
    print(X.head())
    return X, y

# Train the individual models and calculate their weights
def train_models(X_train, y_train, X_val, y_val):
    models = {}
    weights = {}
    clf1 = RandomForestClassifier(random_state=42)
    clf2 = SVC(probability=True, random_state=42)
    clf3 = DecisionTreeClassifier(random_state=42)
    clf4 = KNeighborsClassifier()
    classifiers = [('rf', clf1), ('svc', clf2), ('dt', clf3), ('knn', clf4)]
    for name, clf in classifiers:
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        models[name] = clf
        weights[name] = accuracy
    return models, weights

# Adaptive voting prediction
def adaptive_voting_predict(models, weights, input_data):
    total_weight = sum(weights.values())
    if total_weight == 0:
        raise ValueError("Total weight is zero.")
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    weighted_predictions = np.zeros(input_data.shape[0])
    for name, model in models.items():
        pred_proba = model.predict_proba(input_data)[:, 1]
        weighted_predictions += normalized_weights[name] * pred_proba
    return (weighted_predictions >= 0.5).astype(int)

# Train the model
def train_model():
    os.makedirs('models', exist_ok=True)
    X, y = load_data()
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join('models', 'scaler.pkl'))
    joblib.dump(imputer, os.path.join('models', 'imputer.pkl'))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models, weights = train_models(X_train, y_train, X_test, y_test)
    y_pred = adaptive_voting_predict(models, weights, X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))
    for name, model in models.items():
        joblib.dump(model, os.path.join('models', f'{name}_model.pkl'))
        print(f"{name} model saved.")
    with open(os.path.join('models', 'model_weights.json'), 'w') as f:
        json.dump(weights, f)
    print("Model weights saved.")

# Test the saved models
def test_models():
    models = {
        'rf': joblib.load(os.path.join('models', 'rf_model.pkl')),
        'svc': joblib.load(os.path.join('models', 'svc_model.pkl')),
        'dt': joblib.load(os.path.join('models', 'dt_model.pkl')),
        'knn': joblib.load(os.path.join('models', 'knn_model.pkl')),
    }
    scaler = joblib.load(os.path.join('models', 'scaler.pkl'))
    imputer = joblib.load(os.path.join('models', 'imputer.pkl'))
    with open(os.path.join('models', 'model_weights.json'), 'r') as f:
        weights = json.load(f)
    print("Loaded weights:", weights)

    benign_input = np.array([13.54, 14.36, 87.46, 566.3, 0.09779]).reshape(1, -1)
    malignant_input = np.array([20.57, 17.77, 132.9, 1326.0, 0.08474]).reshape(1, -1)

    benign_input = imputer.transform(benign_input)
    benign_input = scaler.transform(benign_input)
    malignant_input = imputer.transform(malignant_input)
    malignant_input = scaler.transform(malignant_input)

    benign_pred = adaptive_voting_predict(models, weights, benign_input)
    malignant_pred = adaptive_voting_predict(models, weights, malignant_input)

    print("Benign input prediction:", "Malignant" if benign_pred[0] == 1 else "Benign")
    print("Malignant input prediction:", "Malignant" if malignant_pred[0] == 1 else "Benign")

if __name__ == "__main__":
    train_model()
    test_models()