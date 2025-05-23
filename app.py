from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load models, scaler, imputer, and weights
def load_model(name):
    model_path = os.path.join('models', f'{name}_model.pkl')
    return joblib.load(model_path)

def load_scaler():
    return joblib.load(os.path.join('models', 'scaler.pkl'))

def load_imputer():
    return joblib.load(os.path.join('models', 'imputer.pkl'))

def load_weights():
    weights_path = os.path.join('models', 'model_weights.json')
    try:
        with open(weights_path, 'r') as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Error loading weights: {e}")
        return {"rf": 0.25, "svc": 0.25, "dt": 0.25, "knn": 0.25}

# Load trained models and preprocessing tools
models = {
    'rf': load_model('rf'),
    'svc': load_model('svc'),
    'dt': load_model('dt'),
    'knn': load_model('knn'),
}
scaler = load_scaler()
imputer = load_imputer()
weights = load_weights()
print("Loaded weights:", weights)

def adaptive_voting_predict(models, weights, input_data):
    total_weight = sum(weights.values())
    if total_weight == 0:
        raise ValueError("Total weight is zero.")
    
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    print("Normalized weights:", normalized_weights)
    weighted_predictions = np.zeros(input_data.shape[0])

    for name, model in models.items():
        try:
            pred_proba = model.predict_proba(input_data)[:, 1]
            print(f"{name} probability (Malignant): {pred_proba}")
            weighted_predictions += normalized_weights[name] * pred_proba
        except Exception as e:
            print(f"Model {name} error: {e}")
    
    print("Weighted predictions before threshold:", weighted_predictions)
    return (weighted_predictions >= 0.5).astype(int)

def generate_graph(data, labels, title, colors=None):
    plt.figure(figsize=(7, 4))
    colors = colors or ['blue'] * len(labels)
    plt.bar(labels, data, color=colors)
    plt.xlabel("Categories")
    plt.ylabel("Values")
    plt.title(title)
    plt.xticks(rotation=45)
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches="tight")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = {key: float(request.form[key]) for key in request.form}
        expected_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
        if sorted(features.keys()) != sorted(expected_features):
            raise ValueError(f"Expected features {expected_features}, got {list(features.keys())}")
        input_data = np.array(list(features.values())).reshape(1, -1)
        print("Input features:", features)
        print("Raw input data:", input_data)

        input_data = imputer.transform(input_data)
        print("After imputation:", input_data)
        input_data = scaler.transform(input_data)
        print("After scaling:", input_data)

        prediction = adaptive_voting_predict(models, weights, input_data)
        prediction_label = "Malignant" if prediction[0] == 1 else "Benign"
        print("Final prediction:", prediction_label)

        weighted_prob = 0
        for name, model in models.items():
            pred_proba = model.predict_proba(input_data)[:, 1]
            weighted_prob += weights[name] * pred_proba
        total_weight = sum(weights.values())
        weighted_prob = weighted_prob / total_weight
        print("Weighted probability (Malignant):", weighted_prob)
        prediction_probabilities = [1 - weighted_prob[0], weighted_prob[0]]

        graph_url_prob = generate_graph(
            prediction_probabilities, ['Benign', 'Malignant'], "Cancer Prediction Probabilities", colors=['green', 'red']
        )
        graph_url_features = generate_graph(
            list(features.values()), list(features.keys()), "Feature Values", colors=['purple'] * 5
        )

        return render_template(
            'results.html', 
            prediction=prediction_label, 
            graph_url_prob=graph_url_prob, 
            graph_url_features=graph_url_features
        )
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return render_template('results.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)