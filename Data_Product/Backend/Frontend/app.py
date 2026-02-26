
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__, static_folder='../frontend', static_url_path='/')
CORS(app)

# Robust path determination
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Model is saved in the same directory as train_model.py (d:/temp/backend/)
MODEL_PATH = os.path.join(BASE_DIR, 'crab_age_model_pipeline.pkl')

model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        print(f"Model file not found at {MODEL_PATH}. Please run train_model.py first.")

# Load model immediately on import
load_model()

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.json
        # Expecting: length, height, diameter, shucked_weight_ratio
        
        # Validating input
        required_fields = ['length', 'height', 'diameter', 'shucked_weight_ratio']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
            
        # Create DataFrame for prediction (must match training columns)
        # Columns: Length, Height, Diameter, Shucked_Weight_Ratio
        import pandas as pd
        input_df = pd.DataFrame([[
            float(data['length']),
            float(data['height']),
            float(data['diameter']),
            float(data['shucked_weight_ratio'])
        ]], columns=["Length", "Height", "Diameter", "Shucked_Weight_Ratio"])
        
        # Predict (model returns log-age)
        log_prediction = model.predict(input_df)
        
        # Inverse transform (expm1) -> This is ALREADY in MONTHS based on dataset
        predicted_age_months = np.expm1(log_prediction)[0]
        
        return jsonify({
            'predicted_age_months': float(predicted_age_months),
            'inputs': data
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluation', methods=['GET'])
def evaluation():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 503
        
    try:
        # Load dataset matching notebook logic
        csv_path = os.path.join(BASE_DIR, '..', 'cleanCrabAgePrediction.csv')
        if not os.path.exists(csv_path):
             return jsonify({'error': 'Dataset not found'}), 404
             
        df = pd.read_csv(csv_path)
        # Preprocessing: Remove height=0 (Notebook Cell 4)
        df = df[df['Height'] > 0]
        
        # Prepare Features (Notebook Cell 6)
        X = df[["Length", "Height", "Diameter", "Shucked_Weight_Ratio"]].copy()
        Y_true = df["Age"].copy()
        
        # Predict (Notebook Cell 23)
        # Model is the full pipeline, handles scaling automatically
        log_predictions = model.predict(X)
        predicted_ages = np.expm1(log_predictions)
        
        # Calculate MAE (Notebook Cell 23)
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(Y_true, predicted_ages)
        
        # Prepare comparison table (Top 100 rows as per notebook output)
        comparisons = []
        for i in range(100):
            comparisons.append({
                'id': int(df.index[i]),
                'age': int(Y_true.iloc[i]),
                'predicted_age': float(predicted_ages[i])
            })
            
        return jsonify({
            'mae': float(mae),
            'mae_unit': 'years', # Keeping user's preferred label
            'comparisons': comparisons
        })

    except Exception as e:
        print(f"Evaluation error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
