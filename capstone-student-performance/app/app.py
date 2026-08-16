from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
import json
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load models
def load_models():
    """Load trained models"""
    models = {}
    try:
        if os.path.exists('../models/random_forest.pkl'):
            models['random_forest'] = joblib.load('../models/random_forest.pkl')
        if os.path.exists('../models/grade_nn.h5'):
            models['neural_network'] = load_model('../models/grade_nn.h5')
        if os.path.exists('../models/lstm_model.h5'):
            models['lstm'] = load_model('../models/lstm_model.h5')
    except Exception as e:
        print(f"Error loading models: {e}")
    return models

models = load_models()

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html', models_loaded=bool(models))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction endpoint"""
    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        data = request.get_json()
        
        # Features expected by the model
        features = ['Semester', 'InternalMarks', 'ExternalMarks', 'subject_category',
                   'avg_semester_marks', 'std_semester_marks', 'min_semester_marks', 
                   'max_semester_marks', 'avg_internal_marks', 'avg_external_marks',
                   'overall_avg_marks', 'overall_std_marks', 'total_subjects',
                   'overall_avg_internal', 'overall_avg_external', 'current_semester',
                   'internal_external_ratio']
        
        # Create feature array
        input_features = np.array([[data.get(f, 0) for f in features]])
        
        predictions = {}
        
        # Random Forest prediction
        if 'random_forest' in models:
            rf_pred = models['random_forest'].predict(input_features)[0]
            rf_prob = models['random_forest'].predict_proba(input_features)[0]
            predictions['random_forest'] = {
                'prediction': 'Pass' if rf_pred == 1 else 'Fail',
                'probability': float(rf_prob[1] if rf_pred == 1 else rf_prob[0]),
                'pass_probability': float(rf_prob[1])
            }
        
        # Neural Network prediction
        if 'neural_network' in models:
            nn_pred = (models['neural_network'].predict(input_features) > 0.5).astype(int)[0][0]
            nn_prob = models['neural_network'].predict(input_features)[0][0]
            predictions['neural_network'] = {
                'prediction': 'Pass' if nn_pred == 1 else 'Fail',
                'probability': float(nn_prob if nn_pred == 1 else 1 - nn_prob),
                'pass_probability': float(nn_prob)
            }
        
        # LSTM prediction (if available and sequential data provided)
        if 'lstm' in models and 'sequence_data' in data:
            seq_data = np.array(data['sequence_data']).reshape(-1, len(data['sequence_data']), 1)
            lstm_pred = models['lstm'].predict(seq_data)[0][0]
            predictions['lstm'] = {
                'predicted_score': float(lstm_pred)
            }
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'features_used': features,
            'input_data': data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(models.keys()),
        'version': '1.0.0'
    })

@app.route('/model_info')
def model_info():
    """Get model information"""
    info = {}
    for name in models:
        info[name] = {
            'type': 'Random Forest' if name == 'random_forest' else 'Neural Network' if name == 'neural_network' else 'LSTM',
            'loaded': True
        }
    return jsonify(info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)