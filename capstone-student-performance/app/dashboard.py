import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from data_preprocessing import StudentDataPreprocessor

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

st.title("🎓 Student Performance Prediction Dashboard")

# Load models
@st.cache_resource
def load_models():
    models = {}
    try:
        if os.path.exists(os.path.join(project_root, 'models', 'random_forest.pkl')):
            models['random_forest'] = joblib.load(os.path.join(project_root, 'models', 'random_forest.pkl'))
        if os.path.exists(os.path.join(project_root, 'models', 'grade_nn.h5')):
            from tensorflow.keras.models import load_model
            models['neural_network'] = load_model(os.path.join(project_root, 'models', 'grade_nn.h5'))
    except Exception as e:
        st.warning(f"Error loading models: {e}")
    return models

models = load_models()

if not models:
    st.warning("⚠️ Models not loaded. Please train models first.")
else:
    st.success(f"✅ Models loaded: {', '.join(models.keys())}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Student Information")
        semester = st.slider("Semester", 1, 8, 1)
        internal_marks = st.slider("Internal Marks", 0, 100, 45)
        external_marks = st.slider("External Marks", 0, 100, 35)
        
    with col2:
        st.subheader("Academic Context")
        avg_semester = st.slider("Avg Semester Marks", 40, 95, 75)
        overall_avg = st.slider("Overall Avg Marks", 40, 95, 72)
        
    if st.button("Predict Performance", type="primary"):
        # Prepare input
        input_data = {
            'Semester': semester,
            'InternalMarks': internal_marks,
            'ExternalMarks': external_marks,
            'subject_category': 0,
            'avg_semester_marks': avg_semester,
            'std_semester_marks': 10,
            'min_semester_marks': avg_semester - 20,
            'max_semester_marks': avg_semester + 20,
            'avg_internal_marks': internal_marks,
            'avg_external_marks': external_marks,
            'overall_avg_marks': overall_avg,
            'overall_std_marks': 12,
            'total_subjects': 40,
            'overall_avg_internal': internal_marks,
            'overall_avg_external': external_marks,
            'current_semester': semester,
            'internal_external_ratio': internal_marks / (external_marks + 1)
        }
        
        input_df = pd.DataFrame([input_data])
        
        st.subheader("📊 Prediction Results")
        
        results = {}
        
        if 'random_forest' in models:
            rf_pred = models['random_forest'].predict(input_df)[0]
            rf_prob = models['random_forest'].predict_proba(input_df)[0]
            # Handle case where only 1 class exists
            if len(rf_prob) > 1:
                pass_prob = rf_prob[1] * 100
            else:
                pass_prob = rf_prob[0] * 100 if rf_pred == 1 else (1 - rf_prob[0]) * 100
            results['Random Forest'] = {
                'Prediction': '✅ PASS' if rf_pred == 1 else '❌ FAIL',
                'Pass Probability': f"{pass_prob:.1f}%"
            }
        
        if 'neural_network' in models:
            nn_pred = (models['neural_network'].predict(input_df) > 0.5).astype(int)[0][0]
            nn_prob = models['neural_network'].predict(input_df)[0][0]
            results['Neural Network'] = {
                'Prediction': '✅ PASS' if nn_pred == 1 else '❌ FAIL',
                'Pass Probability': f"{nn_prob*100:.1f}%"
            }
        
        st.dataframe(pd.DataFrame(results).T)