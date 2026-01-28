"""
Flask Web Application for Student Risk Detection Dashboard
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.predictor import StudentRiskPredictor
from src.utils.config import *

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Global variables
predictor = None
feature_data = None

def load_model_and_data():
    """Load the trained model and feature data"""
    global predictor, feature_data
    
    print("Loading model and data...")
    
    # Load trained model
    predictor = StudentRiskPredictor()
    predictor.load_model(model_name='best_model')
    
    # Load feature data
    features_path = os.path.join(PROCESSED_DATA_DIR, 'student_features_1000.csv')
    feature_data = pd.read_csv(features_path)
    
    # Generate risk scores
    X = feature_data.drop(['student_id', 'at_risk'], axis=1)
    risk_scores = predictor.predict_risk(X)
    feature_data['risk_score'] = risk_scores
    
    print(f"✓ Loaded model and {len(feature_data)} student records")

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/predictions')
def get_predictions():
    """Get risk predictions for all students"""
    if predictor is None or feature_data is None:
        return jsonify({'error': 'Model not initialized'})
    
    risk_scores = feature_data['risk_score'].values
    
    # Categorize risk levels
    high_risk = int(sum(score > HIGH_RISK_THRESHOLD for score in risk_scores))
    medium_risk = int(sum(MEDIUM_RISK_THRESHOLD <= score <= HIGH_RISK_THRESHOLD for score in risk_scores))
    low_risk = int(sum(score < MEDIUM_RISK_THRESHOLD for score in risk_scores))
    
    # Get recent high-risk predictions
    high_risk_students = feature_data[feature_data['risk_score'] > HIGH_RISK_THRESHOLD].sort_values('risk_score', ascending=False).head(20)
    
    recent_predictions = []
    for _, row in high_risk_students.iterrows():
        score = row['risk_score']
        risk_level = 'High Risk' if score > HIGH_RISK_THRESHOLD else 'Medium Risk' if score >= MEDIUM_RISK_THRESHOLD else 'Low Risk'
        recent_predictions.append({
            'student_id': str(row['student_id']),
            'risk_score': float(score),
            'risk_level': risk_level
        })
    
    return jsonify({
        'total_students': int(len(feature_data)),
        'high_risk_count': high_risk,
        'medium_risk_count': medium_risk,
        'low_risk_count': low_risk,
        'recent_predictions': recent_predictions
    })

@app.route('/api/student/<student_id>')
def get_student_details(student_id):
    """Get details for a specific student"""
    if predictor is None or feature_data is None:
        return jsonify({'error': 'Model not initialized'})
    
    student_data = feature_data[feature_data['student_id'] == student_id]
    
    if student_data.empty:
        return jsonify({'error': 'Student not found'})
    
    risk_score = student_data['risk_score'].values[0]
    risk_level = 'High Risk' if risk_score > HIGH_RISK_THRESHOLD else 'Medium Risk' if risk_score >= MEDIUM_RISK_THRESHOLD else 'Low Risk'
    
    # Get feature values
    features_dict = {}
    for col in student_data.columns:
        if col not in ['student_id', 'at_risk', 'risk_score']:
            value = student_data[col].values[0]
            if pd.isna(value):
                features_dict[col] = None
            elif isinstance(value, (np.integer, np.int64, np.int32)):
                features_dict[col] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                features_dict[col] = float(value)
            else:
                features_dict[col] = value
    
    return jsonify({
        'student_id': str(student_id),
        'risk_score': float(risk_score),
        'risk_level': risk_level,
        'features': features_dict
    })

if __name__ == '__main__':
    load_model_and_data()
    print("\n" + "="*60)
    print("Starting Flask Web Application")
    print("="*60)
    print(f"Dashboard: http://localhost:{FLASK_PORT}")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)