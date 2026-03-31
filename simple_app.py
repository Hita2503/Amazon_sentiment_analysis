#!/usr/bin/env python3
"""
Simple Flask application for sentiment analysis with traditional TF-IDF + Logistic Regression
"""

import os
import json
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from enhanced_ml_model import SentimentAnalyzer
from database import PredictionDatabase

app = Flask(__name__)

# Initialize the traditional sentiment analyzer
print("Initializing Traditional Sentiment Analyzer...")
analyzer = SentimentAnalyzer()
db = PredictionDatabase()

# Load the model
print("Loading model...")
if analyzer.load_model():
    print("✓ Model loaded successfully!")
    model_info = analyzer.get_model_info()
    print(f"Model Info: {model_info}")
else:
    print("❌ Failed to load model")

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('.', filename)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment for given text"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Get prediction from enhanced analyzer
        result = analyzer.predict_sentiment(text)
        
        # Store prediction in database
        db.save_prediction(text, result['sentiment'], result['confidence'])
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    model_info = analyzer.get_model_info()
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_info.get('model_loaded', False),
        'model_type': model_info.get('model_type', 'unknown'),
        'supports_sarcasm': model_info.get('supports_sarcasm', False),
        'supports_mixed_tone': model_info.get('supports_mixed_tone', False)
    })

@app.route('/history')
def history():
    """Get recent prediction history"""
    try:
        recent_predictions = db.get_recent_predictions(10)
        return jsonify(recent_predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server with Traditional TF-IDF + Logistic Regression Model...")
    app.run(debug=True, host='0.0.0.0', port=8000)