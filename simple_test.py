#!/usr/bin/env python3
"""
Simple test script to verify the traditional sentiment analysis model works
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ml_model import SentimentAnalyzer
    print("Testing traditional sentiment analyzer...")
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Try to load existing model
    if analyzer.load_model():
        print("✓ Model loaded successfully")
        
        # Test predictions
        test_cases = [
            "This product is amazing!",
            "I hate this terrible product",
            "It's okay, nothing special",
            "Wow, this is absolutely fantastic!",
            "Complete waste of money"
        ]
        
        print("\nTesting predictions:")
        for text in test_cases:
            result = analyzer.predict_sentiment(text)
            print(f"Text: '{text}' -> {result}")
            
        print("\n✓ All tests completed successfully!")
        
    else:
        print("Model not found, training new model...")
        analyzer.train_model()
        analyzer.save_model()
        print("✓ Model trained and saved")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()