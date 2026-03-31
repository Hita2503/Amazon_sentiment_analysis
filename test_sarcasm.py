#!/usr/bin/env python3
"""
Test script to evaluate sentiment analysis performance on sarcastic and mixed-tone examples
"""

import os
import sys
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_model import SentimentAnalyzer

def test_sarcasm_detection():
    """Test the model's performance on sarcastic and mixed-tone examples"""
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    if not analyzer.load_model():
        print("Error: Could not load model")
        return
    
    print("Testing Sentiment Analysis on Sarcastic and Mixed-Tone Examples")
    print("=" * 70)
    
    # Test cases with expected sentiment (for evaluation)
    test_cases = [
        # Sarcastic examples (should be negative but might be detected as positive)
        {
            "text": "Oh great, another broken product. Just what I needed!",
            "expected": "negative",
            "type": "sarcastic"
        },
        {
            "text": "Wow, this is absolutely fantastic... NOT!",
            "expected": "negative", 
            "type": "sarcastic"
        },
        {
            "text": "Perfect! It broke on the first day. Amazing quality!",
            "expected": "negative",
            "type": "sarcastic"
        },
        
        # Mixed-tone examples
        {
            "text": "The product works well but the packaging was terrible",
            "expected": "neutral",
            "type": "mixed"
        },
        {
            "text": "Great features but overpriced for what you get",
            "expected": "neutral",
            "type": "mixed"
        },
        {
            "text": "I love the design but hate the customer service",
            "expected": "neutral",
            "type": "mixed"
        },
        
        # Clear positive examples (baseline)
        {
            "text": "This product is amazing! Highly recommend it!",
            "expected": "positive",
            "type": "clear_positive"
        },
        {
            "text": "Excellent quality and fast shipping. Very satisfied!",
            "expected": "positive",
            "type": "clear_positive"
        },
        
        # Clear negative examples (baseline)
        {
            "text": "Terrible product. Complete waste of money.",
            "expected": "negative",
            "type": "clear_negative"
        },
        {
            "text": "Poor quality and bad customer service. Avoid this!",
            "expected": "negative",
            "type": "clear_negative"
        }
    ]
    
    results = []
    correct_predictions = 0
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing {case['type']} example:")
        print(f"Text: \"{case['text']}\"")
        print(f"Expected: {case['expected']}")
        
        # Get prediction
        result = analyzer.predict_sentiment(case['text'])
        predicted = result['sentiment']
        confidence = result['confidence']
        
        print(f"Predicted: {predicted} (confidence: {confidence:.3f})")
        
        # Check if prediction matches expectation
        is_correct = predicted == case['expected']
        if is_correct:
            correct_predictions += 1
            print("✓ CORRECT")
        else:
            print("✗ INCORRECT")
        
        results.append({
            'text': case['text'],
            'type': case['type'],
            'expected': case['expected'],
            'predicted': predicted,
            'confidence': confidence,
            'correct': is_correct,
            'probabilities': result['probabilities']
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_tests = len(test_cases)
    accuracy = (correct_predictions / total_tests) * 100
    
    print(f"Total tests: {total_tests}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Overall accuracy: {accuracy:.1f}%")
    
    # Breakdown by type
    types = {}
    for result in results:
        type_name = result['type']
        if type_name not in types:
            types[type_name] = {'total': 0, 'correct': 0}
        types[type_name]['total'] += 1
        if result['correct']:
            types[type_name]['correct'] += 1
    
    print("\nAccuracy by type:")
    for type_name, stats in types.items():
        type_accuracy = (stats['correct'] / stats['total']) * 100
        print(f"  {type_name}: {stats['correct']}/{stats['total']} ({type_accuracy:.1f}%)")
    
    # Identify challenging cases
    print("\nMost challenging cases (lowest confidence):")
    sorted_results = sorted(results, key=lambda x: x['confidence'])
    for result in sorted_results[:3]:
        print(f"  \"{result['text'][:50]}...\" - {result['confidence']:.3f}")
    
    return results

if __name__ == "__main__":
    test_sarcasm_detection()