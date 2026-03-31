#!/usr/bin/env python3
"""
Test script to evaluate transformer model performance on mixed and contradictory tones
"""

import sys
from typing import List, Dict, Tuple
from enhanced_ml_model import SentimentAnalyzer

def test_mixed_tone_examples():
    """Test the transformer model on various mixed tone examples"""
    
    # Initialize the enhanced analyzer (should use transformer by default now)
    print("Initializing Enhanced Sentiment Analyzer with Transformer...")
    analyzer = SentimentAnalyzer()
    
    # Load the model
    if not analyzer.load_model():
        print("❌ Failed to load model")
        return
    
    # Get model info
    info = analyzer.get_model_info()
    print(f"Model Info: {info}")
    print("=" * 80)
    
    # Test cases with expected outcomes
    test_cases = [
        # Clear positive cases
        ("This product is absolutely amazing!", "positive"),
        ("I love this item, it's perfect!", "positive"),
        ("Excellent quality and fast shipping", "positive"),
        
        # Clear negative cases
        ("This product is terrible and broken", "negative"),
        ("I hate this item, complete waste of money", "negative"),
        ("Poor quality and slow delivery", "negative"),
        
        # Clear neutral cases
        ("The product is okay, nothing special", "neutral"),
        ("It's average quality for the price", "neutral"),
        ("Standard product, meets basic expectations", "neutral"),
        
        # Sarcastic cases (should be detected as negative)
        ("Oh great, another broken product. Just what I needed!", "negative"),
        ("Wow, this is absolutely fantastic... NOT!", "negative"),
        ("Perfect! It broke on the first day. Amazing quality!", "negative"),
        ("Yeah, sure, this is the 'best' product ever", "negative"),
        
        # Mixed tone cases (should be detected as neutral or appropriately weighted)
        ("The product works well but the packaging was terrible", "neutral"),
        ("Great features but overpriced for what you get", "neutral"),
        ("I love the design but hate the customer service", "neutral"),
        ("Good quality however the delivery was delayed", "neutral"),
        ("Fast shipping and good packaging, though the product itself is mediocre", "neutral"),
        ("The camera is excellent but the battery life is disappointing", "neutral"),
        
        # Complex contradictory cases
        ("This is both the best and worst purchase I've made", "neutral"),
        ("Amazing product with terrible support", "neutral"),
        ("Love it and hate it at the same time", "neutral"),
        ("Great when it works, but it rarely works", "negative"),
        
        # Edge cases
        ("", "neutral"),
        ("Good bad good bad", "neutral"),
        ("Not bad", "positive"),  # Double negative
        ("Not terrible", "positive"),  # Double negative
    ]
    
    print("Testing Mixed Tone and Sarcasm Detection:")
    print("=" * 80)
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    # Track performance by category
    categories = {
        'clear_positive': [],
        'clear_negative': [],
        'clear_neutral': [],
        'sarcastic': [],
        'mixed_tone': [],
        'complex': [],
        'edge_cases': []
    }
    
    # Categorize test cases
    category_ranges = {
        'clear_positive': (0, 3),
        'clear_negative': (3, 6),
        'clear_neutral': (6, 9),
        'sarcastic': (9, 13),
        'mixed_tone': (13, 19),
        'complex': (19, 23),
        'edge_cases': (23, 27)
    }
    
    for i, (text, expected) in enumerate(test_cases):
        result = analyzer.predict_sentiment(text)
        predicted = result['sentiment']
        confidence = result['confidence']
        probabilities = result['probabilities']
        
        # Determine if prediction is correct
        is_correct = predicted == expected
        if is_correct:
            correct_predictions += 1
        
        # Categorize result
        for category, (start, end) in category_ranges.items():
            if start <= i < end:
                categories[category].append({
                    'text': text,
                    'expected': expected,
                    'predicted': predicted,
                    'correct': is_correct,
                    'confidence': confidence,
                    'probabilities': probabilities
                })
                break
        
        # Print result
        status = "✓" if is_correct else "❌"
        print(f"{status} Test {i+1:2d}: '{text[:60]}{'...' if len(text) > 60 else ''}'")
        print(f"    Expected: {expected:8} | Predicted: {predicted:8} | Confidence: {confidence:.3f}")
        print(f"    Probabilities: {probabilities}")
        print()
    
    # Calculate overall accuracy
    overall_accuracy = (correct_predictions / total_predictions) * 100
    
    print("=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Overall Accuracy: {correct_predictions}/{total_predictions} ({overall_accuracy:.1f}%)")
    print()
    
    # Category-wise performance
    for category, results in categories.items():
        if results:
            correct = sum(1 for r in results if r['correct'])
            total = len(results)
            accuracy = (correct / total) * 100
            print(f"{category.replace('_', ' ').title():15}: {correct:2d}/{total:2d} ({accuracy:5.1f}%)")
    
    print()
    print("=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    
    # Analyze sarcasm detection
    sarcastic_results = categories['sarcastic']
    sarcasm_detected = sum(1 for r in sarcastic_results if r['predicted'] == 'negative')
    print(f"Sarcasm Detection: {sarcasm_detected}/{len(sarcastic_results)} cases correctly identified as negative")
    
    # Analyze mixed tone handling
    mixed_results = categories['mixed_tone']
    mixed_neutral = sum(1 for r in mixed_results if r['predicted'] == 'neutral')
    print(f"Mixed Tone Handling: {mixed_neutral}/{len(mixed_results)} cases identified as neutral")
    
    # Show confidence distribution
    all_confidences = [r['confidence'] for results in categories.values() for r in results]
    avg_confidence = sum(all_confidences) / len(all_confidences)
    print(f"Average Confidence: {avg_confidence:.3f}")
    
    # Show problematic cases
    print("\nProblematic Cases (incorrect predictions):")
    for category, results in categories.items():
        incorrect = [r for r in results if not r['correct']]
        if incorrect:
            print(f"\n{category.replace('_', ' ').title()}:")
            for r in incorrect:
                print(f"  - '{r['text'][:50]}...' | Expected: {r['expected']} | Got: {r['predicted']}")
    
    return overall_accuracy, categories

if __name__ == "__main__":
    test_mixed_tone_examples()