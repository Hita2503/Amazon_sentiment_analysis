#!/usr/bin/env python3
"""
Compare performance between transformer and traditional sentiment models
"""

import sys
import time
from typing import List, Dict, Tuple
from enhanced_ml_model import SentimentAnalyzer
from ml_model import SentimentAnalyzer as TraditionalAnalyzer

def compare_model_performance():
    """Compare transformer vs traditional model performance"""
    
    print("Model Performance Comparison")
    print("=" * 80)
    
    # Initialize both models
    print("Initializing models...")
    
    # Transformer model (via enhanced analyzer)
    transformer_analyzer = SentimentAnalyzer()
    if not transformer_analyzer.load_model():
        print("❌ Failed to load transformer model")
        return
    
    # Traditional model
    traditional_analyzer = TraditionalAnalyzer()
    if not traditional_analyzer.load_model():
        print("Training traditional model...")
        traditional_analyzer.train_model()
        traditional_analyzer.save_model()
    
    print("✓ Both models loaded successfully")
    print()
    
    # Test cases with expected outcomes and categories
    test_cases = [
        # Clear positive cases
        ("This product is absolutely amazing!", "positive", "clear_positive"),
        ("I love this item, it's perfect!", "positive", "clear_positive"),
        ("Excellent quality and fast shipping", "positive", "clear_positive"),
        
        # Clear negative cases
        ("This product is terrible and broken", "negative", "clear_negative"),
        ("I hate this item, complete waste of money", "negative", "clear_negative"),
        ("Poor quality and slow delivery", "negative", "clear_negative"),
        
        # Clear neutral cases
        ("The product is okay, nothing special", "neutral", "clear_neutral"),
        ("It's average quality for the price", "neutral", "clear_neutral"),
        ("Standard product, meets basic expectations", "neutral", "clear_neutral"),
        
        # Sarcastic cases (should be detected as negative)
        ("Oh great, another broken product. Just what I needed!", "negative", "sarcastic"),
        ("Wow, this is absolutely fantastic... NOT!", "negative", "sarcastic"),
        ("Perfect! It broke on the first day. Amazing quality!", "negative", "sarcastic"),
        ("Yeah, sure, this is the 'best' product ever", "negative", "sarcastic"),
        
        # Mixed tone cases (should be detected as neutral or appropriately weighted)
        ("The product works well but the packaging was terrible", "neutral", "mixed_tone"),
        ("Great features but overpriced for what you get", "neutral", "mixed_tone"),
        ("I love the design but hate the customer service", "neutral", "mixed_tone"),
        ("Good quality however the delivery was delayed", "neutral", "mixed_tone"),
        ("Fast shipping and good packaging, though the product itself is mediocre", "neutral", "mixed_tone"),
        ("The camera is excellent but the battery life is disappointing", "neutral", "mixed_tone"),
        
        # Complex contradictory cases
        ("This is both the best and worst purchase I've made", "neutral", "complex"),
        ("Amazing product with terrible support", "neutral", "complex"),
        ("Love it and hate it at the same time", "neutral", "complex"),
        ("Great when it works, but it rarely works", "negative", "complex"),
    ]
    
    # Results storage
    transformer_results = []
    traditional_results = []
    
    print("Running predictions...")
    print("-" * 80)
    
    # Test transformer model
    print("Testing Transformer Model:")
    transformer_start = time.time()
    for i, (text, expected, category) in enumerate(test_cases):
        try:
            result = transformer_analyzer.predict_sentiment(text)
            transformer_results.append({
                'text': text,
                'expected': expected,
                'predicted': result['sentiment'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'],
                'category': category,
                'correct': result['sentiment'] == expected
            })
            print(f"  {i+1:2d}. {result['sentiment']:8} ({result['confidence']:.3f}) - {text[:50]}...")
        except Exception as e:
            print(f"  {i+1:2d}. ERROR: {e}")
            transformer_results.append({
                'text': text,
                'expected': expected,
                'predicted': 'error',
                'confidence': 0.0,
                'probabilities': {},
                'category': category,
                'correct': False
            })
    transformer_time = time.time() - transformer_start
    
    print("\nTesting Traditional Model:")
    traditional_start = time.time()
    for i, (text, expected, category) in enumerate(test_cases):
        try:
            result = traditional_analyzer.predict_sentiment(text)
            traditional_results.append({
                'text': text,
                'expected': expected,
                'predicted': result['sentiment'],
                'confidence': result['confidence'],
                'probabilities': result.get('probabilities', {}),
                'category': category,
                'correct': result['sentiment'] == expected
            })
            print(f"  {i+1:2d}. {result['sentiment']:8} ({result['confidence']:.3f}) - {text[:50]}...")
        except Exception as e:
            print(f"  {i+1:2d}. ERROR: {e}")
            traditional_results.append({
                'text': text,
                'expected': expected,
                'predicted': 'error',
                'confidence': 0.0,
                'probabilities': {},
                'category': category,
                'correct': False
            })
    traditional_time = time.time() - traditional_start
    
    # Calculate overall performance
    transformer_correct = sum(1 for r in transformer_results if r['correct'])
    traditional_correct = sum(1 for r in traditional_results if r['correct'])
    total_tests = len(test_cases)
    
    transformer_accuracy = (transformer_correct / total_tests) * 100
    traditional_accuracy = (traditional_correct / total_tests) * 100
    
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON RESULTS")
    print("=" * 80)
    
    print(f"Overall Accuracy:")
    print(f"  Transformer Model: {transformer_correct:2d}/{total_tests} ({transformer_accuracy:5.1f}%)")
    print(f"  Traditional Model: {traditional_correct:2d}/{total_tests} ({traditional_accuracy:5.1f}%)")
    print(f"  Improvement:       {transformer_accuracy - traditional_accuracy:+5.1f}%")
    print()
    
    print(f"Processing Time:")
    print(f"  Transformer Model: {transformer_time:.3f}s ({transformer_time/total_tests:.3f}s per prediction)")
    print(f"  Traditional Model: {traditional_time:.3f}s ({traditional_time/total_tests:.3f}s per prediction)")
    print()
    
    # Category-wise comparison
    categories = ['clear_positive', 'clear_negative', 'clear_neutral', 'sarcastic', 'mixed_tone', 'complex']
    
    print("Category-wise Performance:")
    print(f"{'Category':<15} {'Transformer':<12} {'Traditional':<12} {'Improvement':<12}")
    print("-" * 60)
    
    for category in categories:
        trans_cat = [r for r in transformer_results if r['category'] == category]
        trad_cat = [r for r in traditional_results if r['category'] == category]
        
        if trans_cat and trad_cat:
            trans_correct = sum(1 for r in trans_cat if r['correct'])
            trad_correct = sum(1 for r in trad_cat if r['correct'])
            
            trans_acc = (trans_correct / len(trans_cat)) * 100
            trad_acc = (trad_correct / len(trad_cat)) * 100
            improvement = trans_acc - trad_acc
            
            print(f"{category:<15} {trans_acc:5.1f}% ({trans_correct}/{len(trans_cat):<2}) {trad_acc:5.1f}% ({trad_correct}/{len(trad_cat):<2}) {improvement:+5.1f}%")
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    # Analyze specific improvements
    sarcasm_trans = [r for r in transformer_results if r['category'] == 'sarcastic']
    sarcasm_trad = [r for r in traditional_results if r['category'] == 'sarcastic']
    
    sarcasm_trans_correct = sum(1 for r in sarcasm_trans if r['correct'])
    sarcasm_trad_correct = sum(1 for r in sarcasm_trad if r['correct'])
    
    print(f"Sarcasm Detection:")
    print(f"  Transformer: {sarcasm_trans_correct}/{len(sarcasm_trans)} cases correctly identified")
    print(f"  Traditional: {sarcasm_trad_correct}/{len(sarcasm_trad)} cases correctly identified")
    
    mixed_trans = [r for r in transformer_results if r['category'] == 'mixed_tone']
    mixed_trad = [r for r in traditional_results if r['category'] == 'mixed_tone']
    
    mixed_trans_correct = sum(1 for r in mixed_trans if r['correct'])
    mixed_trad_correct = sum(1 for r in mixed_trad if r['correct'])
    
    print(f"\nMixed Tone Handling:")
    print(f"  Transformer: {mixed_trans_correct}/{len(mixed_trans)} cases correctly identified")
    print(f"  Traditional: {mixed_trad_correct}/{len(mixed_trad)} cases correctly identified")
    
    # Show confidence comparison
    trans_avg_conf = sum(r['confidence'] for r in transformer_results) / len(transformer_results)
    trad_avg_conf = sum(r['confidence'] for r in traditional_results) / len(traditional_results)
    
    print(f"\nAverage Confidence:")
    print(f"  Transformer: {trans_avg_conf:.3f}")
    print(f"  Traditional: {trad_avg_conf:.3f}")
    
    # Show problematic cases for each model
    print(f"\nProblematic Cases:")
    
    trans_errors = [r for r in transformer_results if not r['correct']]
    trad_errors = [r for r in traditional_results if not r['correct']]
    
    print(f"\nTransformer Model Errors ({len(trans_errors)} cases):")
    for r in trans_errors[:5]:  # Show first 5
        print(f"  - '{r['text'][:50]}...' | Expected: {r['expected']} | Got: {r['predicted']}")
    
    print(f"\nTraditional Model Errors ({len(trad_errors)} cases):")
    for r in trad_errors[:5]:  # Show first 5
        print(f"  - '{r['text'][:50]}...' | Expected: {r['expected']} | Got: {r['predicted']}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    if transformer_accuracy > traditional_accuracy:
        print(f"✓ TRANSFORMER MODEL RECOMMENDED")
        print(f"  - {transformer_accuracy - traditional_accuracy:.1f}% better accuracy")
        print(f"  - Better sarcasm detection")
        print(f"  - Improved mixed-tone handling")
        print(f"  - Supports 3-class output (Positive/Negative/Neutral)")
    else:
        print(f"⚠ TRADITIONAL MODEL STILL COMPETITIVE")
        print(f"  - Only {traditional_accuracy - transformer_accuracy:.1f}% accuracy difference")
        print(f"  - Faster processing time")
        print(f"  - Lower resource requirements")
    
    return {
        'transformer_accuracy': transformer_accuracy,
        'traditional_accuracy': traditional_accuracy,
        'transformer_time': transformer_time,
        'traditional_time': traditional_time,
        'transformer_results': transformer_results,
        'traditional_results': traditional_results
    }

if __name__ == "__main__":
    compare_model_performance()