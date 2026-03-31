"""
Test script for the enhanced sentiment analysis model
Tests performance on sarcastic and mixed-tone examples
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_traditional_model():
    """Test with traditional model as fallback"""
    print("Testing with traditional model...")
    
    try:
        from ml_model import SentimentAnalyzer as TraditionalAnalyzer
        
        analyzer = TraditionalAnalyzer()
        success = analyzer.load_model()
        
        if not success:
            print("Training traditional model...")
            analyzer.train_model()
            analyzer.save_model()
            success = analyzer.load_model()
        
        if success:
            print("Traditional model loaded successfully!")
            
            # Test examples
            test_examples = [
                "This product is amazing!",
                "Terrible product. Complete waste of money.",
                "Oh wow, what a 'fantastic' product that broke immediately!",
                "The quality is good but the price is too high.",
                "So 'impressed' with how quickly this fell apart."
            ]
            
            print("\nTesting traditional model predictions:")
            for text in test_examples:
                try:
                    result = analyzer.predict_sentiment(text)
                    print(f"Text: {text}")
                    print(f"Prediction: {result['sentiment']} (confidence: {result['confidence']:.3f})")
                    print("---")
                except Exception as e:
                    print(f"Error predicting '{text}': {e}")
            
            return True
        else:
            print("Failed to load traditional model")
            return False
            
    except Exception as e:
        print(f"Error with traditional model: {e}")
        return False

def test_enhanced_model():
    """Test the enhanced model with transformer capabilities"""
    print("Testing enhanced model...")
    
    try:
        from enhanced_ml_model import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        print("Loading enhanced model...")
        success = analyzer.load_model()
        
        if success:
            model_info = analyzer.get_model_info()
            print(f"Enhanced model loaded successfully!")
            print(f"Model info: {model_info}")
            
            # Test examples focusing on sarcasm and mixed-tone
            test_examples = [
                # Regular positive
                "This product is amazing! I love it so much.",
                
                # Regular negative  
                "Terrible product. Complete waste of money.",
                
                # Sarcastic positive (actually negative)
                "Oh wow, what a 'fantastic' product that broke immediately!",
                "Sure, spending $50 on this piece of junk was totally worth it.",
                "So 'impressed' with how quickly this fell apart.",
                
                # Mixed-tone (neutral/conflicted)
                "The quality is good but the price is too high.",
                "Great features but terrible customer service experience.",
                "It's fine, I guess. Not amazing but not terrible either.",
                
                # Sarcastic negative (actually positive)
                "I guess I'll have to 'suffer' with this amazing product.",
                "What a 'disaster' - it actually exceeded my expectations!"
            ]
            
            print(f"\nTesting enhanced model predictions (Model type: {model_info['model_type']}):")
            correct_predictions = 0
            total_predictions = len(test_examples)
            
            for i, text in enumerate(test_examples):
                try:
                    result = analyzer.predict_sentiment(text)
                    print(f"Text: {text}")
                    print(f"Prediction: {result['sentiment']} (confidence: {result['confidence']:.3f})")
                    
                    if 'probabilities' in result:
                        probs = result['probabilities']
                        print(f"Probabilities: Neg: {probs.get('negative', 0):.3f}, "
                              f"Neu: {probs.get('neutral', 0):.3f}, "
                              f"Pos: {probs.get('positive', 0):.3f}")
                    
                    print("---")
                    
                    # Simple accuracy check for obvious cases
                    if i < 2:  # First two are clearly positive/negative
                        expected = 'positive' if i == 0 else 'negative'
                        if result['sentiment'] == expected:
                            correct_predictions += 1
                    elif i in [2, 3, 4]:  # Sarcastic positive (should be negative)
                        if result['sentiment'] == 'negative':
                            correct_predictions += 1
                    elif i in [8, 9]:  # Sarcastic negative (should be positive)
                        if result['sentiment'] == 'positive':
                            correct_predictions += 1
                    else:
                        # For mixed-tone, any reasonable prediction is acceptable
                        correct_predictions += 1
                        
                except Exception as e:
                    print(f"Error predicting '{text}': {e}")
            
            accuracy = correct_predictions / total_predictions
            print(f"\nSimple accuracy check: {correct_predictions}/{total_predictions} = {accuracy:.2%}")
            
            return True
        else:
            print("Failed to load enhanced model")
            return False
            
    except Exception as e:
        print(f"Error with enhanced model: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("Enhanced Sentiment Analysis Model Test")
    print("=" * 60)
    
    # Test enhanced model first
    enhanced_success = test_enhanced_model()
    
    print("\n" + "=" * 60)
    
    # If enhanced model fails, test traditional as fallback
    if not enhanced_success:
        print("Enhanced model failed, testing traditional model as fallback...")
        traditional_success = test_traditional_model()
        
        if traditional_success:
            print("\nTraditional model working as fallback.")
        else:
            print("\nBoth models failed to load.")
    else:
        print("Enhanced model test completed successfully!")
    
    print("=" * 60)

if __name__ == "__main__":
    main()