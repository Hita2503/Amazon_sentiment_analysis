import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            strip_accents='ascii'
        )
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0
        )
        self.is_trained = False
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def create_sample_dataset(self):
        """Create a sample Amazon reviews dataset for training"""
        # Sample Amazon-style reviews with sentiment labels
        sample_data = [
            # Positive reviews
            ("This product is absolutely amazing! The quality exceeded my expectations and shipping was super fast.", "positive"),
            ("Love this item! Great value for money and works perfectly. Highly recommend to everyone!", "positive"),
            ("Excellent quality product. Arrived quickly and exactly as described. Very satisfied with purchase.", "positive"),
            ("Outstanding customer service and fantastic product quality. Will definitely buy again!", "positive"),
            ("Perfect! Exactly what I was looking for. Great build quality and fast delivery.", "positive"),
            ("Amazing product! Works great and the price is unbeatable. Five stars!", "positive"),
            ("Superb quality and excellent packaging. Arrived earlier than expected. Love it!", "positive"),
            ("This is the best purchase I've made in a long time. Highly recommended!", "positive"),
            ("Fantastic product with great features. Easy to use and very reliable.", "positive"),
            ("Excellent value for money. Product works as advertised and shipping was fast.", "positive"),
            ("Great quality item that works perfectly. Very happy with this purchase.", "positive"),
            ("Amazing customer service and quick delivery. Product is exactly as described.", "positive"),
            ("Love the design and functionality. Great product at a reasonable price.", "positive"),
            ("Perfect product! High quality materials and excellent craftsmanship.", "positive"),
            ("Outstanding performance and great value. Would definitely recommend this product.", "positive"),
            
            # Negative reviews
            ("Terrible product! Poor quality and arrived damaged. Complete waste of money.", "negative"),
            ("Worst purchase ever. Product doesn't work and customer service is unhelpful.", "negative"),
            ("Cheap quality materials that broke after one day. Don't buy this junk!", "negative"),
            ("Horrible experience. Product arrived late and was completely different from description.", "negative"),
            ("Poor quality control. Item was defective and return process was a nightmare.", "negative"),
            ("Overpriced garbage. Product failed within a week and no response from seller.", "negative"),
            ("Terrible build quality. Looks nothing like the pictures and feels very cheap.", "negative"),
            ("Worst customer service ever. Product is faulty and they won't provide refund.", "negative"),
            ("Complete scam! Product is fake and doesn't work at all. Avoid at all costs!", "negative"),
            ("Poor packaging resulted in damaged product. Quality is much worse than expected.", "negative"),
            ("Disappointing purchase. Product broke immediately and seller won't respond.", "negative"),
            ("Terrible quality for the price. Would not recommend to anyone.", "negative"),
            ("Product is completely useless. Waste of time and money. Very disappointed.", "negative"),
            ("Poor construction and materials. Product failed after minimal use.", "negative"),
            ("Awful experience from start to finish. Product and service both terrible.", "negative"),
            
            # Neutral reviews
            ("Product is okay, nothing special. Works as described but not impressive.", "neutral"),
            ("Average quality for the price. Does what it's supposed to do.", "neutral"),
            ("It's fine. Not great, not terrible. Just an ordinary product.", "neutral"),
            ("Decent product but could be better. Meets basic expectations.", "neutral"),
            ("Product works but build quality could be improved. It's acceptable.", "neutral"),
            ("Fair value for money. Product is functional but not outstanding.", "neutral"),
            ("Standard quality item. Nothing to complain about but nothing special either.", "neutral"),
            ("Product is adequate for basic needs. Could use some improvements.", "neutral"),
            ("Reasonable price and decent quality. Does the job but not exceptional.", "neutral"),
            ("It's an okay product. Works fine but there are probably better options.", "neutral"),
            ("Average product with standard features. Nothing remarkable about it.", "neutral"),
            ("Decent build quality but design could be better. It's acceptable.", "neutral"),
            ("Product meets expectations but doesn't exceed them. It's fine.", "neutral"),
            ("Fair quality for the price point. Does what it claims to do.", "neutral"),
            ("Standard product that works as intended. Nothing more, nothing less.", "neutral"),
        ]
        
        # Create additional synthetic reviews to increase dataset size
        positive_templates = [
            "Great {product}! {positive_adj} quality and {positive_feature}.",
            "Love this {product}! {positive_adj} and {positive_feature}.",
            "Excellent {product} with {positive_feature}. {positive_adj}!",
            "Amazing {product}! {positive_feature} and {positive_adj}.",
            "Perfect {product}! {positive_adj} and great {positive_feature}."
        ]
        
        negative_templates = [
            "Terrible {product}! {negative_adj} quality and {negative_feature}.",
            "Hate this {product}! {negative_adj} and {negative_feature}.",
            "Awful {product} with {negative_feature}. {negative_adj}!",
            "Horrible {product}! {negative_feature} and {negative_adj}.",
            "Worst {product}! {negative_adj} and terrible {negative_feature}."
        ]
        
        neutral_templates = [
            "Okay {product}. {neutral_adj} quality and {neutral_feature}.",
            "Average {product} with {neutral_feature}. {neutral_adj}.",
            "Standard {product}. {neutral_adj} and {neutral_feature}.",
            "Decent {product} but {neutral_feature}. {neutral_adj}.",
            "Fair {product} with {neutral_adj} {neutral_feature}."
        ]
        
        products = ["item", "product", "purchase", "device", "gadget"]
        positive_adjs = ["fantastic", "excellent", "amazing", "outstanding", "superb"]
        negative_adjs = ["terrible", "awful", "horrible", "disappointing", "poor"]
        neutral_adjs = ["average", "okay", "decent", "standard", "fair"]
        positive_features = ["fast shipping", "great value", "perfect fit", "easy setup", "durable build"]
        negative_features = ["slow delivery", "poor value", "wrong size", "difficult setup", "cheap build"]
        neutral_features = ["standard shipping", "fair value", "adequate fit", "normal setup", "average build"]
        
        # Generate additional samples
        for _ in range(20):
            # Positive
            template = np.random.choice(positive_templates)
            review = template.format(
                product=np.random.choice(products),
                positive_adj=np.random.choice(positive_adjs),
                positive_feature=np.random.choice(positive_features)
            )
            sample_data.append((review, "positive"))
            
            # Negative
            template = np.random.choice(negative_templates)
            review = template.format(
                product=np.random.choice(products),
                negative_adj=np.random.choice(negative_adjs),
                negative_feature=np.random.choice(negative_features)
            )
            sample_data.append((review, "negative"))
            
            # Neutral
            template = np.random.choice(neutral_templates)
            review = template.format(
                product=np.random.choice(products),
                neutral_adj=np.random.choice(neutral_adjs),
                neutral_feature=np.random.choice(neutral_features)
            )
            sample_data.append((review, "neutral"))
        
        return pd.DataFrame(sample_data, columns=['review', 'sentiment'])
    
    def train_model(self, df=None):
        """Train the sentiment analysis model"""
        if df is None:
            print("Creating sample dataset...")
            df = self.create_sample_dataset()
        
        print(f"Training on {len(df)} reviews...")
        
        # Preprocess the text data
        df['cleaned_review'] = df['review'].apply(self.preprocess_text)
        
        # Prepare features and labels
        X = df['cleaned_review']
        y = df['sentiment']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Vectorize the text
        print("Vectorizing text data...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train the model
        print("Training logistic regression model...")
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        
        # Save the model and vectorizer
        self.save_model()
        
        return accuracy
    
    def predict_sentiment(self, text):
        """Predict sentiment for a given text"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Please train the model first.")
        
        # Preprocess the text
        cleaned_text = self.preprocess_text(text)
        
        # Vectorize the text
        text_tfidf = self.vectorizer.transform([cleaned_text])
        
        # Get prediction and probability
        prediction = self.model.predict(text_tfidf)[0]
        probabilities = self.model.predict_proba(text_tfidf)[0]
        
        # Get the confidence (probability of the predicted class)
        class_names = self.model.classes_
        confidence = max(probabilities)
        
        # Create probability dictionary
        prob_dict = dict(zip(class_names, probabilities))
        
        return {
            'sentiment': prediction,
            'confidence': float(confidence),
            'probabilities': {k: float(v) for k, v in prob_dict.items()}
        }
    
    def save_model(self, model_path='sentiment_model.pkl', vectorizer_path='tfidf_vectorizer.pkl'):
        """Save the trained model and vectorizer"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, model_path='sentiment_model.pkl', vectorizer_path='tfidf_vectorizer.pkl'):
        """Load a pre-trained model and vectorizer"""
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.is_trained = True
            print("Model and vectorizer loaded successfully!")
            return True
        else:
            print("Model files not found. Please train the model first.")
            return False

# Initialize and train the model if this script is run directly
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Try to load existing model, if not found, train a new one
    if not analyzer.load_model():
        print("Training new model...")
        analyzer.train_model()
    
    # Test the model
    test_reviews = [
        "This product is amazing! Great quality and fast shipping.",
        "Terrible quality, broke after one day. Don't buy this.",
        "It's okay, nothing special but does the job."
    ]
    
    print("\nTesting the model:")
    for review in test_reviews:
        result = analyzer.predict_sentiment(review)
        print(f"Review: {review}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.3f})")
        print("---")