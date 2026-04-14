import os
import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure all resources are present
nltk.download('punkt_tab', quiet=True) # The 2026 punkt version
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('stopwords', quiet=True)

class IMDBPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def clean_and_tag(self, text):
        # 1. HTML & Punctuation Cleanup
        text = re.sub(r'<.*?>', ' ', str(text).lower())
        text = re.sub(r'[^\w\s]', '', text)
        
        # 2. Tokenization
        tokens = word_tokenize(text)
        
        # 3. Filtering & Tagging
        filtered_tokens = [w for w in tokens if w not in self.stop_words]
        pos_tags = nltk.pos_tag(filtered_tokens)
        
        return filtered_tokens, pos_tags

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("data/processed", exist_ok=True)
    
    cleaner = IMDBPreprocessor()
    
    # Let's test it on one review from your newly downloaded CSV
    try:
        df = pd.read_csv("data/raw/imdb_train_subset.csv")
        sample_text = df['text'].iloc[0]
        
        tokens, tags = cleaner.clean_and_tag(sample_text)
        
        print("\nPREPROCESSING SUCCESSFUL")
        print(f"Original (first 50 chars): {sample_text[:50]}...")
        print(f"Cleaned Tokens (first 10): {tokens[:10]}")
        print(f"POS Tags (first 5): {tags[:5]}")
        
    except FileNotFoundError:
        print("Error: 'data/raw/imdb_train_subset.csv' not found. Run load_data.py first.")