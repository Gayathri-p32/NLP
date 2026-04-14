import nltk
from nltk.util import ngrams
from collections import Counter
import sys
import os

# This line adds the project root to the system path automatically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.prepocessing.cleaner import IMDBPreprocessor

class FeatureExtractor:
    def __init__(self):
        # Need these for the tagging logic
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)

    def extract_features(self,tokens, tags):
        features = {}

        # 1. Bigram Extraction (Semantic)
        # Captures "not good", "really liked"
        bi_grams = list(ngrams(tokens,2))
        for bg in bi_grams:
            features[f"ngram_{bg[0]}_{bg[1]}"] = 1
        
        # 2. POS Filtering (Syntactic)
        # We specifically extract Adjectives (JJ) and Adverbs (RB)
        for word, tag in tags:
            if tag.startswith('JJ') or tag.startswith('RB'):
                features[f"pos_{word}_{tag}"]=1
        return features
    
if __name__ == "__main__":
    from src.prepocessing.cleaner import IMDBPreprocessor

    cleaner = IMDBPreprocessor()
    extractor = FeatureExtractor()

    sample = "The script was not good, but the acting was truly amazing"
    tokens, tags = cleaner.clean_and_tag(sample)

    final_features = extractor.extract_features(tokens,tags)

    print(f"total features extracted: {len(final_features)}")
    print("Example features: ", list(final_features.keys())[:5])