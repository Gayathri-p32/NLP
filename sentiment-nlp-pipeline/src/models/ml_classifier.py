import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from sentence_transformers import SentenceTransformer

def train_baseline(csv_path, vectorizer_path):
    df = pd.read_csv(csv_path)
    X_text = df['text']
    y = df['label']
    print(X_text.head())

    #2. Vectorize (Using our Statustical TF-IDF)
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    # 4. Train Model
    print("Training Logistic Regression...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nDetailed Report:\n", classification_report(y_test, y_pred))
    
    return model

def train_on_bert(csv_path):
    # 1. Load the original labels
    df = pd.read_csv(csv_path)
    sentences = df['text'].astype(str).tolist()
    y = df['label'].values # Labels (0 or 1)
    
    # 2. Extract BERT Features (The "Heavy" part)
    print("Initializing BERT (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Encoding {len(sentences)} reviews...")
    X = model.encode(sentences, show_progress_bar=True)
    # 3. Split (Use the same random_state for a fair comparison!)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train
    print("Training Logistic Regression on BERT Embeddings...")
    model = LogisticRegression(max_iter=1000) # BERT takes more iterations to converge
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred = model.predict(X_test)
    print(f"\nBERT Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nBERT Detailed Report:\n", classification_report(y_test, y_pred))


   

if __name__ == "__main__":
    train_baseline("data/raw/imdb_train_subset.csv", "models/tfidf.pkl")
    train_on_bert(
        "data/raw/imdb_train_subset.csv")