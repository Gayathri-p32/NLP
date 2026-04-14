# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score

# # 1. Load your IMDb Data
# print("📂 Loading data...")
# df = pd.read_csv("data/raw/imdb_train_subset.csv") 
# sentences = df['text'].astype(str).tolist()
# y = df['label'].values

# # 2. Generate BERT Embeddings (The "Features")
# print("🧠 Encoding reviews with BERT (this is the slow part)...")
# model = SentenceTransformer('all-MiniLM-L6-v2')
# X = model.encode(sentences, show_progress_bar=True)

# # 3. Split into Train and Test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 4. Train the Random Forest
# print("🌲 Growing the Random Forest (100 trees)...")
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)

# # 5. See the results
# y_pred = rf.predict(X_test)
# print("\n" + "="*30)
# print(f"🌲 RANDOM FOREST + BERT ACCURACY: {accuracy_score(y_test, y_pred):.4f}")
# print("="*30)
# print(classification_report(y_test, y_pred))


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Data
df = pd.read_csv("data/raw/imdb_train_subset.csv")
X_text = df['text']
y = df['label']

# 2. TF-IDF (Limiting features so the Forest doesn't explode)
print("🧮 Vectorizing with TF-IDF (Top 2000 words)...")
tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
X = tfidf.fit_transform(X_text)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Random Forest
print("🌲 Growing Random Forest on TF-IDF features...")
# n_jobs=-1 uses all your CPU cores to speed it up!
rf_tfidf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_tfidf.fit(X_train, y_train)

# 5. Results
y_pred = rf_tfidf.predict(X_test)
print("\n" + "="*30)
print(f"🌲 TF-IDF + RANDOM FOREST ACCURACY: {accuracy_score(y_test, y_pred):.4f}")
print("="*30)
print(classification_report(y_test, y_pred))