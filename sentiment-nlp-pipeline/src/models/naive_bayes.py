import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
#loading and Prepare data 
print("loading data ....")
df = pd.read_csv("data/raw/imdb_train_subset.csv")

#2. TF-IDF Vectorization 
print("Vectorizing text (5,000 features)")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
X = tfidf.fit_transform(df['text'])
y = df['label']

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

models = {
    "Naive Bayes" : MultinomialNB(),
    "Linear SVM": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
    "Logistic Regression":LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

print("\n Starting the tournament")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name]=acc
    print(f"{name:20} Accuracy: {acc:.4f}")

print("\n "+ "="*30)
print("Final leaderboard")
print("="*30)
sorted_results = sorted(results.items(), key = lambda x:x[1], reverse=True)
for i, (name,acc) in enumerate(sorted_results):
    rank = "First" if i == 0 else "Second" if i == 1 else "Third" if i == 2 else "  "
    print(f"{rank} {name:20}: {acc:.4f}")

os.makedirs('models', exist_ok=True)
joblib.dump(models["Naive Bayes"], 'models/sentiment_nb_model.pkl')
joblib.dump(tfidf,"models/tfidf_vectorizer.pkl")
print("\n SUCCESS: Champion model and Vectorizer saved to /models folder.")
