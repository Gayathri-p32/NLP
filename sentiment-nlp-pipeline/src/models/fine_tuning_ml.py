import pandas as pd
import time 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV


#1. Setup Data
df = pd.read_csv("data/raw/imdb_train_subset.csv")
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf.fit_transform(df['text'])
y = df['label']
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#2. Define the "Search Space"{The Knobs}
# Grid/Random use dictonaries; Bayes often uses specific range objects
param_grid = {
    "n_estimators":[50,100,150],
    "max_depth":[10,20,None],
    "min_samples_split":[2,5]
}

def run_tuner(name, searcher):
    print(f"\n Starting {name} ... ")
    start = time.time()
    searcher.fit(X_train,y_train)
    end = time.time()

    best_model = searcher.best_estimator_
    acc = accuracy_score(y_test, best_model.predict(X_test))

    print(f" {name} Finished in {end-start:.2f}s")
    print(f" Best params: {searcher.best_params_}")
    print(f"Test accuracy:{acc:.4f}")
    return acc 

#initialize the three techniques
#Grid Search (Exhaustive -3X3x2 = 18 combinations)

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid,cv=3)

#random Search (Pick 10 random combinations)
random = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid,n_iter=10, cv=3, random_state=42)

# Bayesian Search (Smart search - 10 iterations)
# Note: Bayes treats 'None' differently, so we use a high number for depth
bayes_params = {
    'n_estimators': (50, 150),
    'max_depth': (10, 50),
    'min_samples_split': (2, 5)
}
bayes = BayesSearchCV(RandomForestClassifier(random_state=42), bayes_params, n_iter=10, cv=3, random_state=42)

# 4. Execution
results = {}
results['Grid'] = run_tuner("Grid Search", grid)
results['Random'] = run_tuner("Random Search", random)
results['Bayesian'] = run_tuner("Bayesian Search", bayes)

print("\n" + "="*30)
print("FINAL LEADERBOARD")
print("="*30)
for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{k}: {v:.4f}")