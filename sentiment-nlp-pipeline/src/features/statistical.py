import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

#Sample Data
reviews = [
    "The acting was great and the movie was great",
    "The movie was bad and the acting was bad",
    "The acting was great but the plot was bad"
]

#1. Bag of Words (Counts)
bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(reviews)

#2.TF-IDF(Weighted Counts)
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(reviews)

#Convert to DataFrame to see the "Math"
print("----- Bag of Words Matrox -------")
print(pd.DataFrame(X_bow.toarray(), columns = bow_vectorizer.get_feature_names_out()))

print("\n ------TF-iDF Mattrix -----")
print(pd.DataFrame(X_tfidf.toarray(), columns = tfidf_vectorizer.get_feature_names_out()))
