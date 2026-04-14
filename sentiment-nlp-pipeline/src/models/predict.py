import joblib

print("Loading model...")
model = joblib.load('models/sentiment_nb_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

def check_sentiment():
    print("\n--- IMDb Sentiment Checker ---")
    print("Type 'quit' to exit.")

    while True:
        user_input = input("\n Enter a movie review: ")
        if user_input.lower() == 'quit':
            break
        
        text_vectorized = tfidf.transform([user_input])

        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)

        label = "Positive" if prediction==1 else "Negative"
        confidence = probability[0][prediction]*100

        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    check_sentiment()