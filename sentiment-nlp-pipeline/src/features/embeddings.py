import pandas as pd 
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

#Ensure you have the tokenizer 
nltk.download('punkt', quiet=True)

def train_custom_w2v(csv_path):
    #1.Load and tokenize
    df = pd.read_csv(csv_path)

    tokenized_sentence = [word_tokenize(str(text).lower()) for text in df['text']]

    #2. Train the model 
    # vector_size = 100: Each word is list of 100 numbers'
    # window=5: look at 5 words context
    #min_couny=2: Ignore words that appear only once (likely typos)
    print(" training Word2Vec on IMDb data ...")
    model = Word2Vec(sentences=tokenized_sentence,
                     vector_size=100,
                     window=5,
                     min_count=2,
                     workers=4)
    #3. Test the "Film Brain"
    word="script"
    if word in model.wv:
        similar = model.wv.most_similar(word,topn=3)
        print(f"\n Words most similar to '{word}' in your dataset:")
        for w,score in similar:
            print(f"-{w}: {score:.4f}")
    return model

model = train_custom_w2v("data/raw/imdb_train_subset.csv")
