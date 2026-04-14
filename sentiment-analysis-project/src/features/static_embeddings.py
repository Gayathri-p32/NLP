import gensim.downloader as api 

# 1, Load Pre-trained Word2Vec 
print("Loading Word2Vec...")
w2v_model = api.load("word2vec-google-news-300")

#2. Load pre-trained GloVe (Twitter/Wikipedia)
print(" Loading GloVe...")
glove_model = api.load("glove-wiki-gigaword-100")

word = "boring"
print(f"\nWord2Vec similar to '{word}':", [w for w, s in w2v_model.most_similar(word)[:3]])
print(f"GloVe similar to '{word}':", [w for w, s in glove_model.most_similar(word)[:3]])