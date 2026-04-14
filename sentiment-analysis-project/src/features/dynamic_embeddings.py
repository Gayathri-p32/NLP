from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = ["The lead actor was amazing",
             "The story felt like a lead weight"]

embeddings = model.encode(sentences)

#these vectors are 384 dimensions long 
print(f"Vectors for sentence 1 (first 5 values):  {embeddings[0][:5]}")
print(f"Vector for sentence 2 (firsy 5 values): {embeddings[1][:5]}")

# 3. Prove they are different
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity([embeddings[0]], [embeddings[1]])
print(f"\nSimilarity between the two 'lead' sentences: {sim[0][0]:.4f}")