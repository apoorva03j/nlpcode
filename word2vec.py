import nltk
import re
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

nltk.download("punkt")

text = """Natural Language Processing is a subfield of Artificial Intelligence.
Word embeddings like Word2Vec help computers understand semantic meaning of words.
Machine learning is widely used in NLP applications."""

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    return tokens

tokens = preprocess(text)
print("Tokens:", tokens)

sentences = [tokens]
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1, epochs=100)

print("\nSimilarity between 'language' and 'processing':", model.wv.similarity("language", "processing"))
print("Most similar to 'learning':", model.wv.most_similar("learning"))
print("\nVector for 'nlp':\n", model.wv['nlp'])
