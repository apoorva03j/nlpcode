import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

text = "Natural Language Processing (NLP) is fun! Let's tokenize this sentence."

def basic_tokenizer(text):
    tokens = re.split(r'(\W+)', text)
    tokens = [t for t in tokens if t.strip()]
    return tokens

nltk_tokens = word_tokenize(text)

print("Input Text:")
print(text)
print("\nTokens using Basic Regex Tokenizer:")
print(basic_tokenizer(text))
print("\nTokens using NLTK Tokenizer:")
print(nltk_tokens)
