import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import RegexpTagger, word_tokenize, pos_tag

sentence = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(sentence)

patterns = [
    (r'.*ing$', 'VBG'),
    (r'.*ed$', 'VBD'),
    (r'.*es$', 'VBZ'),
    (r'^-?[0-9]+$', 'CD'),
    (r'.*', 'NN')
]

rule_based_tagger = RegexpTagger(patterns)
rule_based_tags = rule_based_tagger.tag(tokens)
stochastic_tags = pos_tag(tokens)

print("Input Sentence:")
print(sentence)
print("\nRule-Based POS Tags:")
print(rule_based_tags)
print("\nStochastic POS Tags (NLTK Perceptron Tagger):")
print(stochastic_tags)
