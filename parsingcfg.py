import nltk
from nltk import CFG

grammar = CFG.fromstring("""
    S -> NP VP
    NP -> Det N | Det Adj N | Det Adj Adj N
    VP -> V NP | V NP PP
    PP -> P NP

    Det -> 'the' | 'a'
    Adj -> 'quick' | 'brown' | 'lazy' | 'small'
    N -> 'fox' | 'dog' | 'cat' | 'park'
    V -> 'jumps' | 'runs' | 'sleeps' | 'sees'
    P -> 'in' | 'on' | 'over'
""")

parser = nltk.ChartParser(grammar)
sentence = ['the', 'quick', 'brown', 'fox', 'sees', 'a', 'dog']

for tree in parser.parse(sentence):
    print("\nBracketed Tree format:\n")
    print(tree)
    print("\nPretty Printed Tree:\n")
    tree.pretty_print()
