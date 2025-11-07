import numpy as np
import nltk
nltk.download('words')
from nltk.corpus import words

def edit_distance(str1, str2):
    m, n = len(str1), len(str2)
    dp = np.zeros((m+1, n+1), dtype=int)
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

word_list = words.words()
misspelled_word = "speling"

distances = [(edit_distance(misspelled_word, w), w) for w in word_list]
distances.sort()
print("Misspelled Word:", misspelled_word)
print("Top 5 Suggested Corrections:")
for dist, word in distances[:5]:
    print(f"{word} (Edit Distance: {dist})")
