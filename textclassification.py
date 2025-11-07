import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

texts = [
    "The team won the football match",
    "Government passes new policy",
    "Cricket world cup starts next week",
    "The president gave a speech",
    "New AI technology is emerging",
    "Basketball season has started",
    "The minister announced reforms",
    "Advances in machine learning are rapid"
]
labels = ["Sports", "Politics", "Sports", "Politics", "Technology", "Sports", "Politics", "Technology"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

while True:
    user_input = input("\nEnter a sentence to classify (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Exiting... Goodbye!")
        break
    user_vector = vectorizer.transform([user_input])
    prediction = clf.predict(user_vector)
    print(f"Predicted Category: {prediction[0]}")
