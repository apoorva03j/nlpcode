import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

texts = [
    "I love this new phone, it’s amazing!",
    "Worst service ever, I’m so disappointed",
    "The concert was fantastic, had a great time",
    "I hate this product, total waste of money",
    "Not bad, but could be better",
    "Absolutely wonderful experience!",
    "This is the most boring show I’ve ever watched",
    "I am happy with the purchase",
    "Terrible quality, would not recommend",
    "It’s okay, nothing special"
]
labels = [2, 0, 2, 0, 1, 2, 0, 2, 0, 1]  # 0=Negative, 1=Neutral, 2=Positive

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_texts, test_texts, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)
train_dataset = SentimentDataset(train_texts, y_train, tokenizer)
test_dataset = SentimentDataset(test_texts, y_test, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
optimizer = AdamW(model.parameters(), lr=2e-5)

model.train()
for batch in train_loader:
    optimizer.zero_grad()
    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
    loss = outputs.loss
    loss.backward()
    optimizer.step()

model.eval()
preds, true_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(batch['labels'].cpu().numpy())

print("Accuracy:", accuracy_score(true_labels, preds))
print("Classification Report:\n", classification_report(true_labels, preds, target_names=["Negative", "Neutral", "Positive"]))

user_text = input("\nEnter a social media post to analyze sentiment: ")
encoding = tokenizer(user_text, truncation=True, padding='max_length', max_length=64, return_tensors='pt')
with torch.no_grad():
    output = model(**encoding)
    prediction = torch.argmax(output.logits, dim=1).item()

sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
print("Predicted Sentiment:", sentiment_map[prediction])
