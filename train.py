import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load datasets
fake = pd.read_csv("data/fake.csv")
true = pd.read_csv("data/true.csv")

# Labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
df = pd.concat([fake, true])

# Keep required columns
df = df[["text", "label"]]

# Clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', str(text))
    return text.lower()

df["text"] = df["text"].apply(clean_text)

# Convert text to numbers
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save files
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model Saved Successfully")