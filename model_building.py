import pandas as pd
import numpy as np
import pickle
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE

# 1. Load Data
df = pd.read_excel('amazon.xlsx')
df.drop_duplicates(inplace=True)
df['Review'] = df['Review'].fillna('')

# 2. Preprocessing
def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'<.*?>', '', text)
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return " ".join(text.split())

df['Clean_Review'] = df['Review'].apply(preprocess_text)

# 3. Feature Engineering
X = df['Clean_Review']
y = df['Star']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)

# 4. Handle Imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# 5. Build and Train Model
# Using alpha=0.1 found during grid search
model = MultinomialNB(alpha=0.1)
model.fit(X_train_resampled, y_train_resampled)

# 6. Save Assets
with open('naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Model and Vectorizer have been built and saved successfully.")
