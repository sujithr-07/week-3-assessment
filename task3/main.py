import re
import string
import numpy as np
import pandas as pd
import nltk

from nltk.corpus import movie_reviews

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# LOAD LABELLED TEXT DATASET

print("\n1. Loading dataset...")

nltk.download("movie_reviews")

texts = []
labels = []

for category in movie_reviews.categories():

    for file_id in movie_reviews.fileids(category):

        text = movie_reviews.raw(file_id)

        texts.append(text)
        labels.append(category)

df = pd.DataFrame({
    "text": texts,
    "label": labels
})

print("Total dataset size:", len(df))
print("\nDataset sample:")
print(df.head())


# CLEAN TEXT

print("\n2. Cleaning text...")

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["clean_text"] = df["text"].apply(clean_text)

# Create short texts (first 40 words)
df["short_text"] = df["clean_text"].apply(
    lambda text: " ".join(text.split()[:40])
)

print("Cleaned text:")
print(df["short_text"].iloc[0])


# TRAIN-TEST SPLIT

print("\n3. Splitting dataset...")

X = df["short_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# TF-IDF + LOGISTIC REGRESSION

print("\n4. Training TF-IDF model...")

tfidf_model = Pipeline([

    (
        "tfidf",
        TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),

    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )

])

tfidf_model.fit(X_train, y_train)

tfidf_predictions = tfidf_model.predict(X_test)


# EVALUATE TF-IDF MODEL

print("\n========== TF-IDF RESULTS ==========")

tfidf_accuracy = accuracy_score(
    y_test,
    tfidf_predictions
)

tfidf_f1 = f1_score(
    y_test,
    tfidf_predictions,
    average="macro"
)

print("Accuracy:", round(tfidf_accuracy, 4))
print("Macro F1:", round(tfidf_f1, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        tfidf_predictions
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        tfidf_predictions,
        labels=["neg", "pos"]
    )
)


# TOP 10 IMPORTANT WORDS FOR EACH CLASS

print("\n========== IMPORTANT WORDS ==========")

vectorizer = tfidf_model.named_steps["tfidf"]

classifier = tfidf_model.named_steps["classifier"]

feature_names = vectorizer.get_feature_names_out()

coefficients = classifier.coef_[0]

# Negative sentiment words
negative_indices = np.argsort(coefficients)[:10]

print("\nTop 10 Negative Sentiment Words:")

for idx in negative_indices:

    print(
        feature_names[idx],
        round(coefficients[idx], 4)
    )


# Positive sentiment words
positive_indices = np.argsort(coefficients)[-10:][::-1]

print("\nTop 10 Positive Sentiment Words:")

for idx in positive_indices:

    print(
        feature_names[idx],
        round(coefficients[idx], 4)
    )


# LOAD GLOVE EMBEDDINGS USING NUMPY

print("\n7. Loading GloVe embeddings...")

glove = {}

glove_file = "glove.6B.50d.txt"

with open(glove_file, "r", encoding="utf-8") as file:

    for line in file:

        values = line.split()

        word = values[0]

        vector = np.asarray(
            values[1:],
            dtype=np.float32
        )

        glove[word] = vector


print("GloVe loaded successfully!")

print("Vocabulary size:", len(glove))

print("Embedding dimensions:", len(glove["king"]))


# FIND 5 NEAREST WORDS

print("\n========== NEAREST WORDS ==========")

def most_similar(word, topn=5):

    if word not in glove:

        print("Word not found:", word)
        return []

    vector = glove[word]

    vector_norm = np.linalg.norm(vector)

    similarities = []

    for other_word, other_vector in glove.items():

        if other_word == word:
            continue

        similarity = np.dot(vector, other_vector) / (
            vector_norm * np.linalg.norm(other_vector)
        )

        similarities.append(
            (other_word, similarity)
        )

    similarities.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return similarities[:topn]


# First chosen word
print("\n5 nearest words to KING:")

for word, score in most_similar("king", 5):

    print(word, round(float(score), 4))


# Second chosen word
print("\n5 nearest words to COMPUTER:")

for word, score in most_similar("computer", 5):

    print(word, round(float(score), 4))


# WORD ANALOGY
# king - man + woman = queen

print("\n========== WORD ANALOGY ==========")

def word_analogy(a, b, c):

    target = glove[a] - glove[b] + glove[c]

    target_norm = np.linalg.norm(target)

    best_word = None
    best_similarity = -1

    for word, vector in glove.items():

        if word in [a, b, c]:
            continue

        similarity = np.dot(target, vector) / (
            target_norm * np.linalg.norm(vector)
        )

        if similarity > best_similarity:

            best_similarity = similarity
            best_word = word

    return best_word


analogy_result = word_analogy(
    "king",
    "man",
    "woman"
)

print(
    "king - man + woman ->",
    analogy_result
)


# CREATE SENTENCE VECTORS

print("\n10. Creating sentence vectors...")

def sentence_vector(sentence):

    words = sentence.split()

    vectors = []

    for word in words:

        if word in glove:

            vectors.append(glove[word])

    # Handle words not found in GloVe
    if len(vectors) == 0:

        return np.zeros(50, dtype=np.float32)

    # Average all word vectors
    return np.mean(vectors, axis=0)


X_train_glove = np.array([

    sentence_vector(text)
    for text in X_train

])

X_test_glove = np.array([

    sentence_vector(text)
    for text in X_test

])

print("Training vector shape:", X_train_glove.shape)
print("Testing vector shape:", X_test_glove.shape)


# LOGISTIC REGRESSION USING GLOVE

print("\n11. Training GloVe classifier...")

glove_classifier = LogisticRegression(
    max_iter=1000
)

glove_classifier.fit(
    X_train_glove,
    y_train
)

glove_predictions = glove_classifier.predict(
    X_test_glove
)


# EVALUATE GLOVE MODEL

glove_accuracy = accuracy_score(
    y_test,
    glove_predictions
)

glove_f1 = f1_score(
    y_test,
    glove_predictions,
    average="macro"
)

print("\n========== GLOVE RESULTS ==========")

print("Accuracy:", round(glove_accuracy, 4))

print("Macro F1:", round(glove_f1, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        glove_predictions
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        glove_predictions,
        labels=["neg", "pos"]
    )
)


# COMPARE BOTH MODELS

print("\n========== MODEL COMPARISON ==========")

comparison = pd.DataFrame({

    "Model": [
        "TF-IDF + Logistic Regression",
        "GloVe + Logistic Regression"
    ],

    "Accuracy": [
        tfidf_accuracy,
        glove_accuracy
    ],

    "Macro F1": [
        tfidf_f1,
        glove_f1
    ]

})

print(comparison.to_string(index=False))


# FINAL OUTPUT

print("\n========== FINAL SUMMARY ==========")

print("Dataset size:", len(df))

print(
    "TF-IDF Accuracy:",
    round(tfidf_accuracy, 4)
)

print(
    "TF-IDF Macro F1:",
    round(tfidf_f1, 4)
)

print(
    "GloVe Accuracy:",
    round(glove_accuracy, 4)
)

print(
    "GloVe Macro F1:",
    round(glove_f1, 4)
)

print(
    "Word analogy:",
    "king - man + woman ->",
    analogy_result
)
