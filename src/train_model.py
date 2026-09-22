from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spam_dataset_cleaned.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "spam_classifier.joblib"

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_dataset.csv"
)


# ============================================================
# 2. LOAD CLEANED DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("Cleaned dataset loaded successfully!")
print(f"Total messages: {len(df)}")


# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================

X = df["clean_message"]

y = df["label_numeric"]


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nDataset split:")
print(f"Training messages: {len(X_train)}")
print(f"Testing messages:  {len(X_test)}")


# ============================================================
# 5. TF-IDF VECTORIZATION
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)


# IMPORTANT:
# TF-IDF vocabulary is learned only from training data.
X_train_tfidf = vectorizer.fit_transform(X_train)

# Test data is transformed using the same vocabulary.
X_test_tfidf = vectorizer.transform(X_test)


print("\nTF-IDF vectorization completed!")

print(f"Training TF-IDF shape: {X_train_tfidf.shape}")
print(f"Testing TF-IDF shape:  {X_test_tfidf.shape}")


# ============================================================
# 6. TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)


model.fit(
    X_train_tfidf,
    y_train
)


print("\nLogistic Regression model trained successfully!")


# ============================================================
# 7. CREATE MODEL DIRECTORY
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 8. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)


# ============================================================
# 9. SAVE TF-IDF VECTORIZER
# ============================================================

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)


# ============================================================
# 10. SAVE TEST DATA
# ============================================================

test_df = pd.DataFrame(
    {
        "message": X_test,
        "label_numeric": y_test
    }
)

test_df.to_csv(
    TEST_DATA_PATH,
    index=False,
    encoding="utf-8"
)


# ============================================================
# 11. FINAL INFORMATION
# ============================================================

print("\nTraining completed successfully!")

print(f"Model saved to:")
print(MODEL_PATH)

print(f"\nTF-IDF vectorizer saved to:")
print(VECTORIZER_PATH)

print(f"\nTest dataset saved to:")
print(TEST_DATA_PATH)