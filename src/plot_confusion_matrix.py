from pathlib import Path

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Project root folder:
# Task2_Spam_Classifier
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Test dataset path
TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_dataset.csv"
)

# Trained Logistic Regression model
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "spam_classifier.joblib"
)

# Saved TF-IDF vectorizer
VECTORIZER_PATH = (
    PROJECT_ROOT
    / "models"
    / "tfidf_vectorizer.joblib"
)

# Results folder
RESULTS_DIR = PROJECT_ROOT / "results"

# Final confusion matrix image
OUTPUT_PATH = RESULTS_DIR / "confusion_matrix.png"


# ============================================================
# 2. LOAD TEST DATASET
# ============================================================

test_df = pd.read_csv(TEST_DATA_PATH)

print("Test dataset loaded successfully!")
print(f"Test messages: {len(test_df)}")


# ============================================================
# 3. SEPARATE INPUT AND ACTUAL LABELS
# ============================================================

# Messages are the input to the model
X_test = test_df["message"]

# Actual labels:
# 0 = Ham
# 1 = Spam
y_test = test_df["label_numeric"]


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

print("Trained model loaded successfully!")


# ============================================================
# 5. LOAD TF-IDF VECTORIZER
# ============================================================

vectorizer = joblib.load(VECTORIZER_PATH)

print("TF-IDF vectorizer loaded successfully!")


# ============================================================
# 6. CONVERT TEST MESSAGES INTO TF-IDF FEATURES
# ============================================================

# The model cannot directly understand text.
# Therefore, we convert the test messages into
# numerical TF-IDF features.

X_test_tfidf = vectorizer.transform(X_test)

print("Test messages transformed using TF-IDF!")
print(f"TF-IDF shape: {X_test_tfidf.shape}")


# ============================================================
# 7. GENERATE MODEL PREDICTIONS
# ============================================================

# Model predicts:
# 0 = Ham
# 1 = Spam

y_pred = model.predict(X_test_tfidf)

print("Predictions generated successfully!")


# ============================================================
# 8. CALCULATE CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 9. CREATE RESULTS DIRECTORY
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 10. CREATE CONFUSION MATRIX HEATMAP
# ============================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"],
    cbar=True
)


# ============================================================
# 11. ADD CHART LABELS
# ============================================================

plt.xlabel(
    "Predicted Label",
    fontsize=12
)

plt.ylabel(
    "Actual Label",
    fontsize=12
)

plt.title(
    "Spam SMS Classifier - Confusion Matrix",
    fontsize=14,
    fontweight="bold"
)


# ============================================================
# 12. SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 13. CLOSE PLOT
# ============================================================

plt.close()


# ============================================================
# 14. FINAL MESSAGE
# ============================================================

print("\nConfusion matrix plot created successfully!")

print("\nSaved to:")
print(OUTPUT_PATH)