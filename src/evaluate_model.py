from pathlib import Path

import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_dataset.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "spam_classifier.joblib"
)

VECTORIZER_PATH = (
    PROJECT_ROOT
    / "models"
    / "tfidf_vectorizer.joblib"
)

RESULTS_DIR = PROJECT_ROOT / "results"

METRICS_PATH = (
    RESULTS_DIR
    / "evaluation_metrics.txt"
)

CONFUSION_MATRIX_PATH = (
    RESULTS_DIR
    / "confusion_matrix.txt"
)


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

test_df = pd.read_csv(TEST_DATA_PATH)

X_test = test_df["message"]

y_test = test_df["label_numeric"]

print("Test dataset loaded successfully!")
print(f"Test messages: {len(test_df)}")


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

print("\nTrained model loaded successfully!")


# ============================================================
# 4. LOAD TF-IDF VECTORIZER
# ============================================================

vectorizer = joblib.load(VECTORIZER_PATH)

print("TF-IDF vectorizer loaded successfully!")


# ============================================================
# 5. TRANSFORM TEST MESSAGES
# ============================================================

X_test_tfidf = vectorizer.transform(X_test)

print("\nTest messages transformed using TF-IDF!")
print(f"Test TF-IDF shape: {X_test_tfidf.shape}")


# ============================================================
# 6. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test_tfidf)

# Probability of spam
y_probability = model.predict_proba(X_test_tfidf)[:, 1]


print("Predictions generated successfully!")


# ============================================================
# 7. CALCULATE EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=["Ham", "Spam"],
    zero_division=0
)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"Accuracy :  {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall   :  {recall:.4f}")
print(f"F1 Score :  {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)


# ============================================================
# 11. CREATE RESULTS DIRECTORY
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 12. SAVE EVALUATION METRICS
# ============================================================

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write("Spam SMS Classifier - Evaluation Results\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Test samples: {len(test_df)}\n\n")

    file.write(f"Accuracy :  {accuracy:.4f}\n")
    file.write(f"Precision:  {precision:.4f}\n")
    file.write(f"Recall   :  {recall:.4f}\n")
    file.write(f"F1 Score :  {f1:.4f}\n\n")

    file.write("Classification Report:\n")
    file.write(report)


# ============================================================
# 13. SAVE CONFUSION MATRIX
# ============================================================

with open(
    CONFUSION_MATRIX_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Confusion Matrix\n"
    )

    file.write(
        "Rows = Actual, Columns = Predicted\n\n"
    )

    file.write(
        str(cm)
    )

    file.write("\n\n")

    file.write(
        "Matrix interpretation:\n"
    )

    file.write(
        "[[TN, FP],\n"
        " [FN, TP]]\n"
    )


# ============================================================
# 14. FINAL OUTPUT
# ============================================================

print("\nEvaluation completed successfully!")

print("\nMetrics saved to:")
print(METRICS_PATH)

print("\nConfusion matrix saved to:")
print(CONFUSION_MATRIX_PATH)