from pathlib import Path

import pandas as pd
import joblib


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "sample_predictions"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "sample_predictions.csv"
)


# ============================================================
# 2. SAMPLE SMS MESSAGES
# ============================================================

sample_messages = [
    "Hey, are we still meeting for lunch today?",
    
    "Congratulations! You have won a free prize. Call now to claim your reward.",
    
    "Can you send me the notes from today's class?",
    
    "URGENT! You have won a cash prize of 1000 pounds. Text WIN to 80000 now.",
    
    "Your appointment is confirmed for tomorrow at 10 AM.",
    
    "Free entry in a weekly competition to win a brand new car. Text WIN now!",
    
    "Please call me when you reach home.",
    
    "You have been selected for a special cash reward. Claim your prize immediately.",
    
    "Don't forget to bring your project report tomorrow.",
    
    "Congratulations! You are the lucky winner of a free vacation. Call now to claim."
]


# ============================================================
# 3. LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

print("Trained model loaded successfully!")


# ============================================================
# 4. LOAD TF-IDF VECTORIZER
# ============================================================

vectorizer = joblib.load(VECTORIZER_PATH)

print("TF-IDF vectorizer loaded successfully!")


# ============================================================
# 5. TRANSFORM SMS MESSAGES
# ============================================================

messages_tfidf = vectorizer.transform(
    sample_messages
)


# ============================================================
# 6. GENERATE PREDICTIONS
# ============================================================

predictions = model.predict(
    messages_tfidf
)


# ============================================================
# 7. GET PROBABILITIES
# ============================================================

probabilities = model.predict_proba(
    messages_tfidf
)


# ============================================================
# 8. CREATE RESULT LIST
# ============================================================

results = []


for i, (message, prediction, probability) in enumerate(
    zip(
        sample_messages,
        predictions,
        probabilities
    ),
    start=1
):

    # Probability of Spam
    spam_probability = probability[1]

    # Probability of Ham
    ham_probability = probability[0]

    # Convert numeric prediction to label
    if prediction == 1:
        predicted_label = "spam"
        confidence = spam_probability
    else:
        predicted_label = "ham"
        confidence = ham_probability

    results.append(
        {
            "sample_id": i,
            "message": message,
            "predicted_label": predicted_label,
            "confidence_percent": round(
                confidence * 100,
                2
            ),
            "spam_probability_percent": round(
                spam_probability * 100,
                2
            ),
            "ham_probability_percent": round(
                ham_probability * 100,
                2
            )
        }
    )


# ============================================================
# 9. CREATE DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n==============================")
print("SAMPLE PREDICTIONS")
print("==============================")

for _, row in results_df.iterrows():

    print(
        f"\nSample {row['sample_id']}"
    )

    print(
        f"Message: {row['message']}"
    )

    print(
        f"Prediction: {row['predicted_label']}"
    )

    print(
        f"Confidence: "
        f"{row['confidence_percent']:.2f}%"
    )

    print(
        f"Spam probability: "
        f"{row['spam_probability_percent']:.2f}%"
    )

    print(
        f"Ham probability: "
        f"{row['ham_probability_percent']:.2f}%"
    )


# ============================================================
# 11. CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print(
    f"\n10 sample predictions saved to:\n"
    f"{OUTPUT_PATH}"
)