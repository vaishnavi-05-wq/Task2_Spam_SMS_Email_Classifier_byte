from pathlib import Path

import joblib
import streamlit as st


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

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


# ============================================================
# 2. LOAD MODEL AND VECTORIZER
# ============================================================

@st.cache_resource
def load_model_and_vectorizer():
    """
    Load the trained Logistic Regression model
    and TF-IDF vectorizer.

    @st.cache_resource makes sure Streamlit does not
    reload these files every time the user interacts
    with the application.
    """

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


model, vectorizer = load_model_and_vectorizer()


# ============================================================
# 3. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Spam SMS Classifier",
    page_icon="📩",
    layout="centered"
)


# ============================================================
# 4. APPLICATION TITLE
# ============================================================

st.title("📩 Spam SMS Classifier")

st.write(
    "Enter an SMS message below and the trained machine "
    "learning model will predict whether it is Ham or Spam."
)


# ============================================================
# 5. INFORMATION ABOUT THE MODEL
# ============================================================

with st.expander("ℹ️ About this model"):

    st.write(
        """
        This application uses:

        - TF-IDF for converting text into numerical features
        - Logistic Regression for classification
        - Ham = normal/non-spam message
        - Spam = unwanted or promotional message
        """
    )


# ============================================================
# 6. MESSAGE INPUT
# ============================================================

message = st.text_area(
    "Enter your SMS message:",
    placeholder="Example: Congratulations! You have won a free prize!",
    height=150
)


# ============================================================
# 7. PREDICTION BUTTON
# ============================================================

if st.button(
    "🔍 Check Message",
    use_container_width=True
):

    # --------------------------------------------------------
    # Check whether the user entered a message
    # --------------------------------------------------------

    if not message.strip():

        st.warning(
            "Please enter an SMS message first."
        )

    else:

        # ----------------------------------------------------
        # Convert message into TF-IDF features
        # ----------------------------------------------------

        message_tfidf = vectorizer.transform(
            [message]
        )


        # ----------------------------------------------------
        # Generate prediction
        # ----------------------------------------------------

        prediction = model.predict(
            message_tfidf
        )[0]


        # ----------------------------------------------------
        # Generate probabilities
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            message_tfidf
        )[0]


        # Probability of Ham
        ham_probability = probabilities[0]

        # Probability of Spam
        spam_probability = probabilities[1]


        # ----------------------------------------------------
        # Determine predicted label and confidence
        # ----------------------------------------------------

        if prediction == 1:

            predicted_label = "SPAM"

            confidence = spam_probability

        else:

            predicted_label = "HAM"

            confidence = ham_probability


        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        st.subheader("Prediction")


        if predicted_label == "SPAM":

            st.error(
                f"🚨 {predicted_label}"
            )

        else:

            st.success(
                f"✅ {predicted_label}"
            )


        # ----------------------------------------------------
        # Display confidence
        # ----------------------------------------------------

        st.metric(
            "Prediction Confidence",
            f"{confidence * 100:.2f}%"
        )


        # ----------------------------------------------------
        # Display probability details
        # ----------------------------------------------------

        st.subheader("Probability Details")


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Ham Probability",
                f"{ham_probability * 100:.2f}%"
            )


        with col2:

            st.metric(
                "Spam Probability",
                f"{spam_probability * 100:.2f}%"
            )


        # ----------------------------------------------------
        # Probability bar
        # ----------------------------------------------------

        st.write("Spam probability:")

        st.progress(
            float(spam_probability)
        )


# ============================================================
# 8. FOOTER
# ============================================================

st.divider()

st.caption(
    "AVIP 2026 AI/ML Engineering Internship — Task 2"
)