# Spam SMS Classifier

A machine learning project that classifies SMS messages as **Ham** (normal message) or **Spam** (unwanted/promotional message).

This project was developed as part of the **AVIP 2026 AI/ML Engineering Internship — Task 2**.

---

## 📌 Project Overview

Spam messages are unwanted messages that may contain advertisements, fraudulent offers, prize claims, or other promotional content.

The objective of this project is to build a machine learning model that can automatically classify an SMS message into one of two categories:

- **Ham** — Normal/non-spam message
- **Spam** — Spam/unwanted message

The project uses:

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Matplotlib
- Seaborn
- Joblib
- Streamlit

---

## 🎯 Objective

The main objectives of this project are:

1. Load and validate the SMS Spam Collection dataset.
2. Clean and preprocess the SMS messages.
3. Convert text into numerical features using TF-IDF.
4. Train a Logistic Regression classification model.
5. Evaluate the model using:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - Confusion Matrix
6. Generate sample predictions with confidence probabilities.
7. Save the trained model and TF-IDF vectorizer.
8. Provide a Streamlit web application for real-time predictions.

---

# 📊 Dataset

## Dataset Name

**SMS Spam Collection**

The dataset contains SMS messages labeled as either:

- `ham`
- `spam`

### Dataset Source

The dataset was obtained from the **UCI Machine Learning Repository**.

Source:

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

---

## Dataset Statistics

### Original Dataset

| Category | Number of Messages |
|---|---:|
| Ham | 4,827 |
| Spam | 747 |
| **Total** | **5,574** |

The raw dataset contains **5,574 SMS messages**.

---

# 🧹 Data Preprocessing

The raw SMS dataset was cleaned before training the machine learning model.

The preprocessing pipeline includes:

1. Removing missing values.
2. Removing duplicate messages.
3. Converting text to lowercase.
4. Removing URLs.
5. Removing email addresses.
6. Removing special characters and punctuation.
7. Removing extra spaces.
8. Converting class labels into numerical values.

The labels were encoded as:

```text
ham  → 0
spam → 1

Preprocessing Results
Stage	Messages
Raw dataset	5,574
Duplicate rows removed	403
Empty messages removed after cleaning	2
Final cleaned dataset	5,169

Final class distribution:

Category	Messages
Ham	4,516
Spam	653
Total	5,169
🔤 Feature Extraction — TF-IDF

Machine learning models cannot directly understand raw text.

Therefore, the cleaned SMS messages were converted into numerical features using:

TF-IDF — Term Frequency-Inverse Document Frequency

The vectorizer was configured with:

English stop-word removal
Unigrams and bigrams
Minimum document frequency = 2
Maximum document frequency = 0.95
Sublinear TF scaling

Configuration:

TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

The resulting feature matrix contained:

5,634 features
🤖 Machine Learning Model
Logistic Regression

The classification model used in this project is:

Logistic Regression

Configuration:

LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
Why Logistic Regression?

Logistic Regression is well suited for binary text classification problems.

In this project:

Input SMS
    ↓
Text Cleaning
    ↓
TF-IDF
    ↓
Logistic Regression
    ↓
Ham / Spam

The model also provides class probabilities, which are used to display prediction confidence.

📚 Train/Test Split

The cleaned dataset was divided into training and testing sets.

The split used:

Training data = 80%
Testing data  = 20%

A fixed random state was used:

random_state = 42

Stratified splitting was used to preserve the class distribution.

Dataset Split
Dataset	Messages
Training	4,135
Testing	1,034
Total	5,169
📈 Model Evaluation

The trained model was evaluated on the unseen test dataset containing:

1,034 messages
Evaluation Metrics
Metric	Score
Accuracy	97.20%
Precision	88.06%
Recall	90.08%
F1 Score	89.06%
Accuracy

Accuracy measures the proportion of total predictions that were correct.

Accuracy = Correct Predictions / Total Predictions

The model achieved:

97.20%
Precision

Precision measures how many messages predicted as spam were actually spam.

Precision = True Positives / (True Positives + False Positives)

Result:

88.06%
Recall

Recall measures how many actual spam messages were successfully detected.

Recall = True Positives / (True Positives + False Negatives)

Result:

90.08%
F1 Score

F1 Score combines precision and recall into a single metric.

F1 = 2 × (Precision × Recall) / (Precision + Recall)

Result:

89.06%
🔲 Confusion Matrix

The confusion matrix obtained on the test dataset was:

[[887, 16],
 [ 13, 118]]

Rows represent the actual labels and columns represent the predicted labels.

                 Predicted
                 Ham    Spam

Actual Ham       887     16
Actual Spam       13    118
Interpretation
887 Ham messages were correctly classified as Ham.
118 Spam messages were correctly classified as Spam.
16 Ham messages were incorrectly classified as Spam.
13 Spam messages were incorrectly classified as Ham.

The graphical confusion matrix is saved at:

results/confusion_matrix.png
🧪 Sample Predictions

The project includes 10 example SMS messages to demonstrate model inference.

The predictions include:

Predicted label
Prediction confidence
Spam probability
Ham probability

The results are saved in:

sample_predictions/sample_predictions.csv
Example Results
Sample	Predicted Label	Confidence
1	Ham	92.78%
2	Spam	96.41%
3	Ham	83.78%
4	Spam	99.43%
5	Ham	74.64%
6	Spam	91.47%
7	Ham	90.51%
8	Spam	95.04%
9	Ham	87.02%
10	Spam	91.04%
💾 Saved Model Files

The trained machine learning artifacts are stored in the models directory.

models/
├── spam_classifier.joblib
└── tfidf_vectorizer.joblib
spam_classifier.joblib

Contains the trained Logistic Regression model.

tfidf_vectorizer.joblib

Contains the fitted TF-IDF vectorizer required to transform new SMS messages into the same numerical feature representation used during training.

Both files are required for inference.

🌐 Streamlit Web Application

A Streamlit application is included for interactive SMS classification.

The application allows a user to:

Enter an SMS message.
Convert the message into TF-IDF features.
Generate a prediction.
Display Ham/Spam classification.
Display prediction confidence.
Display Ham probability.
Display Spam probability.


# **🚀 Live Demo**

The deployed Streamlit application is available here:

👉 **[Open Spam SMS Classifier Live Demo](https://task2-spam-sms-classifier.streamlit.app/)**

The live application allows users to:

- Enter an SMS message
- Get a Ham/Spam prediction
- View prediction confidence
- View Ham probability
- View Spam probability



▶️ Run the Project Locally
1. Clone the Repository
git clone <YOUR_GITHUB_REPOSITORY_URL>

Move into the project directory:

cd Task2_Spam_Classifier
2. Create/Activate the Conda Environment
conda activate avip-ml

The project was developed using Python 3.12.

Check Python version:

python --version
3. Install Dependencies

Run:

python -m pip install -r requirements.txt
🔍 Run Model Evaluation

To evaluate the trained model:

python .\src\evaluate_model.py

This generates:

results/evaluation_metrics.txt
results/confusion_matrix.txt
📊 Generate Confusion Matrix Plot

Run:

python .\src\plot_confusion_matrix.py

The generated image will be saved as:

results/confusion_matrix.png
🧪 Generate Sample Predictions

Run:

python .\src\predict_samples.py

The 10 sample predictions will be saved as:

sample_predictions/sample_predictions.csv
🌐 Run Streamlit Application

Run:

streamlit run app.py

Streamlit will start a local web server.

Open the displayed local URL in your browser.

📁 Project Structure
Task2_Spam_Classifier/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── SMSSpamCollection
│   ├── readme
│   │
│   └── processed/
│       ├── spam_dataset_cleaned.csv
│       └── test_dataset.csv
│
├── models/
│   ├── spam_classifier.joblib
│   └── tfidf_vectorizer.joblib
│
├── results/
│   ├── evaluation_metrics.txt
│   ├── confusion_matrix.txt
│   └── confusion_matrix.png
│
├── sample_predictions/
│   └── sample_predictions.csv
│
└── src/
    ├── load_dataset.py
    ├── preprocess_dataset.py
    ├── train_model.py
    ├── evaluate_model.py
    ├── predict_samples.py
    └── plot_confusion_matrix.py
🛠️ Engineering Challenges

During development, several practical issues were handled:

1. Dataset Parsing

The SMS dataset contains tab-separated labels and messages.

A manual line-by-line parser using:

line.split("\t", maxsplit=1)

was used to preserve the complete message text and correctly load all 5,574 raw records.

2. Duplicate Data

Duplicate messages were removed during preprocessing to reduce repeated examples and help prevent data leakage between similar records.

3. Text-to-Numeric Conversion

Since machine learning models require numerical input, TF-IDF was used to transform SMS text into numerical feature vectors.

4. Class Imbalance

The dataset contains considerably more Ham messages than Spam messages.

Therefore, the Logistic Regression model uses:

class_weight="balanced"

to give additional importance to the minority class during training.

5. Model Persistence

The trained model and TF-IDF vectorizer were serialized using Joblib so they can be reused for predictions without retraining.

🔄 Complete Machine Learning Pipeline

The complete workflow is:

SMS Spam Collection Dataset
            ↓
       Data Loading
            ↓
      Data Validation
            ↓
       Text Cleaning
            ↓
    Duplicate Removal
            ↓
       Label Encoding
            ↓
      Train/Test Split
            ↓
       TF-IDF Vectorizer
            ↓
    Logistic Regression
            ↓
        Prediction
            ↓
     Model Evaluation
            ↓
 Confusion Matrix + Metrics
            ↓
       Model Saving
            ↓
     Streamlit Deployment
📦 Technologies Used
Technology	Purpose
Python	Programming language
Pandas	Data loading and manipulation
NumPy	Numerical operations
Scikit-learn	Machine learning and evaluation
TF-IDF	Text feature extraction
Logistic Regression	Spam classification
Matplotlib	Visualization
Seaborn	Confusion matrix visualization
Joblib	Model serialization
Streamlit	Web application
Git	Version control
GitHub	Source code hosting
📝 Task Summary

This project implements an SMS spam classification system using the SMS Spam Collection dataset. The raw dataset contains 5,574 messages labeled as Ham or Spam. The data was cleaned by removing missing values, duplicates, URLs, email addresses, special characters, and extra spaces. After preprocessing, 5,169 messages remained.

The cleaned messages were converted into numerical features using TF-IDF with unigram and bigram features. The data was divided into 80% training and 20% testing sets using stratified sampling. A Logistic Regression classifier with balanced class weights was trained on the TF-IDF features.

On the test set of 1,034 messages, the model achieved 97.20% accuracy, 88.06% precision, 90.08% recall, and an F1 score of 89.06%. The confusion matrix was [[887, 16], [13, 118]].

The trained model and TF-IDF vectorizer were serialized using Joblib. Ten sample messages were also tested and their predicted labels and probabilities were saved. A Streamlit web application provides an interactive interface for classifying new SMS messages.

⚠️ Limitations

This project has some limitations:

The model was trained specifically on the SMS Spam Collection dataset.
New types of spam messages may not always be classified correctly.
Prediction quality depends on the vocabulary and patterns represented in the training data.
A model confidence/probability is not a guarantee that a message is actually spam.
The system is intended as a machine learning demonstration and should not be treated as a complete anti-spam security system.
👩‍💻 Internship

Program: AVIP 2026 AI/ML Engineering Internship

Task: Task 2 — Spam SMS/Email Classifier

Domain: AI/ML Engineering

📌 Author

Vaishnavi


### Important

README me abhi ye line intentionally placeholder hai:

```text
git clone <YOUR_GITHUB_REPOSITORY_URL>