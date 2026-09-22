from pathlib import Path
import re
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "SMSSpamCollection"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATASET_PATH = PROCESSED_DIR / "spam_dataset_cleaned.csv"


# ============================================================
# 2. LOAD RAW DATASET
# ============================================================

rows = []

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    for line in file:
        line = line.rstrip("\n\r")

        if not line:
            continue

        label, message = line.split("\t", maxsplit=1)

        rows.append((label, message))


df = pd.DataFrame(
    rows,
    columns=["label", "message"]
)


print("Raw dataset loaded successfully!")
print(f"Raw messages: {len(df)}")


# ============================================================
# 3. REMOVE MISSING VALUES
# ============================================================

before_missing = len(df)

df = df.dropna(subset=["label", "message"])

after_missing = len(df)

print(f"Rows removed because of missing values: "
      f"{before_missing - after_missing}")


# ============================================================
# 4. REMOVE DUPLICATE MESSAGES
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["label", "message"]
)

after_duplicates = len(df)

print(f"Duplicate rows removed: "
      f"{before_duplicates - after_duplicates}")


# ============================================================
# 5. CLEAN TEXT
# ============================================================

def clean_text(text):
    """
    Clean an SMS message before TF-IDF vectorization.
    """

    # Convert to string
    text = str(text)

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Keep letters and numbers
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


df["clean_message"] = df["message"].apply(clean_text)


# ============================================================
# 6. REMOVE EMPTY MESSAGES AFTER CLEANING
# ============================================================

before_empty = len(df)

df = df[df["clean_message"].str.len() > 0]

after_empty = len(df)

print(f"Empty messages removed after cleaning: "
      f"{before_empty - after_empty}")


# ============================================================
# 7. CONVERT LABELS TO NUMERIC VALUES
# ============================================================

df["label_numeric"] = df["label"].map(
    {
        "ham": 0,
        "spam": 1
    }
)


# Check whether any labels failed to convert
unknown_labels = df["label_numeric"].isna().sum()

if unknown_labels > 0:
    raise ValueError(
        f"Found {unknown_labels} unknown labels."
    )


# ============================================================
# 8. SAVE CLEANED DATASET
# ============================================================

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    PROCESSED_DATASET_PATH,
    index=False,
    encoding="utf-8"
)


# ============================================================
# 9. DISPLAY FINAL INFORMATION
# ============================================================

print("\nPreprocessing completed successfully!")

print(f"Final messages: {len(df)}")

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nNumeric class distribution:")
print(df["label_numeric"].value_counts())

print("\nSample cleaned messages:")
print(
    df[
        [
            "label",
            "message",
            "clean_message",
            "label_numeric"
        ]
    ].head()
)

print(
    f"\nCleaned dataset saved to:\n"
    f"{PROCESSED_DATASET_PATH}"
)