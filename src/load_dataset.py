from pathlib import Path
import pandas as pd


# Project ke main folder ka path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset ka path
DATASET_PATH = PROJECT_ROOT / "data" / "SMSSpamCollection"


# Dataset ko manually tab-separated format mein load karo
rows = []

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    for line in file:
        line = line.rstrip("\n\r")

        if not line:
            continue

        label, message = line.split("\t", maxsplit=1)
        rows.append((label, message))


# DataFrame create karo
df = pd.DataFrame(rows, columns=["label", "message"])


# Basic information display karo
print("Dataset loaded successfully!")
print(f"Total messages: {len(df)}")

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nFirst 5 rows:")
print(df.head())