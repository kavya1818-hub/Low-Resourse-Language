import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

TEST_PATH = PROJECT_DIR / "dataset" / "indic_sentiment_hi_test.json"
VALIDATION_PATH = PROJECT_DIR / "dataset" / "indic_sentiment_hi_validation.json"

OUTPUT_DIR = PROJECT_DIR / "dataset" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD JSONL FILE
# ============================================================

def load_jsonl(path):

    records = []

    with open(path, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if line:
                records.append(json.loads(line))

    return pd.DataFrame(records)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING AI4BHARAT HINDI SENTIMENT DATA")
print("=" * 70)

test_df = load_jsonl(TEST_PATH)
validation_df = load_jsonl(VALIDATION_PATH)

print(f"Test records       : {len(test_df)}")
print(f"Validation records : {len(validation_df)}")


# ============================================================
# 4. COMBINE DATA
# ============================================================

df = pd.concat(
    [test_df, validation_df],
    ignore_index=True
)

print(f"Combined records   : {len(df)}")


# ============================================================
# 5. SELECT REQUIRED COLUMNS
# ============================================================

df = df[
    [
        "ENGLISH REVIEW",
        "INDIC REVIEW",
        "LABEL"
    ]
].copy()


# ============================================================
# 6. RENAME COLUMNS
# ============================================================

df.rename(
    columns={
        "ENGLISH REVIEW": "English_Text",
        "INDIC REVIEW": "Hindi_Text",
        "LABEL": "Sentiment"
    },
    inplace=True
)


# ============================================================
# 7. REMOVE MISSING LABELS
# ============================================================

before = len(df)

df.dropna(
    subset=["Hindi_Text", "English_Text", "Sentiment"],
    inplace=True
)

after = len(df)

print(f"\nRemoved incomplete rows : {before - after}")
print(f"Usable records          : {after}")


# ============================================================
# 8. REMOVE DUPLICATES
# ============================================================

before = len(df)

df.drop_duplicates(
    subset=["Hindi_Text", "English_Text", "Sentiment"],
    inplace=True
)

after = len(df)

print(f"Removed duplicate rows : {before - after}")
print(f"Final records          : {after}")


# ============================================================
# 9. CHECK LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("FINAL SENTIMENT DISTRIBUTION")
print("=" * 70)

print(df["Sentiment"].value_counts())


# ============================================================
# 10. FIRST TRAIN/TEST SPLIT
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["Sentiment"]
)


# ============================================================
# 11. VALIDATION / TEST SPLIT
# ============================================================

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["Sentiment"]
)


# ============================================================
# 12. RESET INDEX
# ============================================================

train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# 13. DISPLAY SPLIT SIZES
# ============================================================

print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(f"Training samples    : {len(train_df)}")
print(f"Validation samples  : {len(validation_df)}")
print(f"Test samples        : {len(test_df)}")


# ============================================================
# 14. SAVE PROCESSED DATA
# ============================================================

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False,
    encoding="utf-8-sig"
)

validation_df.to_csv(
    OUTPUT_DIR / "validation.csv",
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. DISPLAY SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE PROCESSED DATA")
print("=" * 70)

for i in range(min(3, len(train_df))):

    print(f"\nExample {i + 1}")
    print("-" * 60)

    print("Hindi:")
    print(train_df.iloc[i]["Hindi_Text"])

    print("\nEnglish:")
    print(train_df.iloc[i]["English_Text"])

    print("\nSentiment:")
    print(train_df.iloc[i]["Sentiment"])


print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED")
print("=" * 70)

print(f"\nProcessed files saved in:")
print(OUTPUT_DIR)