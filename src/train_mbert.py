import pandas as pd
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_DIR / "dataset" / "processed" / "train.csv"
VALIDATION_PATH = PROJECT_DIR / "dataset" / "processed" / "validation.csv"
TEST_PATH = PROJECT_DIR / "dataset" / "processed" / "test.csv"

RESULTS_DIR = PROJECT_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "bert-base-multilingual-cased"

MAX_LENGTH = 128

LABEL2ID = {
    "Negative": 0,
    "Positive": 1
}

ID2LABEL = {
    0: "Negative",
    1: "Positive"
}


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("mBERT HINDI SENTIMENT CLASSIFICATION")
print("=" * 70)

train_df = pd.read_csv(
    TRAIN_PATH,
    encoding="utf-8-sig"
)

validation_df = pd.read_csv(
    VALIDATION_PATH,
    encoding="utf-8-sig"
)

test_df = pd.read_csv(
    TEST_PATH,
    encoding="utf-8-sig"
)

print(f"\nTraining samples   : {len(train_df)}")
print(f"Validation samples : {len(validation_df)}")
print(f"Test samples       : {len(test_df)}")


# ============================================================
# 4. CONVERT LABELS TO NUMBERS
# ============================================================

train_df["label"] = train_df["Sentiment"].map(LABEL2ID)
validation_df["label"] = validation_df["Sentiment"].map(LABEL2ID)
test_df["label"] = test_df["Sentiment"].map(LABEL2ID)


# ============================================================
# 5. LOAD mBERT TOKENIZER
# ============================================================

print("\n" + "=" * 70)
print("LOADING mBERT TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded successfully.")


# ============================================================
# 6. DATASET CLASS
# ============================================================

class HindiSentimentDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_length):

        self.texts = dataframe["Hindi_Text"].astype(str).tolist()
        self.labels = dataframe["label"].tolist()

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, index):

        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        item["labels"] = torch.tensor(
            label,
            dtype=torch.long
        )

        return item


# ============================================================
# 7. CREATE DATASETS
# ============================================================

train_dataset = HindiSentimentDataset(
    train_df,
    tokenizer,
    MAX_LENGTH
)

validation_dataset = HindiSentimentDataset(
    validation_df,
    tokenizer,
    MAX_LENGTH
)

test_dataset = HindiSentimentDataset(
    test_df,
    tokenizer,
    MAX_LENGTH
)


# ============================================================
# 8. LOAD PRETRAINED mBERT
# ============================================================

print("\n" + "=" * 70)
print("LOADING PRETRAINED mBERT")
print("=" * 70)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

print("mBERT loaded successfully.")


# ============================================================
# 9. EVALUATION FUNCTION
# ============================================================

def compute_metrics(eval_prediction):

    logits, labels = eval_prediction

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# 10. TRAINING CONFIGURATION
# ============================================================

training_args = TrainingArguments(

    output_dir=str(
        PROJECT_DIR / "results" / "mbert_model"
    ),

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    num_train_epochs=3,

    weight_decay=0.01,

    logging_steps=20,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    report_to="none",

    save_total_limit=1
)


# ============================================================
# 11. TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    compute_metrics=compute_metrics
)


# ============================================================
# 12. TRAIN mBERT
# ============================================================

print("\n" + "=" * 70)
print("STARTING mBERT FINE-TUNING")
print("=" * 70)

trainer.train()

print("\nFine-tuning completed.")


# ============================================================
# 13. TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING ON TEST SET")
print("=" * 70)

test_output = trainer.predict(
    test_dataset
)

test_predictions = np.argmax(
    test_output.predictions,
    axis=-1
)

test_labels = test_output.label_ids


# ============================================================
# 14. METRICS
# ============================================================

accuracy = accuracy_score(
    test_labels,
    test_predictions
)

precision, recall, f1, _ = precision_recall_fscore_support(
    test_labels,
    test_predictions,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 70)
print("mBERT RESULTS")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    test_labels,
    test_predictions,
    target_names=[
        "Negative",
        "Positive"
    ],
    zero_division=0
)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.title("mBERT - Hindi Sentiment Classification")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

confusion_path = (
    FIGURES_DIR / "mbert_confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300
)

plt.close()


# ============================================================
# 17. SAVE METRICS
# ============================================================

metrics_path = (
    METRICS_DIR / "mbert_metrics.txt"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "mBERT Hindi Sentiment Classification\n"
    )

    file.write("=" * 50 + "\n\n")

    file.write(
        f"Model: {MODEL_NAME}\n"
    )

    file.write(
        f"Training samples: {len(train_df)}\n"
    )

    file.write(
        f"Validation samples: {len(validation_df)}\n"
    )

    file.write(
        f"Test samples: {len(test_df)}\n\n"
    )

    file.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"Recall: {recall:.4f}\n"
    )

    file.write(
        f"F1-score: {f1:.4f}\n\n"
    )

    file.write(
        "Classification Report\n"
    )

    file.write("-" * 50 + "\n")

    file.write(report)


# ============================================================
# 18. SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE mBERT PREDICTIONS")
print("=" * 70)

for i in range(min(5, len(test_df))):

    actual_label = ID2LABEL[
        test_labels[i]
    ]

    predicted_label = ID2LABEL[
        test_predictions[i]
    ]

    print(f"\nExample {i + 1}")
    print("-" * 60)

    print("Hindi:")
    print(test_df.iloc[i]["Hindi_Text"])

    print("\nEnglish:")
    print(test_df.iloc[i]["English_Text"])

    print("\nActual:")
    print(actual_label)

    print("Predicted:")
    print(predicted_label)


# ============================================================
# 19. SAVE FINAL MODEL
# ============================================================

MODEL_OUTPUT_DIR = (
    PROJECT_DIR / "results" / "mbert_final_model"
)

trainer.save_model(
    MODEL_OUTPUT_DIR
)

tokenizer.save_pretrained(
    MODEL_OUTPUT_DIR
)


# ============================================================
# 20. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("mBERT IMPLEMENTATION COMPLETED")
print("=" * 70)

print("\nMetrics saved to:")
print(metrics_path)

print("\nConfusion matrix saved to:")
print(confusion_path)

print("\nModel saved to:")
print(MODEL_OUTPUT_DIR)