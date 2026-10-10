import os
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

# ==============================================================
# CONFIGURATION
# ==============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(
    BASE_DIR, "dataset", "processed", "train.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR, "dataset", "processed", "test.csv"
)

MODEL_NAME = "bert-base-multilingual-cased"

RESULTS_DIR = os.path.join(
    BASE_DIR, "results", "metrics"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

# Training sizes for the low-resource experiment
TRAINING_SIZES = [100, 300, 500, 806]

RANDOM_SEED = 42
MAX_LENGTH = 128

# ==============================================================
# REPRODUCIBILITY
# ==============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ==============================================================
# HEADER
# ==============================================================

print("=" * 70)
print("LOW-RESOURCE mBERT EXPERIMENT")
print("=" * 70)

# ==============================================================
# LOAD DATA
# ==============================================================

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"\nFull training dataset : {len(train_df)}")
print(f"Fixed test dataset    : {len(test_df)}")

print("\nTraining sizes:")
print(TRAINING_SIZES)

# ==============================================================
# LABEL MAPPING
# ==============================================================

label2id = {
    "Negative": 0,
    "Positive": 1
}

id2label = {
    0: "Negative",
    1: "Positive"
}

train_df["label"] = train_df["Sentiment"].map(label2id)
test_df["label"] = test_df["Sentiment"].map(label2id)

# ==============================================================
# LOAD TOKENIZER
# ==============================================================

print("\n" + "=" * 70)
print("LOADING mBERT TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded successfully.")

# ==============================================================
# TOKENIZATION FUNCTION
# ==============================================================

def tokenize_function(examples):
    return tokenizer(
        examples["Hindi_Text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

# ==============================================================
# METRICS
# ==============================================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)

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

# ==============================================================
# PREPARE FIXED TEST DATA
# ==============================================================

test_small = test_df[
    ["Hindi_Text", "label"]
].copy()

test_dataset = Dataset.from_pandas(
    test_small,
    preserve_index=False
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)

# ==============================================================
# RUN EXPERIMENTS
# ==============================================================

results = []

for train_size in TRAINING_SIZES:

    print("\n")
    print("=" * 70)
    print(f"TRAINING WITH {train_size} SAMPLES")
    print("=" * 70)

    # ----------------------------------------------------------
    # Select training subset
    # ----------------------------------------------------------

    if train_size == len(train_df):

        subset_df = train_df.copy()

    else:

        # Stratified sampling to preserve Positive/Negative balance
        positive = train_df[
            train_df["Sentiment"] == "Positive"
        ]

        negative = train_df[
            train_df["Sentiment"] == "Negative"
        ]

        positive_count = train_size // 2
        negative_count = train_size - positive_count

        positive_sample = positive.sample(
            n=positive_count,
            random_state=RANDOM_SEED
        )

        negative_sample = negative.sample(
            n=negative_count,
            random_state=RANDOM_SEED
        )

        subset_df = pd.concat(
            [positive_sample, negative_sample]
        ).sample(
            frac=1,
            random_state=RANDOM_SEED
        ).reset_index(drop=True)

    print(f"Selected training samples: {len(subset_df)}")

    print("\nClass distribution:")

    print(
        subset_df["Sentiment"].value_counts()
    )

    # ----------------------------------------------------------
    # Convert training data to Hugging Face Dataset
    # ----------------------------------------------------------

    train_small = subset_df[
        ["Hindi_Text", "label"]
    ].copy()

    train_dataset = Dataset.from_pandas(
        train_small,
        preserve_index=False
    )

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True
    )

    # ----------------------------------------------------------
    # Load fresh mBERT model
    # ----------------------------------------------------------

    print("\nLoading fresh mBERT model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )

    # ----------------------------------------------------------
    # Training arguments
    # ----------------------------------------------------------

    output_dir = os.path.join(
        BASE_DIR,
        "results",
        f"low_resource_{train_size}"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,

        eval_strategy="no",

        save_strategy="no",

        learning_rate=2e-5,

        per_device_train_batch_size=8,

        per_device_eval_batch_size=8,

        num_train_epochs=3,

        weight_decay=0.01,

        logging_steps=10,

        report_to="none"
    )

    # ----------------------------------------------------------
    # Trainer
    # ----------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # ----------------------------------------------------------
    # Train
    # ----------------------------------------------------------

    print("\nStarting training...")

    trainer.train()

    # ----------------------------------------------------------
    # Evaluate on SAME test set
    # ----------------------------------------------------------

    print("\nEvaluating on fixed test set...")

    evaluation = trainer.evaluate(
        test_dataset
    )

    accuracy = evaluation["eval_accuracy"]
    precision = evaluation["eval_precision"]
    recall = evaluation["eval_recall"]
    f1 = evaluation["eval_f1"]

    print("\nResults:")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    # ----------------------------------------------------------
    # Store results
    # ----------------------------------------------------------

    results.append({
        "Training_Samples": train_size,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    # ----------------------------------------------------------
    # Delete model before next experiment
    # ----------------------------------------------------------

    del trainer
    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==============================================================
# SAVE RESULTS
# ==============================================================

results_df = pd.DataFrame(results)

output_file = os.path.join(
    RESULTS_DIR,
    "low_resource_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

# ==============================================================
# DISPLAY FINAL TABLE
# ==============================================================

print("\n")
print("=" * 70)
print("LOW-RESOURCE EXPERIMENT RESULTS")
print("=" * 70)

print(
    results_df.to_string(index=False)
)

print("\nResults saved to:")
print(output_file)

print("\n" + "=" * 70)
print("LOW-RESOURCE EXPERIMENT COMPLETED")
print("=" * 70)