import os
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

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ==============================================================
# CONFIGURATION
# ==============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "train.csv"
)

VAL_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "validation.csv"
)

TEST_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "test.csv"
)

MODEL_NAME = "bert-base-multilingual-cased"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "mbert_lora_model"
)

FIGURE_DIR = os.path.join(
    BASE_DIR,
    "results",
    "figures"
)

METRIC_DIR = os.path.join(
    BASE_DIR,
    "results",
    "metrics"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)


# ==============================================================
# HEADER
# ==============================================================

print("=" * 70)
print("mBERT + LoRA HINDI SENTIMENT CLASSIFICATION")
print("=" * 70)


# ==============================================================
# LOAD DATA
# ==============================================================

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

print()
print(f"Training samples   : {len(train_df)}")
print(f"Validation samples : {len(val_df)}")
print(f"Test samples       : {len(test_df)}")


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
val_df["label"] = val_df["Sentiment"].map(label2id)
test_df["label"] = test_df["Sentiment"].map(label2id)


# ==============================================================
# LOAD TOKENIZER
# ==============================================================

print()
print("=" * 70)
print("LOADING mBERT TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded successfully.")


# ==============================================================
# CONVERT DATA TO HUGGING FACE DATASETS
# ==============================================================

train_dataset = Dataset.from_pandas(
    train_df[["Hindi_Text", "label"]],
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df[["Hindi_Text", "label"]],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df[["Hindi_Text", "label"]],
    preserve_index=False
)


# ==============================================================
# TOKENIZATION
# ==============================================================

def tokenize_function(examples):

    return tokenizer(
        examples["Hindi_Text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# Remove original Hindi text
train_dataset = train_dataset.remove_columns(["Hindi_Text"])
val_dataset = val_dataset.remove_columns(["Hindi_Text"])
test_dataset = test_dataset.remove_columns(["Hindi_Text"])


# ==============================================================
# LOAD PRETRAINED mBERT
# ==============================================================

print()
print("=" * 70)
print("LOADING PRETRAINED mBERT")
print("=" * 70)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

print("mBERT loaded successfully.")


# ==============================================================
# LoRA CONFIGURATION
# ==============================================================

print()
print("=" * 70)
print("CONFIGURING LoRA / PEFT")
print("=" * 70)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,

    # LoRA rank
    r=8,

    # LoRA scaling
    lora_alpha=16,

    # Dropout for LoRA layers
    lora_dropout=0.1,

    # Apply LoRA to BERT attention projections
    target_modules=[
        "query",
        "value"
    ],

    # Keep the classification head trainable
    modules_to_save=[
        "classifier"
    ]
)


# ==============================================================
# APPLY LoRA
# ==============================================================

model = get_peft_model(
    model,
    lora_config
)

print()
print("LoRA applied successfully.")

print()
print("TRAINABLE PARAMETER SUMMARY")
print("-" * 70)

model.print_trainable_parameters()


# ==============================================================
# COMPUTE METRICS
# ==============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

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


# ==============================================================
# TRAINING ARGUMENTS
# ==============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-4,

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


# ==============================================================
# TRAINER
# ==============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    processing_class=tokenizer,

    compute_metrics=compute_metrics
)


# ==============================================================
# FINE-TUNING
# ==============================================================

print()
print("=" * 70)
print("STARTING mBERT + LoRA FINE-TUNING")
print("=" * 70)

trainer.train()


# ==============================================================
# TEST EVALUATION
# ==============================================================

print()
print("=" * 70)
print("EVALUATING mBERT + LoRA ON TEST SET")
print("=" * 70)

test_results = trainer.evaluate(
    test_dataset
)

print()
print("TEST RESULTS")
print("-" * 70)

for key, value in test_results.items():

    if isinstance(value, float):

        print(
            f"{key:<15}: {value:.4f}"
        )

    else:

        print(
            f"{key:<15}: {value}"
        )


# ==============================================================
# PREDICTIONS
# ==============================================================

predictions_output = trainer.predict(
    test_dataset
)

predicted_labels = np.argmax(
    predictions_output.predictions,
    axis=-1
)

true_labels = predictions_output.label_ids


# ==============================================================
# CLASSIFICATION REPORT
# ==============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=[
        "Negative",
        "Positive"
    ],
    digits=4
)

print(report)


# ==============================================================
# CONFUSION MATRIX
# ==============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "mBERT + LoRA - Hindi Sentiment Classification"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["Negative", "Positive"]
)

plt.yticks(
    [0, 1],
    ["Negative", "Positive"]
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=14
        )

plt.tight_layout()

lora_cm_path = os.path.join(
    FIGURE_DIR,
    "mbert_lora_confusion_matrix.png"
)

plt.savefig(
    lora_cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print(
    f"Confusion matrix saved to:\n{lora_cm_path}"
)


# ==============================================================
# PARAMETER ANALYSIS
# ==============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

trainable_percentage = (
    trainable_parameters /
    total_parameters
) * 100


# ==============================================================
# SAVE METRICS
# ==============================================================

metrics_path = os.path.join(
    METRIC_DIR,
    "mbert_lora_metrics.txt"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "mBERT + LoRA Hindi Sentiment Classification\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Model: {MODEL_NAME}\n"
    )

    f.write(
        "PEFT Method: LoRA\n"
    )

    f.write(
        "LoRA rank (r): 8\n"
    )

    f.write(
        "LoRA alpha: 16\n"
    )

    f.write(
        "LoRA dropout: 0.1\n\n"
    )

    f.write(
        f"Training samples: {len(train_df)}\n"
    )

    f.write(
        f"Validation samples: {len(val_df)}\n"
    )

    f.write(
        f"Test samples: {len(test_df)}\n\n"
    )

    f.write(
        f"Accuracy: {test_results['eval_accuracy']:.4f}\n"
    )

    f.write(
        f"Precision: {test_results['eval_precision']:.4f}\n"
    )

    f.write(
        f"Recall: {test_results['eval_recall']:.4f}\n"
    )

    f.write(
        f"F1-score: {test_results['eval_f1']:.4f}\n\n"
    )

    f.write(
        f"Total parameters: {total_parameters:,}\n"
    )

    f.write(
        f"Trainable parameters: {trainable_parameters:,}\n"
    )

    f.write(
        f"Trainable percentage: {trainable_percentage:.4f}%\n\n"
    )

    f.write(
        "Classification Report\n"
    )

    f.write(
        "-" * 60 + "\n"
    )

    f.write(report)


# ==============================================================
# SAVE MODEL
# ==============================================================

FINAL_MODEL_DIR = os.path.join(
    BASE_DIR,
    "results",
    "mbert_lora_final_model"
)

model.save_pretrained(
    FINAL_MODEL_DIR
)

tokenizer.save_pretrained(
    FINAL_MODEL_DIR
)


# ==============================================================
# FINAL SUMMARY
# ==============================================================

print()
print("=" * 70)
print("mBERT + LoRA EXPERIMENT COMPLETED")
print("=" * 70)

print(
    f"Accuracy            : {test_results['eval_accuracy']:.4f}"
)

print(
    f"Precision           : {test_results['eval_precision']:.4f}"
)

print(
    f"Recall              : {test_results['eval_recall']:.4f}"
)

print(
    f"F1-score            : {test_results['eval_f1']:.4f}"
)

print(
    f"Total parameters    : {total_parameters:,}"
)

print(
    f"Trainable parameters: {trainable_parameters:,}"
)

print(
    f"Trainable percentage: {trainable_percentage:.4f}%"
)

print()
print(
    f"Metrics saved to:\n{metrics_path}"
)

print()
print(
    f"Model saved to:\n{FINAL_MODEL_DIR}"
)