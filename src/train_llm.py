import os
import re
import time
import random

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# ==============================================================
# CONFIGURATION
# ==============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "train.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "test.csv"
)

RESULTS_METRICS = os.path.join(
    BASE_DIR,
    "results",
    "metrics"
)

RESULTS_FIGURES = os.path.join(
    BASE_DIR,
    "results",
    "figures"
)

os.makedirs(RESULTS_METRICS, exist_ok=True)
os.makedirs(RESULTS_FIGURES, exist_ok=True)

# Local instruction-tuned LLM
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

RANDOM_SEED = 42

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
print("LLM HINDI SENTIMENT CLASSIFICATION")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")

# ==============================================================
# LOAD DATA
# ==============================================================

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"\nTraining samples : {len(train_df)}")
print(f"Test samples     : {len(test_df)}")

# ==============================================================
# LOAD TOKENIZER
# ==============================================================

print("\n" + "=" * 70)
print("LOADING LLM TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded successfully.")

# ==============================================================
# LOAD LLM
# ==============================================================

print("\n" + "=" * 70)
print("LOADING QWEN LLM")
print("=" * 70)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

model.eval()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

print(f"LLM loaded successfully.")
print(f"Device: {device}")

# ==============================================================
# LABEL NORMALIZATION
# ==============================================================

def normalize_prediction(text):

    text = text.strip().lower()

    # English outputs
    if re.search(r"\bpositive\b", text):
        return "Positive"

    if re.search(r"\bnegative\b", text):
        return "Negative"

    # Hindi outputs
    if "सकारात्मक" in text:
        return "Positive"

    if "नकारात्मक" in text:
        return "Negative"

    if "पॉजिटिव" in text:
        return "Positive"

    if "नेगेटिव" in text:
        return "Negative"

    return None


# ==============================================================
# GENERATION FUNCTION
# ==============================================================

def generate_response(messages):

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=6,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response.strip()


# ==============================================================
# ZERO-SHOT PROMPT
# ==============================================================

def zero_shot_prompt(hindi_text):

    return [
        {
            "role": "system",
            "content": (
                "You are a sentiment classification system. "
                "Classify the sentiment of the Hindi review. "
                "The only possible labels are Positive or Negative."
            )
        },
        {
            "role": "user",
            "content": (
                "Review:\n"
                f"{hindi_text}\n\n"
                "Output only one label: Positive or Negative."
            )
        }
    ]


# ==============================================================
# FEW-SHOT EXAMPLES
# ==============================================================

few_shot_examples = train_df[
    ["Hindi_Text", "Sentiment"]
].copy()

positive_examples = few_shot_examples[
    few_shot_examples["Sentiment"] == "Positive"
].head(2)

negative_examples = few_shot_examples[
    few_shot_examples["Sentiment"] == "Negative"
].head(2)

# ==============================================================
# FEW-SHOT PROMPT
# ==============================================================

def few_shot_prompt(hindi_text):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a sentiment classification system. "
                "Classify Hindi reviews as Positive or Negative. "
                "Learn the classification pattern from the examples. "
                "Output only one label: Positive or Negative."
            )
        }
    ]

    # Add real training examples
    for _, row in positive_examples.iterrows():

        messages.append(
            {
                "role": "user",
                "content": (
                    "Review:\n"
                    f"{row['Hindi_Text']}\n\n"
                    "Output only one label: Positive or Negative."
                )
            }
        )

        messages.append(
            {
                "role": "assistant",
                "content": "Positive"
            }
        )

    for _, row in negative_examples.iterrows():

        messages.append(
            {
                "role": "user",
                "content": (
                    "Review:\n"
                    f"{row['Hindi_Text']}\n\n"
                    "Output only one label: Positive or Negative."
                )
            }
        )

        messages.append(
            {
                "role": "assistant",
                "content": "Negative"
            }
        )

    # Actual test example
    messages.append(
        {
            "role": "user",
            "content": (
                "Review:\n"
                f"{hindi_text}\n\n"
                "Output only one label: Positive or Negative."
            )
        }
    )

    return messages


# ==============================================================
# CLASSIFICATION FUNCTION
# ==============================================================

def classify_dataset(df, mode):

    predictions = []

    print("\n" + "=" * 70)

    if mode == "zero-shot":
        print("RUNNING ZERO-SHOT LLM CLASSIFICATION")
    else:
        print("RUNNING FEW-SHOT LLM CLASSIFICATION")

    print("=" * 70)

    total = len(df)

    start_time = time.time()

    for index, row in df.iterrows():

        hindi_text = str(row["Hindi_Text"])

        if mode == "zero-shot":

            messages = zero_shot_prompt(
                hindi_text
            )

        else:

            messages = few_shot_prompt(
                hindi_text
            )

        response = generate_response(
            messages
        )

        prediction = normalize_prediction(
            response
        )

        predictions.append(
            prediction
        )

        # Progress
        if (index + 1) % 10 == 0 or index == 0:

            elapsed = time.time() - start_time

            print(
                f"Processed {index + 1}/{total} "
                f"| Last response: {response!r} "
                f"| Parsed: {prediction} "
                f"| Time: {elapsed:.1f}s"
            )

    return predictions


# ==============================================================
# METRICS
# ==============================================================

def calculate_metrics(
    actual,
    predictions,
    mode
):

    valid_indices = [
        i
        for i, prediction in enumerate(predictions)
        if prediction in ["Positive", "Negative"]
    ]

    invalid_count = (
        len(predictions) -
        len(valid_indices)
    )

    if invalid_count > 0:

        print(
            f"\nWARNING: {invalid_count} "
            "responses could not be parsed."
        )

    actual_valid = [
        actual.iloc[i]
        for i in valid_indices
    ]

    predicted_valid = [
        predictions[i]
        for i in valid_indices
    ]

    if len(predicted_valid) == 0:

        raise RuntimeError(
            "No valid LLM predictions were produced."
        )

    accuracy = accuracy_score(
        actual_valid,
        predicted_valid
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            actual_valid,
            predicted_valid,
            average="weighted",
            zero_division=0
        )
    )

    print("\n" + "=" * 70)
    print(f"{mode.upper()} RESULTS")
    print("=" * 70)

    print(
        f"\nValid predictions : "
        f"{len(predicted_valid)}/{len(predictions)}"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-score  : {f1:.4f}"
    )

    return {
        "Mode": mode,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Valid_Predictions": len(predicted_valid),
        "Total_Test_Samples": len(predictions)
    }


# ==============================================================
# RUN ZERO-SHOT
# ==============================================================

zero_predictions = classify_dataset(
    test_df,
    "zero-shot"
)

zero_results = calculate_metrics(
    test_df["Sentiment"],
    zero_predictions,
    "zero-shot"
)

# ==============================================================
# RUN FEW-SHOT
# ==============================================================

few_predictions = classify_dataset(
    test_df,
    "few-shot"
)

few_results = calculate_metrics(
    test_df["Sentiment"],
    few_predictions,
    "few-shot"
)

# ==============================================================
# SAVE PREDICTIONS
# ==============================================================

prediction_df = test_df[
    ["English_Text", "Hindi_Text", "Sentiment"]
].copy()

prediction_df[
    "LLM_ZeroShot"
] = zero_predictions

prediction_df[
    "LLM_FewShot"
] = few_predictions

prediction_file = os.path.join(
    RESULTS_METRICS,
    "llm_predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

# ==============================================================
# SAVE METRICS
# ==============================================================

metrics_df = pd.DataFrame(
    [zero_results, few_results]
)

metrics_file = os.path.join(
    RESULTS_METRICS,
    "llm_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)

# ==============================================================
# SAVE TEXT REPORT
# ==============================================================

report_file = os.path.join(
    RESULTS_METRICS,
    "llm_metrics.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "LLM HINDI SENTIMENT CLASSIFICATION\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Model: {MODEL_NAME}\n"
    )

    file.write(
        f"Test samples: {len(test_df)}\n\n"
    )

    for result in [
        zero_results,
        few_results
    ]:

        file.write(
            f"{result['Mode'].upper()}\n"
        )

        file.write(
            f"Accuracy  : "
            f"{result['Accuracy']:.4f}\n"
        )

        file.write(
            f"Precision : "
            f"{result['Precision']:.4f}\n"
        )

        file.write(
            f"Recall    : "
            f"{result['Recall']:.4f}\n"
        )

        file.write(
            f"F1-score  : "
            f"{result['F1']:.4f}\n\n"
        )

# ==============================================================
# DISPLAY FINAL RESULTS
# ==============================================================

print("\n")
print("=" * 70)
print("FINAL LLM RESULTS")
print("=" * 70)

print(
    metrics_df.to_string(index=False)
)

print("\nMetrics saved to:")
print(metrics_file)

print("\nPredictions saved to:")
print(prediction_file)

print("\nReport saved to:")
print(report_file)

print("\n" + "=" * 70)
print("LLM EXPERIMENT COMPLETED")
print("=" * 70)