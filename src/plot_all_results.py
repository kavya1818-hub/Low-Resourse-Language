import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================
# PATHS
# ==============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FIGURE_DIR = os.path.join(
    BASE_DIR,
    "results",
    "figures",
    "final"
)

METRICS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "metrics"
)

os.makedirs(FIGURE_DIR, exist_ok=True)


# ==============================================================
# HELPER FUNCTION
# READ METRICS FROM .TXT FILE
# ==============================================================

def read_txt_metrics(file_path):

    metrics = {}

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    patterns = {
        "accuracy": r"Accuracy\s*[:=]\s*([0-9.]+)",
        "precision": r"Precision\s*[:=]\s*([0-9.]+)",
        "recall": r"Recall\s*[:=]\s*([0-9.]+)",
        "f1": r"F1[- ]?score\s*[:=]\s*([0-9.]+)"
    }

    for metric, pattern in patterns.items():

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            metrics[metric] = float(
                match.group(1)
            )

        else:

            raise ValueError(
                f"{metric} not found in {file_path}"
            )

    return metrics


# ==============================================================
# READ SAVED MODEL METRICS
# ==============================================================

baseline_file = os.path.join(
    METRICS_DIR,
    "baseline_metrics.txt"
)

mbert_file = os.path.join(
    METRICS_DIR,
    "mbert_metrics.txt"
)

lora_file = os.path.join(
    METRICS_DIR,
    "mbert_lora_metrics.txt"
)

llm_file = os.path.join(
    METRICS_DIR,
    "llm_metrics.csv"
)


baseline = read_txt_metrics(
    baseline_file
)

mbert = read_txt_metrics(
    mbert_file
)

lora = read_txt_metrics(
    lora_file
)


# ==============================================================
# READ LLM METRICS
# ==============================================================

llm_df = pd.read_csv(
    llm_file
)

# Normalize column names
llm_df.columns = [
    column.strip()
    for column in llm_df.columns
]


def get_llm_row(mode):

    possible_columns = [
        column
        for column in llm_df.columns
        if column.lower() in [
            "mode",
            "method",
            "strategy",
            "prompting",
            "prompting_strategy"
        ]
    ]

    if not possible_columns:

        raise ValueError(
            "Could not find LLM mode column in llm_metrics.csv"
        )

    mode_column = possible_columns[0]

    rows = llm_df[
        llm_df[mode_column]
        .astype(str)
        .str.lower()
        .str.contains(
            mode.lower()
        )
    ]

    if rows.empty:

        raise ValueError(
            f"{mode} result not found in llm_metrics.csv"
        )

    return rows.iloc[0]


# Find metric columns regardless of capitalization
def find_column(name):

    for column in llm_df.columns:

        if column.lower() == name.lower():

            return column

    raise ValueError(
        f"{name} column not found in llm_metrics.csv"
    )


accuracy_column = find_column("Accuracy")
precision_column = find_column("Precision")
recall_column = find_column("Recall")
f1_column = find_column("F1")


zero_shot_row = get_llm_row(
    "zero-shot"
)

few_shot_row = get_llm_row(
    "few-shot"
)

llm_zero_shot = {
    "accuracy": float(zero_shot_row[accuracy_column]),
    "precision": float(zero_shot_row[precision_column]),
    "recall": float(zero_shot_row[recall_column]),
    "f1": float(zero_shot_row[f1_column])
}

llm_few_shot = {
    "accuracy": float(few_shot_row[accuracy_column]),
    "precision": float(few_shot_row[precision_column]),
    "recall": float(few_shot_row[recall_column]),
    "f1": float(few_shot_row[f1_column])
}


# ==============================================================
# DISPLAY LOADED RESULTS
# ==============================================================

print("\n" + "=" * 70)
print("LOADED MODEL METRICS")
print("=" * 70)

print(
    f"\nTF-IDF + Logistic Regression"
)

print(
    f"Accuracy  : {baseline['accuracy']:.4f}"
)

print(
    f"Precision : {baseline['precision']:.4f}"
)

print(
    f"Recall    : {baseline['recall']:.4f}"
)

print(
    f"F1        : {baseline['f1']:.4f}"
)


print(
    f"\nmBERT Fine-tuning"
)

print(
    f"Accuracy  : {mbert['accuracy']:.4f}"
)

print(
    f"Precision : {mbert['precision']:.4f}"
)

print(
    f"Recall    : {mbert['recall']:.4f}"
)

print(
    f"F1        : {mbert['f1']:.4f}"
)


print(
    f"\nmBERT + LoRA"
)

print(
    f"Accuracy  : {lora['accuracy']:.4f}"
)

print(
    f"Precision : {lora['precision']:.4f}"
)

print(
    f"Recall    : {lora['recall']:.4f}"
)

print(
    f"F1        : {lora['f1']:.4f}"
)


print(
    f"\nLLM Zero-shot"
)

print(
    f"Accuracy  : {llm_zero_shot['accuracy']:.4f}"
)

print(
    f"Precision : {llm_zero_shot['precision']:.4f}"
)

print(
    f"Recall    : {llm_zero_shot['recall']:.4f}"
)

print(
    f"F1        : {llm_zero_shot['f1']:.4f}"
)


print(
    f"\nLLM Few-shot"
)

print(
    f"Accuracy  : {llm_few_shot['accuracy']:.4f}"
)

print(
    f"Precision : {llm_few_shot['precision']:.4f}"
)

print(
    f"Recall    : {llm_few_shot['recall']:.4f}"
)

print(
    f"F1        : {llm_few_shot['f1']:.4f}"
)


# ==============================================================
# MODEL DATA FOR PLOTS
# ==============================================================

models = [
    "TF-IDF +\nLogistic Regression",
    "mBERT\nFine-tuning",
    "mBERT +\nLoRA",
    "LLM\nZero-shot",
    "LLM\nFew-shot"
]

accuracy = [
    baseline["accuracy"],
    mbert["accuracy"],
    lora["accuracy"],
    llm_zero_shot["accuracy"],
    llm_few_shot["accuracy"]
]

precision = [
    baseline["precision"],
    mbert["precision"],
    lora["precision"],
    llm_zero_shot["precision"],
    llm_few_shot["precision"]
]

recall = [
    baseline["recall"],
    mbert["recall"],
    lora["recall"],
    llm_zero_shot["recall"],
    llm_few_shot["recall"]
]

f1 = [
    baseline["f1"],
    mbert["f1"],
    lora["f1"],
    llm_zero_shot["f1"],
    llm_few_shot["f1"]
]

# ==============================================================
# PRINT MAIN RESULTS COMPARISON TABLE
# ==============================================================

comparison_df = pd.DataFrame({
    "Approach": [
        "TF-IDF + Logistic Regression",
        "mBERT full fine-tuning",
        "mBERT + LoRA",
        "Qwen zero-shot",
        "Qwen few-shot"
    ],
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1
})

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score"
]:
    comparison_df[column] = (
        comparison_df[column] * 100
    ).map(lambda value: f"{value:.2f}%")

print("\n" + "=" * 100)
print("MAIN TEST-SET RESULTS COMPARISON")
print("=" * 100)
print(comparison_df.to_string(index=False))
print("=" * 100)

# ==============================================================
# 0. DATASET CLASS DISTRIBUTION
# ==============================================================

sentiments = [
    "Negative",
    "Positive"
]

class_counts = [
    572,
    580
]

plt.figure(
    figsize=(8, 6)
)

bars = plt.bar(
    sentiments,
    class_counts
)

for bar, value in zip(
    bars,
    class_counts
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 10,
        str(value),
        ha="center",
        fontsize=12
    )

plt.title(
    "Sentiment Class Distribution in Hindi Dataset"
)

plt.xlabel(
    "Sentiment"
)

plt.ylabel(
    "Number of Samples"
)

plt.ylim(
    0,
    650
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "00_dataset_class_distribution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==============================================================
# 1. MODEL ACCURACY COMPARISON
# ==============================================================

plt.figure(
    figsize=(11, 6)
)

bars = plt.bar(
    models,
    accuracy
)

for bar, value in zip(
    bars,
    accuracy
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.2%}",
        ha="center",
        fontsize=10
    )

plt.title(
    "Accuracy Comparison of Hindi Sentiment Models"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Accuracy"
)

plt.ylim(
    0,
    1
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "01_model_accuracy_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==============================================================
# 2. MODEL F1 COMPARISON
# ==============================================================

plt.figure(
    figsize=(11, 6)
)

bars = plt.bar(
    models,
    f1
)

for bar, value in zip(
    bars,
    f1
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.2%}",
        ha="center",
        fontsize=10
    )

plt.title(
    "F1-score Comparison of Hindi Sentiment Models"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "F1-score"
)

plt.ylim(
    0,
    1
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "02_model_f1_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==============================================================
# 3. PRECISION vs RECALL COMPARISON
# ==============================================================

x = range(
    len(models)
)

width = 0.35

plt.figure(
    figsize=(12, 6)
)

precision_bars = plt.bar(
    [i - width / 2 for i in x],
    precision,
    width,
    label="Precision"
)

recall_bars = plt.bar(
    [i + width / 2 for i in x],
    recall,
    width,
    label="Recall"
)

for bar, value in zip(
    precision_bars,
    precision
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.2%}",
        ha="center",
        fontsize=9
    )

for bar, value in zip(
    recall_bars,
    recall
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.2%}",
        ha="center",
        fontsize=9
    )

plt.xticks(
    list(x),
    models
)

plt.title(
    "Precision and Recall Comparison"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Score"
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "03_precision_recall_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==============================================================
# 4. ZERO-SHOT vs FEW-SHOT
# ==============================================================

llm_models = [
    "Zero-shot",
    "Few-shot"
]

llm_accuracy = [
    llm_zero_shot["accuracy"],
    llm_few_shot["accuracy"]
]

llm_f1 = [
    llm_zero_shot["f1"],
    llm_few_shot["f1"]
]

x = range(
    len(llm_models)
)

width = 0.35

plt.figure(
    figsize=(8, 6)
)

accuracy_bars = plt.bar(
    [i - width / 2 for i in x],
    llm_accuracy,
    width,
    label="Accuracy"
)

f1_bars = plt.bar(
    [i + width / 2 for i in x],
    llm_f1,
    width,
    label="F1-score"
)

for bar, value in zip(
    accuracy_bars,
    llm_accuracy
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.2%}",
        ha="center"
    )

for bar, value in zip(
    f1_bars,
    llm_f1
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.2%}",
        ha="center"
    )

plt.xticks(
    list(x),
    llm_models
)

plt.title(
    "Effect of Few-shot Examples on LLM Performance"
)

plt.xlabel(
    "LLM Prompting Strategy"
)

plt.ylabel(
    "Score"
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "04_llm_zero_vs_few_shot.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==============================================================
# 5. LOW-RESOURCE ACCURACY AND F1
# ==============================================================

low_resource_file = os.path.join(
    METRICS_DIR,
    "low_resource_results.csv"
)

df = pd.read_csv(
    low_resource_file
)
# ==============================================================
# PRINT LOW-RESOURCE RESULTS TABLE
# ==============================================================

print("\n" + "=" * 70)
print("LOW-RESOURCE EXPERIMENT RESULTS")
print("=" * 70)

low_resource_table = df[
    ["Training_Samples", "Accuracy", "F1"]
].copy()

low_resource_table["Accuracy"] = (
    low_resource_table["Accuracy"] * 100
).map(lambda value: f"{value:.2f}%")

low_resource_table["F1"] = (
    low_resource_table["F1"] * 100
).map(lambda value: f"{value:.2f}%")

print(
    low_resource_table.to_string(index=False)
)

print("=" * 70)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    df["Training_Samples"],
    df["Accuracy"],
    marker="o",
    linewidth=2,
    label="Accuracy"
)

plt.plot(
    df["Training_Samples"],
    df["F1"],
    marker="s",
    linewidth=2,
    label="F1-score"
)

for x_value, y_value in zip(
    df["Training_Samples"],
    df["Accuracy"]
):

    plt.text(
        x_value,
        y_value + 0.02,
        f"{y_value:.3f}",
        ha="center",
        fontsize=9
    )

for x_value, y_value in zip(
    df["Training_Samples"],
    df["F1"]
):

    plt.text(
        x_value,
        y_value - 0.04,
        f"{y_value:.3f}",
        ha="center",
        fontsize=9
    )

plt.title(
    "mBERT Accuracy and F1-score Under Low-Resource Conditions"
)

plt.xlabel(
    "Number of Training Samples"
)

plt.ylabel(
    "Score"
)

plt.xticks(
    df["Training_Samples"]
)

plt.ylim(
    0.45,
    0.95
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "05_low_resource_accuracy_f1.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()



# ==============================================================
# 6. TRAINING DATA PERCENTAGE vs F1-SCORE
# ==============================================================

full_training_size = 806

training_samples = df[
    "Training_Samples"
]

training_percentage = (
    training_samples /
    full_training_size
) * 100

f1_scores = df["F1"]

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    training_percentage,
    f1_scores,
    marker="o",
    linewidth=2
)

for x_value, y_value, samples in zip(
    training_percentage,
    f1_scores,
    training_samples
):

    plt.text(
        x_value,
        y_value + 0.018,
        f"{x_value:.1f}%\nF1={y_value:.3f}",
        ha="center",
        fontsize=9
    )

plt.title(
    "mBERT F1-score vs Percentage of Available Training Data"
)

plt.xlabel(
    "Percentage of Available Training Data (%)"
)

plt.ylabel(
    "F1-score"
)

plt.xticks(
    training_percentage,
    [
        f"{x:.1f}%"
        for x in training_percentage
    ]
)

plt.ylim(
    0.45,
    0.95
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "06_training_percentage_vs_f1.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==============================================================
# 7. LoRA PARAMETER EFFICIENCY
# ==============================================================

lora_total_parameters = 178151428
lora_trainable_parameters = 296450

lora_trainable_percentage = (
    lora_trainable_parameters /
    lora_total_parameters
) * 100

lora_frozen_percentage = (
    100 -
    lora_trainable_percentage
)

parameter_types = [
    "Trainable",
    "Frozen"
]

parameter_percentages = [
    lora_trainable_percentage,
    lora_frozen_percentage
]

plt.figure(
    figsize=(9, 6)
)

bars = plt.bar(
    parameter_types,
    parameter_percentages
)

for bar, value in zip(
    bars,
    parameter_percentages
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.4f}%",
        ha="center",
        fontsize=11
    )

plt.title(
    "mBERT + LoRA Trainable Parameter Percentage"
)

plt.xlabel(
    "Parameter Type"
)

plt.ylabel(
    "Percentage of Total Parameters"
)

plt.ylim(
    0,
    110
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "07_lora_parameter_efficiency.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()
# ==============================================================
# FINAL MESSAGE
# ==============================================================

print(
    "\n" + "=" * 70
)

print(
    "ALL FINAL FIGURES GENERATED"
)

print(
    "=" * 70
)

print(
    "\nSaved to:"
)

print(
    FIGURE_DIR
)

print(
    "\nFiles created:"
)

for file in sorted(
    os.listdir(FIGURE_DIR)
):

    print(
        " -",
        file
    )