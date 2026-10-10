import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
TEST_PATH = PROJECT_DIR / "dataset" / "processed" / "test.csv"

RESULTS_DIR = PROJECT_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("TF-IDF + LOGISTIC REGRESSION BASELINE")
print("=" * 70)

train_df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
test_df = pd.read_csv(TEST_PATH, encoding="utf-8-sig")

print(f"\nTraining samples : {len(train_df)}")
print(f"Test samples     : {len(test_df)}")


# ============================================================
# 3. SELECT INPUT AND LABEL
# ============================================================

X_train = train_df["Hindi_Text"].astype(str)
y_train = train_df["Sentiment"]

X_test = test_df["Hindi_Text"].astype(str)
y_test = test_df["Sentiment"]


# ============================================================
# 4. TF-IDF VECTORIZATION
# ============================================================

print("\n" + "=" * 70)
print("TF-IDF VECTORIZATION")
print("=" * 70)

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    min_df=2,
    max_features=10000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Training TF-IDF shape : {X_train_tfidf.shape}")
print(f"Test TF-IDF shape     : {X_test_tfidf.shape}")


# ============================================================
# 5. TRAIN LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Training completed.")


# ============================================================
# 6. PREDICTION
# ============================================================

y_pred = model.predict(X_test_tfidf)


# ============================================================
# 7. EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 70)
print("BASELINE RESULTS")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")


# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print(report)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Negative", "Positive"]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.title("TF-IDF + Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

confusion_path = FIGURES_DIR / "baseline_confusion_matrix.png"

plt.savefig(confusion_path, dpi=300)

plt.close()


# ============================================================
# 10. SAVE METRICS
# ============================================================

metrics_path = METRICS_DIR / "baseline_metrics.txt"

with open(metrics_path, "w", encoding="utf-8") as file:

    file.write("TF-IDF + Logistic Regression Baseline\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Training samples: {len(train_df)}\n")
    file.write(f"Test samples: {len(test_df)}\n\n")

    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1-score: {f1:.4f}\n\n")

    file.write("Classification Report\n")
    file.write("-" * 50 + "\n")
    file.write(report)


# ============================================================
# 11. SHOW SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

for i in range(min(5, len(test_df))):

    print(f"\nExample {i + 1}")
    print("-" * 60)

    print("Hindi:")
    print(test_df.iloc[i]["Hindi_Text"])

    print("\nEnglish:")
    print(test_df.iloc[i]["English_Text"])

    print("\nActual:")
    print(y_test.iloc[i])

    print("Predicted:")
    print(y_pred[i])


# ============================================================
# 12. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("BASELINE IMPLEMENTATION COMPLETED")
print("=" * 70)

print(f"\nMetrics saved to:")
print(metrics_path)

print(f"\nConfusion matrix saved to:")
print(confusion_path)