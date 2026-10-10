import os
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================
# PATHS
# ==============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "metrics",
    "low_resource_results.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "figures"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "low_resource_learning_curve.png"
)

# ==============================================================
# LOAD RESULTS
# ==============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("LOW-RESOURCE LEARNING CURVE")
print("=" * 70)

print(df)

# ==============================================================
# PLOT
# ==============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    df["Training_Samples"],
    df["F1"],
    marker="o",
    linewidth=2,
    markersize=7
)

# Add values above points
for x, y in zip(
    df["Training_Samples"],
    df["F1"]
):
    plt.text(
        x,
        y + 0.015,
        f"{y:.3f}",
        ha="center",
        fontsize=10
    )

plt.xlabel("Number of Training Samples")
plt.ylabel("F1-score")

plt.title(
    "mBERT Performance Under Low-Resource Conditions"
)

plt.xticks(
    df["Training_Samples"]
)

plt.ylim(0.45, 0.95)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==============================================================
# COMPLETE
# ==============================================================

print("\nLearning curve saved to:")
print(OUTPUT_FILE)