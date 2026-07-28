import os
import pandas as pd

os.makedirs("tables", exist_ok=True)

# =========================
# 1. CV Results Table
# =========================
def save_cv_table(fold_scores):
    df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(len(fold_scores))],
        "Accuracy": fold_scores
    })

    df.loc["Mean"] = ["Mean", df["Accuracy"].mean()]
    df.loc["Std"] = ["Std", df["Accuracy"].std()]

    df.to_csv("tables/cv_results.csv", index=False)


# =========================
# 2. Classification Report Table
# =========================
def save_classification_table(report_dict):
    df = pd.DataFrame(report_dict).transpose()
    df.to_csv("tables/classification_report.csv")
