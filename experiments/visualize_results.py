from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

DETECTORS = ["Modified Z-Score", "Isolation Forest"]


# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def load_dataset(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def save_figure(filename):
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Generated: {filename}")


def save_summary(data, group_columns, metrics, filename):
    summary = (
        data.groupby(group_columns)[metrics]
        .agg(["mean", "std"])
        .round(4)
    )

    summary.to_csv(TABLE_DIR / filename)

    print(f"Generated: {filename}")


# --------------------------------------------------
# 3. CONCEPT DRIFT ANALYSIS
# --------------------------------------------------

def visualize_concept_drift():

    data = load_dataset("concept_drift_results.csv")

    summary = (
        data.groupby(["phase", "detector"])
        ["false_positive_rate"]
        .agg(["mean", "std"])
        .reset_index()
    )

    phase_order = [
        "Baseline",
        "Mild Drift",
        "Moderate Drift",
        "Strong Drift",
        "Severe Drift"
    ]

    summary["phase"] = pd.Categorical(
        summary["phase"],
        categories=phase_order,
        ordered=True
    )

    summary = summary.sort_values("phase")

    plt.figure(figsize=(10, 6))

    for detector in DETECTORS:

        subset = summary[
            summary["detector"] == detector
        ]

        plt.errorbar(
            subset["phase"].astype(str),
            subset["mean"],
            yerr=subset["std"],
            marker="o",
            capsize=5,
            label=detector
        )

    plt.title("False Positive Rate Under Concept Drift")
    plt.xlabel("Concept Drift Condition")
    plt.ylabel("Mean False Positive Rate")
    plt.ylim(bottom=0, top=1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_figure("concept_drift_comparison.png")

    summary.to_csv(
        TABLE_DIR / "concept_drift_summary.csv",
        index=False
    )


# --------------------------------------------------
# 4. ANOMALY PREVALENCE ANALYSIS
# --------------------------------------------------

def visualize_anomaly_prevalence():

    data = load_dataset("anomaly_prevalence_results.csv")

    summary = (
        data.groupby(["anomaly_prevalence", "detector"])
        ["f1_score"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(10, 6))

    for detector in DETECTORS:

        subset = summary[
            summary["detector"] == detector
        ].sort_values("anomaly_prevalence")

        plt.errorbar(
            subset["anomaly_prevalence"] * 100,
            subset["mean"],
            yerr=subset["std"],
            marker="o",
            capsize=5,
            label=detector
        )

    plt.title("Detector Performance Across Anomaly Prevalence")
    plt.xlabel("Anomaly Prevalence (%)")
    plt.ylabel("Mean F1 Score")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_figure("anomaly_prevalence_comparison.png")

    summary.to_csv(
        TABLE_DIR / "anomaly_prevalence_summary.csv",
        index=False
    )


# --------------------------------------------------
# 5. ANOMALY SEVERITY ANALYSIS
# --------------------------------------------------

def visualize_anomaly_severity():

    data = load_dataset("anomaly_severity_results.csv")

    summary = (
        data.groupby(["severity", "detector"])
        [["precision", "recall", "f1_score", "accuracy"]]
        .mean()
        .reset_index()
    )

    severity_order = ["Mild", "Moderate", "Severe"]

    plt.figure(figsize=(10, 6))

    for detector in DETECTORS:

        subset = summary[
            summary["detector"] == detector
        ].copy()

        subset["severity"] = pd.Categorical(
            subset["severity"],
            categories=severity_order,
            ordered=True
        )

        subset = subset.sort_values("severity")

        plt.plot(
            subset["severity"].astype(str),
            subset["f1_score"],
            marker="o",
            label=detector
        )

    plt.title("Anomaly Detection Performance by Severity")
    plt.xlabel("Anomaly Severity")
    plt.ylabel("Mean F1 Score")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_figure("anomaly_severity_comparison.png")

    summary.to_csv(
        TABLE_DIR / "anomaly_severity_summary.csv",
        index=False
    )


# --------------------------------------------------
# 6. FALSE ALARM ANALYSIS
# --------------------------------------------------

def visualize_false_alarm_rate():

    data = load_dataset("false_alarm_results.csv")

    summary = (
        data.groupby("detector")["false_positive_rate"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(8, 6))

    plt.bar(
        summary["detector"],
        summary["mean"],
        yerr=summary["std"],
        capsize=6
    )

    plt.title("False Alarm Rate Comparison")
    plt.xlabel("Detector")
    plt.ylabel("Mean False Positive Rate")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)

    save_figure("false_alarm_comparison.png")

    summary.to_csv(
        TABLE_DIR / "false_alarm_summary.csv",
        index=False
    )


# --------------------------------------------------
# 7. DATASET ROBUSTNESS ANALYSIS
# --------------------------------------------------

def visualize_robustness():

    data = load_dataset("robustness_experiment_results.csv")

    metrics = [
        "precision",
        "recall",
        "f1_score",
        "accuracy"
    ]

    summary = (
        data.groupby("detector")[metrics]
        .mean()
        .reset_index()
    )

    summary.to_csv(
        TABLE_DIR / "robustness_summary.csv",
        index=False
    )

    for metric in metrics:

        plt.figure(figsize=(8, 6))

        plt.bar(
            summary["detector"],
            summary[metric]
        )

        plt.title(
            f"Detector Robustness — {metric.replace('_', ' ').title()}"
        )

        plt.xlabel("Detector")
        plt.ylabel(f"Mean {metric.replace('_', ' ').title()}")
        plt.ylim(0, 1.05)
        plt.grid(axis="y", alpha=0.3)

        save_figure(f"robustness_{metric}.png")


# --------------------------------------------------
# 8. MAIN EXECUTION
# --------------------------------------------------

def main():

    print("\nTAAHIS EXPERIMENTAL VISUALIZATION ENGINE")
    print("=" * 50)

    visualize_concept_drift()
    visualize_anomaly_prevalence()
    visualize_anomaly_severity()
    visualize_false_alarm_rate()
    visualize_robustness()

    print("\nVisualization completed successfully.")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Tables: {TABLE_DIR}")


if __name__ == "__main__":
    main()
