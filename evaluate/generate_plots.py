import logging
import sys
import os 
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc
from upsetplot import UpSet, from_memberships

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from data_access.data_pipeline import run_data_pipeline

MODEL_COLOURS = {
    "PNet":                         "#DC2626",
    "PNetSingle_output_layer_0":    "#2563EB",   
    "PNetSingle_output_layer_1":    "#9CA3AF",   
    "PNetSingle_output_layer_2":    "#9CA3AF",   
    "PNetSingle_output_layer_3":    "#9CA3AF",   
    "PNetSingle_output_layer_4":    "#9CA3AF", 
    "PNetSingle_output_layer_5":    "#78716C",  
    "ReactomeGNN":                  "#16A34A",  
    "Baseline":                     "#7C3AED",   
}

SINGLE_LAYER_LABELS = {
    "PNetSingle_output_layer_0": "Layer 0 (genes)",
    "PNetSingle_output_layer_1": "Layer 1",
    "PNetSingle_output_layer_2": "Layer 2",
    "PNetSingle_output_layer_3": "Layer 3",
    "PNetSingle_output_layer_4": "Layer 4",
    "PNetSingle_output_layer_5": "Layer 5 (top)",
    "PNet":                      "PNet (avg)",
}

MODEL_DISPLAY_NAMES = {
    "PNet": "PNet",
    "Baseline": "Baseline",
    "ReactomeGNN": "Reactome GNN"
}

PNET_ORDER = [
    "PNetSingle_output_layer_0",
    "PNetSingle_output_layer_1",
    "PNetSingle_output_layer_2",
    "PNetSingle_output_layer_3",
    "PNetSingle_output_layer_4",
    "PNetSingle_output_layer_5",
    "PNet",
]

DISSERTATION_STYLE = {
    "font.family":        "serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.5,
    "grid.alpha":         0.5,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
}

OUTPUT_DIR = os.path.join(os.getcwd(), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)



def load_eval_results(results_filename):
    load_path = os.path.join(os.getcwd(), results_filename)
    
    with open(load_path, "r") as f:
        raw = json.load(f)

    results = {}
    for model_name, data in raw.items():
        results[model_name] = {
            "metrics": data["metrics"],
            "probs": torch.tensor(data["probs"]),
            "preds": torch.tensor(data["preds"]).long()
        }

    logging.info(f"Loaded evaluation results for models: {list(results.keys())}")
    return results

def _savefig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path)
    print(f"Saved → {path}")
    plt.close(fig)

def _bar_chart(ax, keys, values, title, ylabel, colours, x_labels):
    x = np.arange(len(keys))
    bars = ax.bar(x, values, color=colours, width=0.55, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=7.5,
        )
    return ax

CAT_LABELS = ["TP", "FP", "TN", "FN"]
def _confusion_categories(preds: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Returns a 1-D array of per-sample confusion labels:
        0 = TP, 1 = FP, 2 = TN, 3 = FN
    """
    preds  = np.asarray(preds).flatten()
    y_true = np.asarray(y_true).flatten()
    cats = np.empty(len(y_true), dtype=int)
    cats[(preds == 1) & (y_true == 1)] = 0   # TP
    cats[(preds == 1) & (y_true == 0)] = 1   # FP
    cats[(preds == 0) & (y_true == 0)] = 2   # TN
    cats[(preds == 0) & (y_true == 1)] = 3   # FN
    return cats



# ── PNet layer-by-layer AUC and F1 bar charts ───────────────────
def plot_pnet_layer_barcharts(results):
    keys   = PNET_ORDER
    labels = [SINGLE_LAYER_LABELS[k] for k in keys]
    aucs   = [results[k]["metrics"]["auc"] for k in keys]
    f1s    = [results[k]["metrics"]["f1"]  for k in keys]
    cols   = [MODEL_COLOURS[k] for k in keys]

    with plt.rc_context(DISSERTATION_STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        _bar_chart(ax, keys, f1s, "PNet Output Layer Comparison — F1", "F1 Score", cols, labels)
        _savefig(fig, "pnet_layer_f1.pdf")



# ── PNet layer-by-layer ROC overlay ───────────────────────────────
def plot_pnet_roc_overlay_with_labels(results, y_true: np.ndarray):
    with plt.rc_context(DISSERTATION_STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))

        for key in PNET_ORDER:
            probs = results[key]["probs"].numpy().squeeze()
            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_auc = auc(fpr, tpr)
            label = f"{SINGLE_LAYER_LABELS[key]}  (AUC = {roc_auc:.3f})"
            ax.plot(fpr, tpr, color=MODEL_COLOURS[key], linewidth=1.6,
                    linestyle="--" if key != "PNet" else "-", label=label)

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.set_title("ROC Curves — PNet Output Layers", fontsize=10, fontweight="bold", pad=8)
        ax.legend(fontsize=7.5, loc="lower right")
        _savefig(fig, "pnet_roc_overlay.pdf")



# ── PNet vs Baseline vs GNN side-by-side metric bar chart ─────────

def plot_baseline_comparison_barchart(results, models, plot_name):
    metrics = ["auc", "f1", "precision", "recall", "accuracy"]
    labels  = ["AUC", "F1", "Precision", "Recall", "Accuracy"]

    n_metrics = len(metrics)
    n_models  = len(models)
    x = np.arange(n_metrics)
    width = 0.22

    with plt.rc_context(DISSERTATION_STYLE):
        fig, ax = plt.subplots(figsize=(9, 4.5))

        for i, model in enumerate(models):
            vals   = [results[model]["metrics"][m] for m in metrics]
            offset = (i - n_models / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, vals,
                width=width,
                color=MODEL_COLOURS[model],
                edgecolor="white",
                linewidth=0.5,
                label=model,
            )
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=6.5,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_ylim(0, 1.12)
        title_models = " vs ".join(MODEL_DISPLAY_NAMES[m] for m in models)
        ax.set_title(f"ROC Curves — {title_models}",
                    fontsize=10, fontweight="bold", pad=8)
        ax.legend(fontsize=8)
        _savefig(fig, plot_name)



# ── PNet vs Baseline vs GNN ROC overlay ───────────────────────────

def plot_baseline_roc_overlay(results, y_true, models, plot_name):
    with plt.rc_context(DISSERTATION_STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))

        for model in models:
            probs = results[model]["probs"].numpy().squeeze()
            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=MODEL_COLOURS[model], linewidth=2,
                    label=f"{model}  (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        title_models = " vs ".join(MODEL_DISPLAY_NAMES[m] for m in models)
        ax.set_title(f"ROC Curves — {title_models}",
                    fontsize=10, fontweight="bold", pad=8)
        ax.legend(fontsize=8, loc="lower right")
        _savefig(fig, plot_name)



# ── Individual confusion matrices ─────────────────────────────────

def plot_individual_confusion_matrices(results, y_true: np.ndarray):
    """
    One 2x2 confusion matrix per model (PNet, ReactomeGNN, Baseline),
    arranged side by side. Cells are annotated with raw counts and
    row-normalised percentages.
    """
    models       = ["PNet", "ReactomeGNN", "Baseline"]

    with plt.rc_context(DISSERTATION_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle("Confusion Matrices — PNet, ReactomeGNN, Baseline",
                     fontsize=11, fontweight="bold", y=1.02)

        for ax, model in zip(axes, models):
            preds  = np.asarray(results[model]["preds"]).flatten()
            labels = np.asarray(y_true).flatten()

            # rows = actual, cols = predicted
            # [[TN, FP],
            #  [FN, TP]]
            tn = int(np.sum((preds == 0) & (labels == 0)))
            fp = int(np.sum((preds == 1) & (labels == 0)))
            fn = int(np.sum((preds == 0) & (labels == 1)))
            tp = int(np.sum((preds == 1) & (labels == 1)))

            matrix = np.array([[tn, fp],
                                [fn, tp]])

            # Row-normalised (recall perspective: of all actual X, how many predicted correctly)
            row_sums = matrix.sum(axis=1, keepdims=True)
            norm     = matrix / row_sums.clip(min=1)

            im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

            cell_labels = [["TN", "FP"], ["FN", "TP"]]
            for i in range(2):
                for j in range(2):
                    count   = matrix[i, j]
                    pct     = norm[i, j] * 100
                    txt_col = "white" if norm[i, j] > 0.55 else "black"
                    ax.text(j, i - 0.12,
                            f"{cell_labels[i][j]}  {count}",
                            ha="center", va="center",
                            fontsize=12, fontweight="bold", color=txt_col)
                    ax.text(j, i + 0.18,
                            f"({pct:.1f}%)",
                            ha="center", va="center",
                            fontsize=9, color=txt_col)

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Predicted\nNegative", "Predicted\nPositive"], fontsize=8)
            ax.set_yticklabels(["Actual\nNegative", "Actual\nPositive"], fontsize=8)
            ax.set_title(model, fontsize=10, fontweight="bold",
                         color=MODEL_COLOURS[model], pad=10)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
                "Row-normalised rate", fontsize=7, color="grey"
            )

        plt.tight_layout()
        _savefig(fig, "individual_confusion_matrices.pdf")



# ── Pairwise error consistency matrices ────────────────────────────

def plot_error_consistency_matrices(results, y_true: np.ndarray):
    """
    For every pair of (model_a, model_b), plots a '4x4' heatmap where cell (i,j)
    is the number of test samples classified as category i by model_a and
    category j by model_b.  Categories: TP, FP, TN, FN.

    Three model pairs → 3 subplots arranged in a single row.
    """
    models = ["PNet", "ReactomeGNN", "Baseline"]
    pairs  = [("PNet", "ReactomeGNN"), ("PNet", "Baseline"), ("ReactomeGNN", "Baseline")]

    # Pre-compute per-sample confusion categories for each model
    cats = {
        m: _confusion_categories(results[m]["preds"].numpy().squeeze(), y_true)
        for m in models
    }

    with plt.rc_context(DISSERTATION_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.suptitle("Error Consistency Matrices", fontsize=11, fontweight="bold", y=1.02)

        for ax, (model_a, model_b) in zip(axes, pairs):
            # Build 4×4 count matrix
            matrix = np.zeros((4, 4), dtype=int)
            for i in range(4):
                for j in range(4):
                    matrix[i, j] = int(np.sum((cats[model_a] == i) & (cats[model_b] == j)))

            # Colour scale: white → model_a colour, so each panel has its own tint
            base_colour = MODEL_COLOURS[model_a]

            im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0)

            # Annotate every cell with the count
            for i in range(4):
                for j in range(4):
                    count = matrix[i, j]
                    # Use white text on dark cells, dark on light
                    text_colour = "white" if count > matrix.max() * 0.55 else "black"
                    ax.text(j, i, str(count), ha="center", va="center",
                            fontsize=11, fontweight="bold", color=text_colour)

            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(CAT_LABELS, fontsize=9)
            ax.set_yticklabels(CAT_LABELS, fontsize=9)
            ax.set_xlabel(model_b, fontsize=9, labelpad=6)
            ax.set_ylabel(model_a, fontsize=9, labelpad=6)
            ax.set_title(f"{model_a}  vs  {model_b}", fontsize=9, fontweight="bold", pad=8)

            # Diagonal highlight — these are cells where both models agree
            for k in range(4):
                ax.add_patch(plt.Rectangle(
                    (k - 0.5, k - 0.5), 1, 1,
                    fill=False, edgecolor="#F59E0B", linewidth=1.8, linestyle="--"
                ))

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
                "Sample count", fontsize=7, color="grey"
            )

        plt.tight_layout()
        _savefig(fig, "error_consistency_matrices.pdf")



def _apply_model_styles(upset):
    models = ["PNet", "ReactomeGNN", "Baseline"]
    for m in models:
        upset.style_subsets(
            present=m,
            absent=[x for x in models if x != m],
            facecolor=MODEL_COLOURS[m],
            label=f"{m} only",
        )


def plot_upset_correct(results, y_true: np.ndarray):
    """
    UpSet plot showing the overlap of correctly classified test samples across
    PNet, ReactomeGNN, and Baseline.
    """
    models = ["PNet", "ReactomeGNN", "Baseline"]

    correct = {
        m: (results[m]["preds"].numpy().squeeze() == y_true)
        for m in models
    }

    memberships = [
        [m for m in models if correct[m][i]]
        for i in range(len(y_true))
    ]

    data = from_memberships(memberships)

    with plt.rc_context(DISSERTATION_STYLE):
        fig = plt.figure(figsize=(9, 5))

        upset = UpSet(
            data,
            subset_size="count",
            show_counts=True,
            sort_by="cardinality",
            totals_plot_elements=3,
        )

        _apply_model_styles(upset)
        upset.plot(fig)
        fig.suptitle(
            "Correctly Classified Samples — Model Agreement (UpSet Plot)",
            fontsize=10, fontweight="bold", y=1.02
        )

        _savefig(fig, "upset_correct_samples.pdf")


def plot_upset_incorrect(results, y_true: np.ndarray):
    """
    UpSet plot showing the overlap of incorrectly classified test samples across
    PNet, ReactomeGNN, and Baseline. The 'all models wrong' bar reveals the
    hard samples no model can handle.
    """
    models = ["PNet", "ReactomeGNN", "Baseline"]

    incorrect = {
        m: (results[m]["preds"].numpy().squeeze() != y_true)
        for m in models
    }

    memberships = [
        [m for m in models if incorrect[m][i]]
        for i in range(len(y_true))
    ]

    # Filter out samples all models got right (empty membership not relevant)
    memberships = [m for m in memberships if len(m) > 0]

    data = from_memberships(memberships)

    with plt.rc_context(DISSERTATION_STYLE):
        fig = plt.figure(figsize=(9, 5))

        upset = UpSet(
            data,
            subset_size="count",
            show_counts=True,
            sort_by="cardinality",
            totals_plot_elements=3,
        )

        _apply_model_styles(upset)
        upset.plot(fig)
        fig.suptitle(
            "Incorrectly Classified Samples — Shared Errors (UpSet Plot)",
            fontsize=10, fontweight="bold", y=1.02
        )

        _savefig(fig, "upset_incorrect_samples.pdf")



def generate_all_plots(y_true, results_filename="evaluation_results.json"):
    results = load_eval_results(results_filename)
    y_true  = np.asarray(y_true).flatten()

    plot_pnet_layer_barcharts(results)
    plot_pnet_roc_overlay_with_labels(results, y_true)

    models = ["PNet", "Baseline"]
    barchart_name = "pnet_baseline_comparison_barchart.pdf"
    roc_curve_name = "pnet_baseline_comparison_roc.pdf"
    plot_baseline_comparison_barchart(results, models, barchart_name)
    plot_baseline_roc_overlay(results, y_true, models, roc_curve_name)

    models = ["PNet", "ReactomeGNN"]
    barchart_name = "pnet_gnn_comparison_barchart.pdf"
    roc_curve_name = "pnet_gnn_comparison_roc.pdf"
    plot_baseline_comparison_barchart(results, models, barchart_name)
    plot_baseline_roc_overlay(results, y_true, models, roc_curve_name)

    plot_individual_confusion_matrices(results, y_true)
    plot_error_consistency_matrices(results, y_true)
    plot_upset_correct(results, y_true)
    plot_upset_incorrect(results, y_true)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    _, _, test_loader = run_data_pipeline()
    y_true = torch.cat([y for _, y in test_loader]).numpy()
    generate_all_plots(y_true)