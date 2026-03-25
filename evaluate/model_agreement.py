import os
import sys
import json
import logging
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from data_access.data_pipeline import run_data_pipeline

def load_eval_results(filename="evaluation_results.json"):
    load_path = os.path.join(os.getcwd(), filename)
    with open(load_path, "r") as f:
        raw = json.load(f)

    results = {}
    for model_name, data in raw.items():
        results[model_name] = {
            "metrics": data["metrics"],
            "probs":   torch.tensor(data["probs"]),
            "preds":   torch.tensor(data["preds"]).long(),
        }
    return results

def analyse_unique_correct(results, y_true):
    """
    For each model, find samples it uniquely correctly classifies
    (i.e. the other two models get wrong) and report the
    primary/metastatic breakdown.
    """
    models = ["PNet", "ReactomeGNN", "Baseline"]
    y_true = np.asarray(y_true).flatten()

    correct = {
        m: (results[m]["preds"].numpy().squeeze() == y_true)
        for m in models
    }

    print("=" * 55)
    print("Unique Correct Classifications — Primary/Metastatic Breakdown")
    print("=" * 55)

    for m in models:
        others = [o for o in models if o != m]
        unique_mask = (
            correct[m] &
            ~correct[others[0]] &
            ~correct[others[1]]
        )
        unique_indices = np.where(unique_mask)[0]
        unique_labels  = y_true[unique_indices]

        n_total      = unique_mask.sum()
        n_metastatic = (unique_labels == 1).sum()
        n_primary    = (unique_labels == 0).sum()

        print(f"\n{m}:")
        print(f"  Total unique correct:  {n_total}")
        print(f"  Metastatic (label=1):  {n_metastatic} "
              f"({100 * n_metastatic / n_total:.1f}%)" if n_total > 0
              else "  Metastatic (label=1):  0")
        print(f"  Primary    (label=0):  {n_primary} "
              f"({100 * n_primary / n_total:.1f}%)" if n_total > 0
              else "  Primary    (label=0):  0")


def analyse_unique_errors(results, y_true):
    """
    For each model, find samples it uniquely misclassifies
    (i.e. the other two models get right) and report the
    primary/metastatic breakdown of the true labels,
    as well as the error type (FP or FN).
    """
    models = ["PNet", "ReactomeGNN", "Baseline"]
    y_true = np.asarray(y_true).flatten()

    correct = {
        m: (results[m]["preds"].numpy().squeeze() == y_true)
        for m in models
    }

    print("=" * 55)
    print("Unique Errors — Primary/Metastatic and Error Type Breakdown")
    print("=" * 55)

    for m in models:
        others = [o for o in models if o != m]
        unique_mask = (
            ~correct[m] &
            correct[others[0]] &
            correct[others[1]]
        )
        unique_indices = np.where(unique_mask)[0]
        unique_labels  = y_true[unique_indices]
        unique_preds   = results[m]["preds"].numpy().squeeze()[unique_indices]

        n_total = unique_mask.sum()
        n_fn    = int(((unique_preds == 0) & (unique_labels == 1)).sum())  # missed metastatic
        n_fp    = int(((unique_preds == 1) & (unique_labels == 0)).sum())  # false metastatic

        print(f"\n{m}:")
        print(f"  Total unique errors:   {n_total}")
        print(f"  FN (missed metastatic): {n_fn} "
              f"({100 * n_fn / n_total:.1f}%)" if n_total > 0
              else "  FN (missed metastatic): 0")
        print(f"  FP (false metastatic):  {n_fp} "
              f"({100 * n_fp / n_total:.1f}%)" if n_total > 0
              else "  FP (false metastatic):  0")


def analyse_universally_missed(results, y_true):
    models = ["PNet", "ReactomeGNN", "Baseline"]
    y_true = np.asarray(y_true).flatten()

    correct = {
        m: (results[m]["preds"].numpy().squeeze() == y_true)
        for m in models
    }

    all_wrong_mask = ~correct["PNet"] & ~correct["ReactomeGNN"] & ~correct["Baseline"]
    indices        = np.where(all_wrong_mask)[0]
    labels         = y_true[indices]

    print("=" * 55)
    print("Universally Missed Samples")
    print("=" * 55)
    print(f"  Total:               {all_wrong_mask.sum()}")
    print(f"  Metastatic (label=1): {(labels == 1).sum()}")
    print(f"  Primary    (label=0): {(labels == 0).sum()}")
    print(f"\n  Per-model error types for these samples:")

    for m in models:
        preds  = results[m]["preds"].numpy().squeeze()[indices]
        n_fn   = int(((preds == 0) & (labels == 1)).sum())
        n_fp   = int(((preds == 1) & (labels == 0)).sum())
        print(f"    {m}: FN={n_fn}, FP={n_fp}")

def analyse_complementary_behaviour(results, y_true):
    """
    Checks whether P-NET's uniquely correct samples overlap with the
    Baseline's uniquely incorrect samples and vice versa, to assess
    complementary behaviour between the two models.
    Also reports total error breakdowns by class for P-NET and Baseline.
    """
    models = ["PNet", "ReactomeGNN", "Baseline"]
    y_true = np.asarray(y_true).flatten()

    correct = {
        m: (results[m]["preds"].numpy().squeeze() == y_true)
        for m in models
    }

    pnet_unique_correct = (
        correct["PNet"] &
        ~correct["ReactomeGNN"] &
        ~correct["Baseline"]
    )

    baseline_unique_correct = (
        correct["Baseline"] &
        ~correct["PNet"] &
        ~correct["ReactomeGNN"]
    )

    baseline_unique_errors = (
        ~correct["Baseline"] &
        correct["PNet"] &
        correct["ReactomeGNN"]
    )

    pnet_unique_errors = (
        ~correct["PNet"] &
        correct["Baseline"] &
        correct["ReactomeGNN"]
    )

    pnet_correct_baseline_wrong = pnet_unique_correct & baseline_unique_errors
    baseline_correct_pnet_wrong = baseline_unique_correct & pnet_unique_errors

    print("=" * 55)
    print("Complementary Behaviour — P-NET vs Baseline")
    print("=" * 55)

    overlap_1 = pnet_correct_baseline_wrong.sum()
    indices_1 = np.where(pnet_correct_baseline_wrong)[0]
    labels_1  = y_true[indices_1]
    print(f"\nP-NET uniquely correct AND Baseline uniquely wrong: {overlap_1}")
    if overlap_1 > 0:
        print(f"  True labels: {labels_1.tolist()}")
        print(f"  Metastatic (label=1): {(labels_1 == 1).sum()}")
        print(f"  Primary    (label=0): {(labels_1 == 0).sum()}")

    overlap_2 = baseline_correct_pnet_wrong.sum()
    indices_2 = np.where(baseline_correct_pnet_wrong)[0]
    labels_2  = y_true[indices_2]
    print(f"\nBaseline uniquely correct AND P-NET uniquely wrong: {overlap_2}")
    if overlap_2 > 0:
        print(f"  True labels: {labels_2.tolist()}")
        print(f"  Metastatic (label=1): {(labels_2 == 1).sum()}")
        print(f"  Primary    (label=0): {(labels_2 == 0).sum()}")

    print(f"\nInterpretation:")
    total_complementary = overlap_1 + overlap_2
    print(f"  Total samples where models complement each other: "
          f"{total_complementary}")
    print(f"  These are cases an ensemble of P-NET and Baseline "
          f"could potentially resolve correctly.")

    print()
    print("=" * 55)
    print("Total Error Breakdown by Class")
    print("=" * 55)

    for model_name, model_key in [("P-NET", "PNet"), ("Baseline", "Baseline")]:
        errors      = ~correct[model_key]
        error_idx   = np.where(errors)[0]
        error_labels = y_true[error_idx]

        total_errors      = errors.sum()
        metastatic_errors = (error_labels == 1).sum()
        primary_errors    = (error_labels == 0).sum()

        # Per-class error rates
        metastatic_mask = (y_true == 1)
        primary_mask    = (y_true == 0)

        meta_error_rate    = (errors & metastatic_mask).sum() / metastatic_mask.sum()
        primary_error_rate = (errors & primary_mask).sum()    / primary_mask.sum()

        print(f"\n{model_name} — Total errors: {total_errors} "
              f"/ {len(y_true)} ({100 * total_errors / len(y_true):.1f}%)")
        print(f"  Metastatic (label=1): {metastatic_errors} errors "
              f"/ {metastatic_mask.sum()} samples "
              f"({100 * meta_error_rate:.1f}% error rate)")
        print(f"  Primary    (label=0): {primary_errors} errors "
              f"/ {primary_mask.sum()} samples "
              f"({100 * primary_error_rate:.1f}% error rate)")


def check_model_agreement(y_true):
    results = load_eval_results()
    analyse_unique_correct(results, y_true)
    analyse_unique_errors(results, y_true)
    analyse_universally_missed(results, y_true)
    analyse_complementary_behaviour(results, y_true)


if __name__ == '__main__':
    _, _, test_loader = run_data_pipeline()
    labels = torch.cat([y for _, y in test_loader]).numpy()
    check_model_agreement(labels)