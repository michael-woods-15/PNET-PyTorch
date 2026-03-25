import os
import sys
import json
import logging
import numpy as np
import torch
from statsmodels.stats.contingency_tables import mcnemar

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from data_access.data_pipeline import run_data_pipeline


PNET_VARIANTS = [
    "PNetSingle_output_layer_0",
    "PNetSingle_output_layer_1",
    "PNetSingle_output_layer_2",
    "PNetSingle_output_layer_3",
    "PNetSingle_output_layer_4",
    "PNetSingle_output_layer_5",
]


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


def build_contingency_table(preds_a, preds_b, labels):
    """
    Builds the 2x2 McNemar contingency table:

        [[a, b],    a = both correct
         [c, d]]    b = A correct, B wrong
                    c = A wrong,   B correct
                    d = both wrong

    Only b and c (the disagreements) factor into the test statistic.
    """
    preds_a = np.asarray(preds_a).flatten()
    preds_b = np.asarray(preds_b).flatten()
    labels  = np.asarray(labels).flatten()

    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    a = int(np.sum( correct_a &  correct_b))  # both correct
    b = int(np.sum( correct_a & ~correct_b))  # A correct, B wrong
    c = int(np.sum(~correct_a &  correct_b))  # A wrong, B correct
    d = int(np.sum(~correct_a & ~correct_b))  # both wrong

    return np.array([[a, b], [c, d]])

def run_mcnemar(preds_a, preds_b, labels):
    """
    Runs McNemar's test using the statsmodels exact binomial method.

    The exact test is preferred over the chi-squared approximation when
    b + c < 25.
    """
    table = build_contingency_table(preds_a, preds_b, labels)
    b, c  = table[0, 1], table[1, 0]

    if (b + c) == 0:
        return {
            "table":     table,
            "b":         b,
            "c":         c,
            "statistic": None,
            "p_value":   1.0,
            "note":      "No disagreements — models are identical on this test set",
        }

    result = mcnemar(table, exact=True, correction=False)

    return {
        "table":     table,
        "b":         b,
        "c":         c,
        "statistic": result.statistic,
        "p_value":   result.pvalue,
    }


def run_pnet_layers_mcnemar_tests(results, labels, alpha = 0.05):
    """
    Compares averaged PNet against each single-output layer variant using
    McNemar's exact test.

    Applies Bonferroni correction across the 6 simultaneous comparisons,
    adjusting the significance threshold to alpha / n_comparisons.
    """
    n_comparisons      = len(PNET_VARIANTS)
    alpha_corrected    = alpha / n_comparisons

    pnet_preds = results["PNet"]["preds"].numpy().flatten()

    print("\nMcNemar's Exact Test — PNet (averaged) vs Single Output Layers")
    print(f"Bonferroni-corrected α = {alpha} / {n_comparisons} = {alpha_corrected:.4f}\n")
    print(f"{'Comparison':<40} {'b':>5} {'c':>5} {'p-value':>10} {'Significant':>12}")
    print("-" * 75)

    test_results = {}
    for variant in PNET_VARIANTS:
        variant_preds = results[variant]["preds"].numpy().flatten()
        result = run_mcnemar(pnet_preds, variant_preds, labels)

        label = variant.replace("PNetSingle_output_layer_", "PNet vs Single Output Layer ")
        sig   = "Yes" if result["p_value"] < alpha_corrected else "No"

        if "note" in result:
            print(f"{label:<40} {'—':>5} {'—':>5} {'—':>10} {'—':>12}  ({result['note']})")
        else:
            print(f"{label:<40} {result['b']:>5} {result['c']:>5} "
                  f"{result['p_value']:>10.4f} {sig:>12}")

        test_results[variant] = result

    print()
    return test_results


def run_pnet_vs_baseline_mcnemar(results, labels, alpha = 0.05):
    """
    McNemar's exact test comparing PNet (averaged) against the Dense Baseline.
    Single comparison — no Bonferroni correction needed.
    """
    pnet_preds     = results["PNet"]["preds"].numpy().flatten()
    baseline_preds = results["Baseline"]["preds"].numpy().flatten()

    result = run_mcnemar(pnet_preds, baseline_preds, labels)

    print("\nMcNemar's Exact Test — PNet vs Baseline")
    print(f"α = {alpha}\n")
    print(f"{'Comparison':<30} {'b':>5} {'c':>5} {'p-value':>10} {'Significant':>12}")
    print("-" * 65)

    sig = "Yes" if result["p_value"] < alpha else "No"
    print(f"{'PNet vs Baseline':<30} {result['b']:>5} {result['c']:>5} "
          f"{result['p_value']:>10.4f} {sig:>12}")

    print(f"\nContingency table:\n{result['table']}")
    print()
    return result


def run_pnet_vs_gnn_mcnemar(results, labels, alpha = 0.05):
    """
    McNemar's exact test comparing PNet (averaged) against ReactomeGNN.
    Single comparison — no Bonferroni correction needed.
    """
    pnet_preds = results["PNet"]["preds"].numpy().flatten()
    gnn_preds  = results["ReactomeGNN"]["preds"].numpy().flatten()

    result = run_mcnemar(pnet_preds, gnn_preds, labels)

    print("\nMcNemar's Exact Test — PNet vs ReactomeGNN")
    print(f"α = {alpha}\n")
    print(f"{'Comparison':<30} {'b':>5} {'c':>5} {'p-value':>10} {'Significant':>12}")
    print("-" * 65)

    sig = "Yes" if result["p_value"] < alpha else "No"
    print(f"{'PNet vs ReactomeGNN':<30} {result['b']:>5} {result['c']:>5} "
          f"{result['p_value']:>10.4f} {sig:>12}")

    print(f"\nContingency table:\n{result['table']}")
    print()
    return result

def run_statistical_tests(labels):
    results = load_eval_results()
    run_pnet_layers_mcnemar_tests(results, labels)
    run_pnet_vs_baseline_mcnemar(results, labels)
    run_pnet_vs_gnn_mcnemar(results, labels)


if __name__ == "__main__":
    _, _, test_loader = run_data_pipeline()
    labels = torch.cat([y for _, y in test_loader]).numpy()
    run_statistical_tests(labels)