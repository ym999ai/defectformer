"""
Generates a formatted ablation results table from results.json.

Usage:
    python ablation/report.py --results ./ablation_results/results.json
    python ablation/report.py --results ./ablation_results/results.json --markdown
    python ablation/report.py --results ./ablation_results/results.json --latex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ablation.configs import ABLATION_EXPERIMENTS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v, paper=None, bold_if_better=False):
    """Format a metric value; append (paper) and delta in parentheses."""
    if v is None:
        return "N/A"
    s = f"{v:.1f}"
    if paper is not None:
        delta = v - paper
        sign  = "+" if delta >= 0 else ""
        s    += f" ({sign}{delta:.1f})"
    return s


def _delta_row(exp_name: str, results: dict) -> str | None:
    """Compute Δ AP vs. the cumulative previous ablation config."""
    order = [e.name for e in ABLATION_EXPERIMENTS]
    try:
        idx = order.index(exp_name)
    except ValueError:
        return None
    if idx == 0 or order[idx - 1] not in results:
        return None
    prev_ap = results[order[idx - 1]].get("AP")
    curr_ap = results[exp_name].get("AP")
    if prev_ap is None or curr_ap is None:
        return None
    delta = curr_ap - prev_ap
    return f"{'+' if delta >= 0 else ''}{delta:.1f}"


# ─────────────────────────────────────────────────────────────────────────────
# Plain text
# ─────────────────────────────────────────────────────────────────────────────

def print_plain(results: dict):
    order = [e.name for e in ABLATION_EXPERIMENTS
             if e.name in results and "error" not in results[e.name]]

    w = 44
    header = (f"  {'Configuration':<{w}}  {'AP':>6}  {'AP_S':>6}  "
              f"{'AP50':>6}  {'AP75':>6}  {'mIoU':>6}  "
              f"{'ΔAP':>6}  {'Paper':>6}")
    print("\n" + "=" * (len(header) + 2))
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name in order:
        r     = results[name]
        delta = _delta_row(name, results) or "  ---"
        paper_str = f"{r['paper_AP']:.1f}" if r.get("paper_AP") else "  ---"
        ap_s  = f"{r['AP_S']:.1f}" if r.get("AP_S") is not None else "  N/A"

        print(f"  {r['label']:<{w}}  {r['AP']:6.1f}  {ap_s:>6}  "
              f"{r['AP50']:6.1f}  {r['AP75']:6.1f}  {r['mIoU']:6.1f}  "
              f"{delta:>6}  {paper_str:>6}")

    print("=" * (len(header) + 2))


# ─────────────────────────────────────────────────────────────────────────────
# Markdown
# ─────────────────────────────────────────────────────────────────────────────

def print_markdown(results: dict):
    order = [e.name for e in ABLATION_EXPERIMENTS
             if e.name in results and "error" not in results[e.name]]

    print("| Configuration | AP | AP_S | AP50 | AP75 | mIoU | ΔAP | Paper AP |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")

    for name in order:
        r     = results[name]
        delta = _delta_row(name, results) or "---"
        paper = f"{r['paper_AP']:.1f}" if r.get("paper_AP") else "---"
        ap_s  = f"{r['AP_S']:.1f}" if r.get("AP_S") is not None else "N/A"
        lbl   = r["label"].replace("↳", "→")

        print(f"| {lbl} | **{r['AP']:.1f}** | {ap_s} | "
              f"{r['AP50']:.1f} | {r['AP75']:.1f} | {r['mIoU']:.1f} | "
              f"{delta} | {paper} |")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX
# ─────────────────────────────────────────────────────────────────────────────

def print_latex(results: dict):
    order = [e.name for e in ABLATION_EXPERIMENTS
             if e.name in results and "error" not in results[e.name]]

    lines = [
        r"\begin{table}[h]",
        r"\renewcommand{\arraystretch}{1.25}",
        r"\centering",
        r"\caption{Ablation Study on MVTec AD (reproduced).}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{AP} & \textbf{AP$_S$} "
        r"& \textbf{AP$_{50}$} & \textbf{AP$_{75}$} & $\bm{\Delta}$\textbf{AP} \\",
        r"\midrule",
    ]

    for name in order:
        r     = results[name]
        delta = _delta_row(name, results) or "---"
        ap_s  = f"{r['AP_S']:.1f}" if r.get("AP_S") is not None else "N/A"
        lbl   = r["label"].replace("↳", r"\hookrightarrow")
        lines.append(
            f"{lbl} & {r['AP']:.1f} & {ap_s} & "
            f"{r['AP50']:.1f} & {r['AP75']:.1f} & {delta} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Per-experiment timing summary
# ─────────────────────────────────────────────────────────────────────────────

def print_timing(results: dict):
    total = 0.0
    print("\nTraining time per experiment:")
    for name, r in results.items():
        t = r.get("train_min")
        if t:
            print(f"  {r.get('label', name):<48}  {t:6.1f} min")
            total += t
    if total:
        print(f"  {'TOTAL':<48}  {total:6.1f} min  ({total/60:.1f} h)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results",  required=True,
                   help="Path to results.json produced by run_ablation.py")
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--latex",    action="store_true")
    p.add_argument("--timing",   action="store_true")
    args = p.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    if args.markdown:
        print_markdown(results)
    elif args.latex:
        print_latex(results)
    else:
        print_plain(results)

    if args.timing:
        print_timing(results)


if __name__ == "__main__":
    main()
