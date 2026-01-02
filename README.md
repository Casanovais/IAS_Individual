# The Price of Progress  
## An Empirical Analysis of the Performance–Fairness–Privacy Trade-off in Responsible AI

> “Can we achieve privacy, fairness, and utility simultaneously — or is ethical AI a zero-sum game?”

---

## Overview

This repository contains the full experimental pipeline for the project:

**The Price of Progress: An Empirical Analysis of the Performance–Fairness–Privacy Trade-off in AI**

The study empirically investigates the **“Three-Way Knot”**: the tension between

- **Utility** — maintaining high predictive performance  
- **Fairness** — ensuring non-discriminatory outcomes (measured via *Equalized Odds*)  
- **Privacy** — preventing re-identification of original records  

Using the `PrivateSMOTE` algorithm, we systematically apply privacy-preserving transformations across four benchmark datasets:

- Adult Income  
- German Credit  
- Bank Marketing  
- Heart Disease  

The goal is to quantify the **cost of ethical compliance** in machine learning systems and identify when privacy and fairness objectives reinforce—or undermine—each other.

---

## Repository Structure

| Path | Description |
|-----|-------------|
| `dataprep/` | Data cleaning and preprocessing scripts (encoding, scaling). |
| `modeling/` | Model training (Random Forest) and fairness evaluation logic (Equalized Odds). |
| `transformations/` | Implementation of `PrivateSMOTE` and differential privacy noise injection. |
| `submission/` | Final report, analysis scripts, and figure generation utilities. |
| `MASTER_RESULTS.csv` | Consolidated raw experimental results used in all analyses. |

---

## Reproducibility

### 1. Requirements

- Python 3.10 or higher  
- Recommended: use a virtual environment

Install dependencies:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn fairlearn

2. Reproduce All Figures

To regenerate every visualization and statistical result reported in the paper, run:

python submission/scripts/generate_all_visuals.py

This script generates the following high-resolution figures in the working directory:

    chart_k_sensitivity.png
    Impact of k-neighbors on privacy risk

    chart_price_of_progress.png
    Accuracy loss versus fairness gain

    chart_bias_privacy_correlation.png
    Correlation analysis of the Three-Way Knot

    chart_pareto_cloud.png
    Pareto trade-off scatter plot (Adult dataset)

    chart_optimization.png
    Runtime comparison (naïve vs. hash-based audit)

3. Verify Analytical Claims

To reproduce the exact numerical claims cited in the report (e.g., relative fairness gains or accuracy drops), run:

# Verifies relative changes in fairness and utility
python submission/scripts/check_price_progress.py

# Generates the full statistical summary table (mean / standard deviation)
python submission/scripts/check_results.py

Key Findings
The Three-Way Knot Exists

A strong negative correlation between privacy and fairness is observed (r < 0). Stricter anonymization often degrades the model’s ability to detect and correct bias.
Privacy Is Expensive

On the Heart Disease dataset, privacy noise reduced predictive utility by approximately 8 percent without improving fairness, highlighting the fragility of small medical datasets.
The “Free Lunch” Anomaly

On the Bank Marketing dataset, privacy noise occasionally acted as a regularizer, improving fairness by approximately 65 percent with negligible loss in accuracy.
The Imbalance Trap

For the German Credit dataset, privacy mechanisms failed to mitigate bias (fairness remained at 1.00), effectively locking in discrimination due to class imbalance and high dimensionality.
References

This project builds upon the work of Carvalho et al. (EPIA 2023):

    Carvalho et al., A Three-Way Knot: Privacy, Fairness, and Predictive Performance Dynamics

    Original codebase: https://github.com/tmcarvalho/three-way-knot

Author

Rafael Santos Novais
University of Porto
