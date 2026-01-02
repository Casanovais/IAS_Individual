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
