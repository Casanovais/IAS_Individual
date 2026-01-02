The Price of Progress: Empirical Analysis of Responsible AI

This repository contains the source code, experimental data, and analysis scripts for the project "The Price of Progress: An Empirical Analysis of the Performance-Fairness-Privacy Trade-off in AI."

Project Structure

dataprep/: Scripts for cleaning and preprocessing the raw datasets.

modeling/: Core logic for training Random Forest models and evaluating fairness (Equalized Odds).

transformations/: Implementation of the PrivateSMOTE algorithm.

submission/: Contains the final report, generated visualizations, and analysis scripts.

MASTER_RESULTS.csv: The raw experimental results used for analysis.

How to Run

1. Prerequisites

Ensure you have Python 3.10+ installed. Install dependencies using Pipenv or pip:

pip install pandas seaborn matplotlib numpy scikit-learn fairlearn


2. Reproduce the Analysis

To generate the visualizations and statistics presented in the report, run the main analysis script:

python submission/scripts/generate_all_visuals.py


This will generate the following charts in your current directory:

chart_k_sensitivity.png: Impact of $k$-neighbors on privacy.

chart_price_of_progress.png: Accuracy loss vs. Fairness gain.

chart_bias_privacy_correlation.png: The "Three-Way Knot" correlation.

chart_pareto_cloud.png: Detailed trade-off analysis for the Adult dataset.

3. Verify Analytical Claims

To see the exact numbers quoted in the report (e.g., "35% fairness gain"), run the verification scripts:

python submission/scripts/check_price_progress.py
python submission/scripts/check_results.py


Key Findings

The "Three-Way Knot" Exists: We found a strong negative correlation between Privacy and Fairness ($r < 0$).

Privacy is Expensive: On the Heart Disease dataset, privacy noise reduced utility by ~8% without improving fairness.

The "Free Lunch": The Bank Marketing dataset showed that privacy noise can sometimes act as a regularizer, improving fairness by 65% with negligible utility loss.

References

This project builds upon the work of Carvalho et al. (EPIA 2023).

Original Paper: A Three-Way Knot: Privacy, Fairness, and Predictive Performance Dynamics

Original Code: GitHub Repository