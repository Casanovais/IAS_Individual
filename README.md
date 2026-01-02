# The Price of Progress: Empirical Analysis of Responsible AI

> **"Can we achieve privacy, fairness, and utility simultaneously, or is it a zero-sum game?"**

## 📖 Overview
This repository contains the source code, experimental data, and analysis scripts for the project **"The Price of Progress: An Empirical Analysis of the Performance-Fairness-Privacy Trade-off in AI."**

This study empirically evaluates the **"Three-Way Knot"**: the tension between maintaining high model accuracy (**Utility**), ensuring non-discriminatory outcomes (**Fairness**), and preventing re-identification of original records (**Privacy**).

By systematically applying the `PrivateSMOTE` algorithm across four benchmark datasets (Adult, German Credit, Bank Marketing, Heart Disease), we quantify the cost of ethical compliance in Machine Learning.

---

## 📂 Project Structure

| Directory / File | Description |
| :--- | :--- |
| **`dataprep/`** | Scripts for cleaning and preprocessing the raw datasets (One-Hot Encoding, Scaling). |
| **`modeling/`** | Core logic for training Random Forest models and evaluating fairness (Equalized Odds). |
| **`transformations/`** | Implementation of the `PrivateSMOTE` algorithm and differential privacy noise injection. |
| **`submission/`** | Contains the final report, generated visualizations, and analysis scripts. |
| **`MASTER_RESULTS.csv`** | The raw experimental results used for the final analysis. |

---

## 🚀 How to Run

### 1. Prerequisites
Ensure you have **Python 3.10+** installed. Install dependencies using `pip`:

```bash
# Recommended: Use a virtual environment
pip install pandas seaborn matplotlib numpy scikit-learn fairlearn
```


2. Reproduce the Analysis

To generate the visualizations and statistics presented in the report, run the main analysis script. This script consolidates all data processing and plotting logic:
Bash

python submission/scripts/generate_all_visuals.py

Output: This will generate the following high-resolution charts in your current directory:

    📊 chart_k_sensitivity.png: Impact of k-neighbors on privacy risk.

    ⚖️ chart_price_of_progress.png: Bar chart visualizing Accuracy Loss vs. Fairness Gain.

    🔗 chart_bias_privacy_correlation.png: The "Three-Way Knot" correlation plot.

    ☁️ chart_pareto_cloud.png: Detailed trade-off scatter plot for the Adult dataset.

    ⏱️ chart_optimization.png: Runtime comparison (Naive vs. Hash-Based Audit).

3. Verify Analytical Claims

To see the exact numbers quoted in the report (e.g., "35% relative fairness gain"), run the verification scripts:
Bash

# Checks specific percentage changes in Fairness/Accuracy
python submission/scripts/check_price_progress.py

# Generates the full statistical summary table (Mean/Std Dev)
python submission/scripts/check_results.py

🔍 Key Findings

    🔗 The "Three-Way Knot" Exists We found a strong negative correlation between Privacy and Fairness (r<0). Stricter anonymization often degrades the model's ability to detect and correct bias.

    💸 Privacy is Expensive On the Heart Disease dataset, privacy noise reduced utility by ~8% without improving fairness, highlighting the fragility of small medical datasets.

    🎁 The "Free Lunch" Anomaly The Bank Marketing dataset showed that privacy noise can sometimes act as a regularizer, improving fairness by 65% with negligible utility loss.

    🔒 The Imbalance Trap For the German Credit dataset, privacy mechanisms failed to fix bias (Fairness stuck at 1.00), effectively "locking in" discrimination due to the Curse of Dimensionality.

📜 References & Credits

This project builds upon the foundational work of Carvalho et al. (EPIA 2023).

    Original Paper: A Three-Way Knot: Privacy, Fairness, and Predictive Performance Dynamics

    Original Code: GitHub Repository

    Author: Rafael Casanovais

    Institution: University of Porto


### What I improved:
1.  **Header:** Added a clear H1 title.
2.  **Quote:** Formatted the opening question as a Markdown blockquote (`>`) to make it stand out.
3.  **Table:** Converted the corrupted directory list into a proper Markdown table.
4.  **Code Blocks:** Put all commands inside `bash` code blocks for easy copying and reading.
5.  **Formatting:** Used bolding and inline code (`backticks`) to highlight file names, technical terms, and datasets.
6.  **Icons:** Added relevant emojis (📊, ⚖️, 🔗) to the "Outputs" and "Key Findings" to make the text scannable and visually appealing.

Would you like me to help you generate a `requirements.txt` file based on the imports listed in