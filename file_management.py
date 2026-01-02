import os
import shutil

# Define the structure
folders = {
    "submission/images": [
        "chart_bias_privacy_correlation.png",
        "chart_k_sensitivity.png",
        "chart_optimization.png",
        "chart_pareto_cloud.png",
        "chart_price_of_progress.png",
        "tradeoff_chart_final.png",
        "tradeoff_chart.png",          # Added missing file
        "chart_bubble_tradeoff.png",   # Added missing file
        "chart_3d_knot.html",          # Added missing HTML visualization
        "chart_parallel_coords.html"   # Added missing HTML visualization
    ],
    "submission/scripts": [
        "analyze_advanced.py",
        "analyze_results.py",
        "check_price_progress.py",
        "check_results.py",
        "generate_all_visuals.py",
        "generate_pareto_cloud.py",
        "generate_extended_visuals.py", # Added missing script
        "visualize_final.py",           # Added visualization script
        "visualize_results.py"          # Added visualization script
    ],
    "submission/report": [
        "report.tex",
        "report.pdf" # If you have the PDF locally
    ]
}

# Create folders and move files
for folder, files in folders.items():
    os.makedirs(folder, exist_ok=True)
    for file in files:
        if os.path.exists(file):
            shutil.copy(file, os.path.join(folder, file))
            print(f"Copied {file} to {folder}")
        else:
            print(f"Warning: {file} not found (skipping)")

print("\n[x] Submission files organized into 'submission/' folder.")