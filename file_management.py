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
        "tradeoff_chart.png",
        "chart_bubble_tradeoff.png",
        "chart_3d_knot.html",
        "chart_parallel_coords.html"
    ],
    "submission/scripts": [
        "analyze_advanced.py",
        "analyze_results.py",
        "check_price_progress.py",
        "check_results.py",
        "generate_all_visuals.py",  # This now contains the Pareto Cloud logic
        "legacy_visualizations.py", # Contains original visualization logic
        "visualize_advanced.py"     # Contains HTML visualization logic
    ],
    "submission/report": [
        "report.tex",
        "report.pdf" 
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