
import sys
import os
import shutil
from pathlib import Path

# Add current dir to sys.path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'Compare'))

from Compare.plotting import plot_apprentice_comparison, plot_cross_algorithm_comparison

# Dummy data for validation
dummy_data_td3 = [
    {'id': 1, 'name': 'Apprentice 1', 'scores': [1, 2, 3], 'successes': [0, 1, 0]},
    {'id': 2, 'name': 'Apprentice 2', 'scores': [2, 3, 4], 'successes': [0.5, 0.5, 0.5]},
    {'id': 3, 'name': 'Apprentice 3', 'scores': [3, 4, 5], 'successes': [1, 1, 1]},
]

dummy_data_gail = [
    {'id': 1, 'name': 'Apprentice 1', 'scores': [2, 3, 4], 'successes': [0.2, 0.2, 0.2]},
    {'id': 2, 'name': 'Apprentice 2', 'scores': [3, 4, 5], 'successes': [0.6, 0.6, 0.6]},
    {'id': 3, 'name': 'Apprentice 3', 'scores': [4, 5, 6], 'successes': [0.9, 1, 0.9]},
]

# Create a temporary directory for output
output_dir = Path('./dummy_plotting_test')
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir()

print("Testing plot_apprentice_comparison...")
plot_apprentice_comparison("TD3", dummy_data_td3, save_dir=output_dir)

print("Testing plot_cross_algorithm_comparison...")
plot_cross_algorithm_comparison(dummy_data_td3, dummy_data_gail, save_dir=output_dir)

# Check if files were created
expected_files = [
    "TD3_Apprentice_Comparison_Performance.png",
    "TD3_Apprentice_Comparison_Success.png",
    "TD3_Apprentice_Comparison_Raster.png",
    "TD3_vs_GAIL_Comparison_Performance.png",
    "TD3_vs_GAIL_Comparison_Success.png",
    "TD3_vs_GAIL_Comparison_Raster.png",
]

all_exist = True
for f in expected_files:
    if not (output_dir / f).exists():
        print(f"Error: {f} was not created.")
        all_exist = False
    else:
        print(f"Confirmed: {f} created.")

if all_exist:
    print("\nSUCCESS: All expected plot files generated.")
else:
    print("\nFAILURE: Some plot files missing.")
