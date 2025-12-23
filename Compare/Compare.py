# %% [1]
# Imports
import config
import td3_runner
import gail_runner
from video_recorder import global_recorder
from plotting import plot_comparative_dashboard, plot_cross_algorithm_comparison
import contextlib

@contextlib.contextmanager
def record_process(name):
    """Context manager to record video during a process using internal gym rendering."""
    # Internal robot motion recording
    video_dir = config.ROBOT_MOTION_DIR
    video_path = video_dir / f"{name}.webm"
    
    # Overview (Full Screen) - DISABLED due to environment limitation (no ffmpeg/mss)
    # overview_dir = config.OVERVIEW_DIR 
    # To enable overview, we would need a robust screen capture tool here.
    
    # Ensure directories exist
    video_dir.mkdir(parents=True, exist_ok=True)
    # overview_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Start recording (Internal): {video_path}")
    global_recorder.start(str(video_path), fps=1)
    try:
        yield
    finally:
        global_recorder.stop()
        print(f"Saved recording to: {video_path}")

# %% [2]
# Train TD3 Expert
# Train TD3 Expert
with record_process("td3_expert_train"):
    td3_expert, td3_expert_train_data = td3_runner.train_expert()

# %% [3]
# Evaluate TD3 Expert
# Evaluate TD3 Expert
with record_process("td3_expert_eval"):
    td3_expert, td3_expert_eval_data = td3_runner.evaluate_expert(expert=td3_expert)

# %% [4]
# Train TD3 Apprentices 0-3 using projection method
# 1. Compute Expert Feature Expectation
with record_process("td3_expert_features"):
    expert_feature_expectation, expert_mean_reward = td3_runner.compute_expert_features(td3_expert)

# %% [4]
# 2. Train Apprentice 0
with record_process("td3_apprentice_0_train"):
    apprentice_0_result, w_0, margin_0, feature_expectation_0 = td3_runner.train_apprentice_0()

# %% [5]
# 3. Train Remaining Apprentices (1-3)
# Initialize lists for projection method
feature_expectation = [feature_expectation_0]
feature_expectation_bar = []
weights = [w_0]
margins = [margin_0]
td3_apprentice_train_data = [apprentice_0_result]

# %% [5.1]
# Train TD3 Apprentice 1
with record_process("td3_apprentice_1_train"):
    res1, fe1, w1, m1, bar1 = td3_runner.train_single_apprentice(
        1, expert_feature_expectation, feature_expectation, feature_expectation_bar, weights, margins
    )
    if bar1 is not None: feature_expectation_bar.append(bar1)
    feature_expectation.append(fe1)
    weights.append(w1)
    margins.append(m1)
    if res1 is not None:
        td3_apprentice_train_data.append(res1)

# %% [5.2]
# Train TD3 Apprentice 2
with record_process("td3_apprentice_2_train"):
    res2, fe2, w2, m2, bar2 = td3_runner.train_single_apprentice(
        2, expert_feature_expectation, feature_expectation, feature_expectation_bar, weights, margins
    )
    if bar2 is not None: feature_expectation_bar.append(bar2)
    feature_expectation.append(fe2)
    weights.append(w2)
    margins.append(m2)
    if res2 is not None:
        td3_apprentice_train_data.append(res2)

# %% [5.3]
# Train TD3 Apprentice 3
with record_process("td3_apprentice_3_train"):
    res3, fe3, w3, m3, bar3 = td3_runner.train_single_apprentice(
        3, expert_feature_expectation, feature_expectation, feature_expectation_bar, weights, margins
    )
    if bar3 is not None: feature_expectation_bar.append(bar3)
    feature_expectation.append(fe3)
    weights.append(w3)
    margins.append(m3)
    if res3 is not None:
        td3_apprentice_train_data.append(res3)

# No need to combine explicitly as we appended to td3_apprentice_train_data

# %% [5]
# Evaluate TD3 Apprentices
# Evaluate TD3 Apprentices
with record_process("td3_apprentices_eval"):
    td3_apprentice_eval_data = td3_runner.evaluate_apprentices()

# %% [6]
import compare_utils
# Plot TD3 comparisons
# Filter out Apprentice 0
filtered_td3_train_data = compare_utils.filter_apprentice_data(td3_apprentice_train_data, "TD3_Apprentice_0")
filtered_td3_eval_data = compare_utils.filter_apprentice_data(td3_apprentice_eval_data, "TD3_Apprentice_0")

td3_runner.plot_all_comparisons(
    td3_expert_train_data,
    td3_expert_eval_data,
    filtered_td3_train_data,
    filtered_td3_eval_data
)

# %% [7]
# %% [7]
# Train GAIL Apprentices 1-3
gail_apprentice_train_data = []

# %% [7.1]
# Train GAIL Apprentice 1
with record_process("gail_apprentice_1_train"):
    res_g1 = gail_runner.train_single_apprentice(1)
    gail_apprentice_train_data.append(res_g1)

# %% [7.2]
# Train GAIL Apprentice 2
with record_process("gail_apprentice_2_train"):
    res_g2 = gail_runner.train_single_apprentice(2)
    gail_apprentice_train_data.append(res_g2)

# %% [7.3]
# Train GAIL Apprentice 3
with record_process("gail_apprentice_3_train"):
    res_g3 = gail_runner.train_single_apprentice(3)
    gail_apprentice_train_data.append(res_g3)

# %% [8]
# Evaluate GAIL Apprentices
# Evaluate GAIL Apprentices
with record_process("gail_apprentices_eval"):
    gail_apprentice_eval_data = gail_runner.evaluate_apprentices()

# %% [9]
# Plot GAIL comparisons
gail_runner.plot_all_comparisons(
    gail_apprentice_train_data,
    gail_apprentice_eval_data
)

# %% [10]
# Compare TD3 and GAIL Apprentices
from config import TD3_RESULTS_DIR, RESULTS_DIR
import compare_utils

# Combine evaluation data
expert_data, all_apprentices = compare_utils.prepare_final_comparison_data(
    td3_expert_eval_data,
    td3_apprentice_eval_data,
    gail_apprentice_eval_data
)

# %% [11]
plot_comparative_dashboard(
    "TD3 vs GAIL Evaluation Comparison",
    expert_data,
    all_apprentices,
    save_path=str(TD3_RESULTS_DIR.parent / "TD3_vs_GAIL_Comparison.png"))

# %% [None]
# Cross-algorithm comparison: TD3 vs GAIL for each Apprentice (1, 2, 3)
# Filter out Apprentice 0 from TD3 train data for cross-algorithm comparison
filtered_td3_apprentice_train_data = compare_utils.filter_apprentice_data(td3_apprentice_train_data, 'TD3_Apprentice_0')

plot_cross_algorithm_comparison(
    filtered_td3_apprentice_train_data,
    gail_apprentice_train_data,
    save_path=str(RESULTS_DIR / "TD3_vs_GAIL_Training_Apprentice_Comparison.png")
)

# %% [1]
# Cross-algorithm comparison: TD3 vs GAIL for each Apprentice (1, 2, 3)
# Filter out Apprentice 0 from TD3 train data for cross-algorithm comparison
filtered_td3_apprentice_eval_data = compare_utils.filter_apprentice_data(td3_apprentice_eval_data, 'TD3_Apprentice_0')

plot_cross_algorithm_comparison(
    filtered_td3_apprentice_eval_data,
    gail_apprentice_eval_data,
    save_path=str(RESULTS_DIR / "TD3_vs_GAIL_Evaluation_Apprentice_Comparison.png")
)

