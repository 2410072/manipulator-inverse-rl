# config.py - Common configuration for Compare experiments

import sys
from pathlib import Path

# Environment
ENV_NAME = "PandaReach-v3"

# Training parameters
# Main experiment parameters
N_EPISODES_EXPERT = 500  # default: 500
N_EPISODES_APPRENTICE = 500  # default: 500
N_EPISODES_APPRENTICE_0 = 100  # default: 100
OPT_STEPS = 10  # default: 64
BATCH_SIZE = 256
EXPLORATION_PERIOD = 300  # For Apprentices (Matches upstream notebook)
EXPLORATION_PERIOD_EXPERT = 300  # For Expert training (Matches TD3 notebook)
PRINT_EVERY = 50

# Number of apprentices (0 to 3 = 4 total)
NUM_APPRENTICES = 4

# TD3 Hyperparameters
ALPHA = 0.001
BETA = 0.002
GAMMA = 0.99
TAU = 0.05
REPLAY_SIZE = 1000000
NOISE_FACTOR = 0.1
UPDATE_ACTOR_EVERY = 2

# IRL Parameters
EPSILON = 0.001

# Evaluation
# EVAL_EPISODES removed (using N_EPISODES_... instead)
EXPERT_CHECK_STEPS = 100      # For Expert evaluation
FEATURE_CALC_STEPS = 1000     # For Feature Expectation and Apprentice evaluation

# Expert-specific evaluation (post-training)
EXPERT_EVAL_EPISODES = 500

# Feature expectation computation
FEATURE_EXPECTATION_EPISODES = 1000
FEATURE_EXPECTATION_EPISODES_APPRENTICE = 1000  # Matches upstream notebook (m=1000)

# Plotting
PLOT_WINDOW_SIZE = 50

# Paths
COMPARE_DIR = Path(__file__).resolve().parent
MODELS_DIR = COMPARE_DIR / "Models"
RESULTS_DIR = COMPARE_DIR / "Results"
EXPERT_MODEL_PATH = MODELS_DIR / "Expert"

# TD3-specific paths
TD3_MODELS_DIR = MODELS_DIR / "TD3"
TD3_RESULTS_DIR = RESULTS_DIR / "TD3"

# Expert path within Compare
EXPERT_MODEL_PATH = MODELS_DIR / "Expert"

# GAIL-specific paths
GAIL_MODELS_DIR = MODELS_DIR / "GAIL"
GAIL_RESULTS_DIR = RESULTS_DIR / "GAIL"
EXPERT_TRAJECTORIES_PATH = COMPARE_DIR / "expert_trajectories.pt"

# Random Seed
SEED = 3060856897
