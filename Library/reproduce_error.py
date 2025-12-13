import sys
import os
import traceback
sys.path.append(os.getcwd())

from config import EXPERT_TRAJECTORIES_PATH, EXPERT_MODEL_PATH

try:
    print("Importing runners...")
    from td3_runner import train_expert
    from IRL_lib.airl_runner import train_airl
    
    print("Running train_airl(n_episodes=2)...")
    # Need expert trajectories first?
    if not EXPERT_TRAJECTORIES_PATH.exists():
        # Fake it or fail?
        print("Trajectory file missing, cannot run AIRL test.")
    else:
        train_airl(n_episodes=2)
        print("train_airl successful.")

except Exception:
    traceback.print_exc()
