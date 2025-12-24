# Compare.py - Seed Search Script
# Duplicates high-level logic from Compare.ipynb but iterates to find optimal seed.

import os
import sys
import numpy as np
import random
from pathlib import Path

# Ensure current directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import config
from config import N_EPISODES_APPRENTICE, NUM_APPRENTICES
from config import EXPERT_TRAJECTORIES_PATH
from irl_utils import set_seed, create_env, get_obs_shape
import td3_runner
import gail_runner

def run_experiment(seed):
    """Run full training and evaluation for a given seed."""
    print(f"\n{'#'*80}")
    print(f"Running Experiment with SEED = {seed}")
    print(f"{'#'*80}\n")

    # Set seed globally first (though runners will also set it)
    set_seed(seed)

    # --- Phase 1: Expert Training ---
    # We assume Expert is already trained or will be retrained regardless of seed search for this purpose.
    # To save time in search, we can reuse the existing expert if available, 
    # OR we can retrain. Since the prompt asks for "Apprentice 1-3 high accuracy", 
    # arguably we need a good expert first. 
    # However, retraining expert every loop is extremely expensive.
    # Let's assume the user wants to keep the current Expert (if good) and find a seed that works for Apprentices.
    # BUT, if seed controls environment reset, expert trajectories might need to be consistent?
    # Actually, if we fix the seed, we should ideally collect new expert trajectories with that seed.
    
    # For this search script, let's do the rigorous thing:
    # 1. Train/Load Expert (Assuming existing expert is 'good enough' to generate demos).
    # 2. Collect trajectories with the CURRENT seed (to ensure consistency with env reset).
    # 3. Train Apprentices.
    
    # Ensure Expert exists
    expert, _ = td3_runner.train_expert(n_episodes=config.N_EPISODES_EXPERT) 
    
    # --- Phase 2: Apprentice Training ---
    
    # Train TD3 Apprentices
    print("\n--- Training TD3 Apprentices ---")
    
    # Granular execution as requested
    env = create_env()
    obs_shape = get_obs_shape(env)
    
    # 1. Compute Expert Features
    print("Step 1: Computing Expert Features")
    expert_feature_expectation, _ = td3_runner.compute_expert_features(expert, env, m=config.FEATURE_EXPECTATION_EPISODES)
    
    # Initialize projection method variables
    feature_expectation = []
    feature_expectation_bar = []
    all_td3_results = []
    
    # 2. Train Apprentice 0
    print("Step 2: Training Apprentice 0")
    res0, feat0, w0, m0 = td3_runner.train_apprentice_0(env, obs_shape, seed=seed)
    all_td3_results.append(res0)
    feature_expectation.append(feat0)
    
    # 3. Train Apprentice 1-N
    for i in range(1, NUM_APPRENTICES):
        print(f"Step 3.{i}: Training Apprentice {i}")
        res, feat, w, m, new_bar, converged = td3_runner.train_apprentice_i(
            i, env, obs_shape, expert_feature_expectation, 
            feature_expectation, feature_expectation_bar, seed=seed
        )
        
        if new_bar is not None:
            feature_expectation_bar.append(new_bar)
            
        if converged:
            break
            
        all_td3_results.append(res)
        feature_expectation.append(feat)
    
    td3_results = all_td3_results
    
    # Run check for TD3 immediately
    print("\n--- Checking TD3 Success Rates ---")
    if not check_single_algo_success(td3_results, "TD3"):
        print("TD3 failed criteria. Skipping GAIL and retrying with new seed...")
        return td3_results, None # Return None for GAIL to signal skip
    
    # Train GAIL Apprentices
    print("\n--- Training GAIL Apprentices ---")
    
    # Granular execution for GAIL
    # Ensure expert trajectories exist
    gail_runner.ensure_expert_trajectories()
    
    all_gail_results = []
    
    # Train GAIL Apprentice 1-N
    for i in range(1, NUM_APPRENTICES):
        print(f"Step 4.{i}: Training GAIL Apprentice {i}")
        res = gail_runner.train_gail_apprentice_i(i, seed=seed)
        all_gail_results.append(res)
        
    gail_results = all_gail_results
    
    return td3_results, gail_results

def check_single_algo_success(results, algo_name):
    """
    Check if at least one Apprentice (1-3) achieves >= 90% success rate 
    (45/50) in any 50-episode chunk between episodes 400 and 500.
    """
    CHECK_INDICES = [8, 9]
    THRESH_COUNT = 45 # 90% of 50
    
    passed = False
    for res in results:
        app_id = res['id']
        if app_id == 0: continue # Skip Apprentice 0 for all checks
        
        successes = res['successes']
        current_max = 0
        
        valid_chunk_found = False
        for chunk_idx in CHECK_INDICES:
            start_ep = chunk_idx * 50
            end_ep = (chunk_idx + 1) * 50
            
            if len(successes) >= end_ep:
                chunk_data = successes[start_ep:end_ep]
                count = sum(chunk_data)
                print(f"DEBUG: Checking chunk {chunk_idx} (ep {start_ep}-{end_ep-1}): {count}/50") # DEBUG PRINT
                current_max = max(current_max, count)
                if count >= THRESH_COUNT:
                    passed = True
                valid_chunk_found = True
        
        if valid_chunk_found:
             rate = (current_max / 50.0) * 100
             print(f"{algo_name} Apprentice {app_id}: Max success in 400-500 ep: {current_max}/50 ({rate:.1f}%)")
        else:
             print(f"{algo_name} Apprentice {app_id}: Not enough episodes to check 400-500")
             
    return passed

def check_success_rates(td3_results, gail_results, threshold=90.0):
    """
    Check if at least one Apprentice (1-3) for BOTH TD3 and GAIL 
    achieves >= 90% success rate (45/50) in any 50-episode chunk 
    between episodes 400 and 500 (i.e. indices 8 and 9).
    """
    if gail_results is None:
        return False # TD3 failed earlier
        
    print(f"\n--- Checking Final Success Rates (Target: At least one >= 90% in chunk 400-500) ---")
    
    # Re-check TD3 (although already checked, good for logging/confirmation)
    td3_passed = check_single_algo_success(td3_results, "TD3")
    
    # Check GAIL
    gail_passed = check_single_algo_success(gail_results, "GAIL")
    
    if td3_passed and gail_passed:
        print("\nCRITERIA MET: Both TD3 and GAIL have at least one apprentice with >= 90% success in 400-500 ep.")
        return True
    else:
        print("\nCRITERIA FAILED.")
        return False


def update_config_seed(new_seed):
    """Update SEED in config.py."""
    config_path = Path(config.__file__)
    lines = config_path.read_text().splitlines()
    
    new_lines = []
    updated = False
    for line in lines:
        if line.startswith("SEED ="):
            new_lines.append(f"SEED = {new_seed}")
            updated = True
        else:
            new_lines.append(line)
            
    if not updated:
        # Append if not found
        new_lines.append(f"SEED = {new_seed}")
        
    config_path.write_text("\n".join(new_lines) + "\n")
    print(f"\nUpdated config.py with SEED = {new_seed}")

def main():
    # Loop until a good seed is found
    # Start with a random seed, or a specific list?
    # Random search is robust.
    
    # Using a deterministic start for the meta-search to be reproducible itself? 
    # User said "don't fix seed until high accuracy found".
    # So we iterate.
    
    attempt = 0
    # Success threshold
    THRESHOLD = 90.0 # User asked for "high accuracy", 90% is reasonable start.
    
    while True:
        attempt += 1
        # Generate a random 32-bit int seed
        seed = random.randint(0, 2**32 - 1)
        
        print(f"\n\n=== Seed Search Attempt {attempt} (Seed: {seed}) ===")
        
        # Delete expert trajectories to force regeneration with new seed
        if EXPERT_TRAJECTORIES_PATH.exists():
            try:
                os.remove(EXPERT_TRAJECTORIES_PATH)
                print(f"Deleted existing expert trajectories at {EXPERT_TRAJECTORIES_PATH}")
            except OSError as e:
                print(f"Error deleting expert trajectories: {e}")
        
        try:
            td3_res, gail_res = run_experiment(seed)
            
            if check_success_rates(td3_res, gail_res, threshold=THRESHOLD):
                print(f"\n\nSUCCESS! Found good seed: {seed}")
                update_config_seed(seed)
                break
            else:
                print(f"\nSeed {seed} failed criteria. Retrying...")
                
        except Exception as e:
            print(f"An error occurred during attempt {attempt}: {e}")
            # Keep trying other seeds? Or exit?
            # Ideally retry, maybe it was a transient numerical instability.
            import traceback
            traceback.print_exc()
            print("Retrying with new seed...")

if __name__ == "__main__":
    main()
