
print("DEBUG: Script Start", flush=True)
import sys
import os
import traceback

try:
    print("DEBUG: Importing standard libs", flush=True)
    from pathlib import Path
    
    print("DEBUG: Importing matplotlib", flush=True)
    import matplotlib.pyplot as plt

    # Add current directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    print("DEBUG: Importing config", flush=True)
    from config import (
        TD3_MODELS_DIR, GAIL_MODELS_DIR, RESULTS_DIR, 
        BATCH_SIZE, FEATURE_CALC_STEPS, NUM_APPRENTICES
    )
    
    print("DEBUG: Importing irl_utils", flush=True)
    from irl_utils import create_env, get_obs_shape, evaluate_agent
    
    print("DEBUG: Importing Algos", flush=True)
    from td3_algo import TD3Trainer
    from gail_algo import GAILTrainer
    
    print("DEBUG: Importing plotting", flush=True)
    from plotting import plot_cross_algorithm_comparison

    # Hardcode 50 episodes for quick generation (User wants "Comparison" graphs)
    EVAL_EPISODES = 50 

    def generate_plots():
        print(f"Generating Evaluation Comparison Plots over {EVAL_EPISODES} episodes...", flush=True)
        env = create_env()
        obs_shape = get_obs_shape(env)
        
        td3_results = []
        gail_results = []
        
        # Evaluate Apprentices 1, 2, 3
        for i in range(1, NUM_APPRENTICES): # 1, 2, 3
            # --- TD3 ---
            td3_path = TD3_MODELS_DIR / "Apprentices" / f"Apprentice_{i}"
            if (td3_path / "actor.pth").exists():
                print(f"Evaluating TD3 Apprentice {i}...", flush=True)
                td3_agent = TD3Trainer(
                    env=env, input_dims=obs_shape, agent_name=f'Apprentice_{i}',
                    model_load_path=str(td3_path) + "/"
                )
                res = evaluate_agent(td3_agent, env, episodes=EVAL_EPISODES, steps=FEATURE_CALC_STEPS)
                td3_results.append({
                    'id': i,
                    'name': f'TD3_Apprentice_{i}',
                    'scores': res['returns'],
                    'successes': res['successes']
                })
            else:
                print(f"TD3 Apprentice {i} model missing.", flush=True)

            # --- GAIL ---
            gail_path = GAIL_MODELS_DIR / f"Apprentice_{i}"
            if (gail_path / "actor.pth").exists():
                print(f"Evaluating GAIL Apprentice {i}...", flush=True)
                gail_agent = GAILTrainer(
                    env=env, input_dims=obs_shape, agent_name=f'GAIL_Apprentice_{i}',
                    model_load_path=str(gail_path) + "/",
                    expert_loader=None 
                )
                res = evaluate_agent(gail_agent, env, episodes=EVAL_EPISODES, steps=FEATURE_CALC_STEPS)
                gail_results.append({
                    'id': i,
                    'name': f'GAIL_Apprentice_{i}',
                    'scores': res['returns'],
                    'successes': res['successes']
                })
            else:
                print(f"GAIL Apprentice {i} model missing.", flush=True)

        # Plot
        print("Plotting results...", flush=True)
        plot_cross_algorithm_comparison(
            td3_results,
            gail_results,
            window_size=10, 
            save_path=RESULTS_DIR / "Evaluation_Diff.png", 
            phase_name="Evaluation"
        )
        print("Done.", flush=True)

except Exception as e:
    print(f"ERROR: {e}", flush=True)
    traceback.print_exc()

if __name__ == "__main__":
    generate_plots()
