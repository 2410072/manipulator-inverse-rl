import json
import os

NOTEBOOK_PATH = "/home/shimoiyusuke/manipulator-inverse-rl/GAIL/Panda_Reach_v3_GAIL.ipynb"

EVAL_FUNC_CODE = """def evaluate_agent(agent, env, episodes=10, steps=200):
    \"\"\"
    Evaluate agent performance (Mean Return & Success Rate) without exploration noise.
    \"\"\"
    returns = []
    successes = []
    
    for _ in range(episodes):
        ep_ret, success = agent.test_model(env=env, steps=steps, render_save_path=None)
        returns.append(ep_ret)
        successes.append(1 if success else 0)
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    success_rate = np.mean(successes) * 100
    
    return mean_return, std_return, success_rate"""

NEW_LOOP_CODE = """# Auto-train GAIL agents for a fixed number of runs
max_runs = 15
all_results = []

start_run_idx = 0
n_train_episodes = 500 # Aligned with user request

for run_idx in range(start_run_idx, max_runs + 1):
    print(f"\\n-------------------------------------- Iteration: {run_idx} --------------------------------------\\n")
    
    # Warm Start Logic: Load model from previous apprentice if available
    # run_idx=0: Apprentice 0 (Fresh - though typically we might want to name it Apprentice 1 to match IRL, but sticking to 0-base if user used it previously, 
    # BUT wait, IRL uses 1-based indexing in display. The user code here was 0-based. I will stick to existing 0-based variable 'run_idx' but maybe label it consistently?)
    # Existing code used 'Apprentice {run_idx}'.
    
    load_path = f"./Models/Apprentice {run_idx-1}/" if run_idx > 0 else None
    
    # Update kwargs for this run
    current_kwargs = gail_agent_kwargs.copy()
    current_kwargs['batch_size'] = 1024 # GPU Optimization
    current_kwargs['model_load_path'] = load_path
    
    # Reduce exploration if transfer learning
    if run_idx > 0:
        base_expl = current_kwargs.get('exploration_period', 100)
        current_kwargs['exploration_period'] = int(base_expl * 0.5)
        print(f"Warm Start: Loading model from {load_path}")

    # Initialize new agent
    agent = GAILTrainer(**current_kwargs)
    agent.agent_name = f"Apprentice {run_idx}"
    agent.model_save_path = f"./Models/Apprentice {run_idx}/"
    
    # Train (Increased opt_steps for speed)
    score_history, avg_score_history, success_history, avg_success_history = agent.gail_train(
        n_episodes=n_train_episodes,
        opt_steps=50, # Optimized
        print_every=50, 
        plot_save_path=f'../Results/GAIL/Apprentice_{run_idx}_Performance.png'
    )
    
    # Store results
    res = {
        "id": run_idx,
        "success_history": success_history,
        "avg_success_history": avg_success_history,
        "chunk_stats": calculate_chunked_stats(success_history, chunk_size=50)
    }
    all_results.append(res)
    
    # Live Plotting
    try:
        plot_intermediate_results(expert_success_history, all_results, n_episodes=n_train_episodes)
    except Exception as e:
        print(f"Plotting failed: {e}")
        
    # Save Model
    agent.save_model()
    
    # Detailed Stats
    chunk_stats = calculate_chunked_stats(success_history, chunk_size=50)
    print(f"\\nDetailed Stats for Apprentice {run_idx}:")
    print(f"{'Chunk':<6} {'Range':<15} {'Success':<10} {'Fail':<10} {'Rate':<10}")
    print("-" * 60)
    for s in chunk_stats:
        range_str = f"{s['start_episode']}-{s['end_episode']}"
        print(f"{s['chunk_idx']:<6} {range_str:<15} {s['success_count']:<10} {s['failure_count']:<10} {s['success_rate']*100:>6.2f}%")
        
    # --- Post-Training Evaluation (Noise-Free) ---
    print(f"Running Noise-Free Evaluation for Apprentice {run_idx}...")
    eval_mean, eval_std, eval_sr = evaluate_agent(agent, env, episodes=50, steps=200)
    print(f"Apprentice {run_idx} Evaluation: Success Rate {eval_sr:.1f}%, Return {eval_mean:.1f}")
    print(\"\")"""

def optimize_notebook():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    loop_cell_idx = -1
    for idx, cell in enumerate(nb['cells']):
        if cell.get('id') == 'gail_training_loop':
            loop_cell_idx = idx
            break
            
    if loop_cell_idx != -1:
        print("Found training loop cell.")
        
        # 1. Update training loop source
        nb['cells'][loop_cell_idx]['source'] = [line + '\n' for line in NEW_LOOP_CODE.split('\n')]
        
        # 2. Insert evaluate_agent function cell BEFORE loop
        # Check if already exists to avoid dupes (simple check)
        prev_cell = nb['cells'][loop_cell_idx - 1]
        if "def evaluate_agent" not in "".join(prev_cell['source']):
            new_cell = {
                "cell_type": "code",
                "execution_count": None,
                "id": "evaluate_agent_func",
                "metadata": {},
                "outputs": [],
                "source": [line + '\n' for line in EVAL_FUNC_CODE.split('\n')]
            }
            nb['cells'].insert(loop_cell_idx, new_cell)
            print("Inserted evaluate_agent cell.")
        else:
            print("evaluate_agent cell likely already exists, skipping insertion.")
            
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Notebook updated successfully.")
    else:
        print("Error: Could not find cell with id 'gail_training_loop'.")

if __name__ == "__main__":
    optimize_notebook()
