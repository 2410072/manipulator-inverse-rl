import json
import os

notebook_path = 'TD3/Panda_Reach_v3_TD3_IRL.ipynb'

# Code blocks to insert/replace

# 1. Helper functions for analysis (Chunking, Metrics)
analysis_helpers_code = """
import numpy as np
import matplotlib.pyplot as plt

def calculate_chunked_stats(history, chunk_size=50):
    chunks = [history[i:i + chunk_size] for i in range(0, len(history), chunk_size)]
    stats = []
    for i, chunk in enumerate(chunks):
        success_count = np.sum(chunk)
        failure_count = len(chunk) - success_count
        success_rate = success_count / len(chunk)
        stats.append({
            "chunk_idx": i,
            "start_episode": i * chunk_size,
            "end_episode": i * chunk_size + len(chunk),
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_rate
        })
    return stats

def find_fastest_stable_model(all_results, stability_threshold=0.9, window=50):
    best_model_id = -1
    min_episodes_to_stable = float('inf')
    
    for res in all_results:
        # Calculate moving average to determine stability
        success_hist = res['success_history']
        moving_avg = np.convolve(success_hist, np.ones(window)/window, mode='valid')
        
        # Find first index where moving avg >= threshold and stays there (simplified: just first crossing)
        # For stricter stability, check if it drops below threshold later.
        stable_indices = np.where(moving_avg >= stability_threshold)[0]
        
        if len(stable_indices) > 0:
            first_stable = stable_indices[0]
            if first_stable < min_episodes_to_stable:
                min_episodes_to_stable = first_stable
                best_model_id = res['id']
                
    return best_model_id, min_episodes_to_stable

def find_closest_to_expert(all_results, expert_history):
    best_model_id = -1
    min_distance = float('inf')
    
    expert_curve = np.array(expert_history)
    
    for res in all_results:
        model_curve = np.array(res['success_history'])
        # Truncate to matching length if needed
        min_len = min(len(expert_curve), len(model_curve))
        dist = np.mean((model_curve[:min_len] - expert_curve[:min_len])**2)
        
        if dist < min_distance:
            min_distance = dist
            best_model_id = res['id']
            
    return best_model_id, min_distance
"""

# 2. Expert Evaluation Block (Before training loop)
# Checks if expert model exists, then evaluates it to get baseline history.
expert_eval_code = """
# Expert Evaluation Baseline
expert_success_history = []
expert_avg_success_history = []

expert_model_path = '../TD3/Models/Expert/'
if os.path.exists(expert_model_path):
    print("Loading Expert model for baseline comparison...")
    # Initialize Expert Agent (assuming TD3Trainer can load it)
    # We use same config as apprentice roughly, but load weights
    expert_agent = TD3Trainer(env=env, input_dims=obs_shape, agent_name='Expert_Baseline', 
                              model_load_path=expert_model_path, 
                              # Disable exploration noise or reduce it for expert eval if needed, 
                              # though original CollectExpert used select_action which handles trained flag
                              noise_factor=0.0) 
    expert_agent.is_trained = True # Ensure strict exploitation
    
    # Run evaluation for n_episodes (same as apprentice training length to compare curves)
    # Note: 'test_model' runs 1 episode.
    print(f"Evaluating Expert for {n_episodes} episodes...")
    for _ in range(n_episodes):
        _, success = expert_agent.test_model(env=env, steps=300) # steps limit?
        expert_success_history.append(success)
        expert_avg_success_history.append(np.mean(expert_success_history[-100:]))
        
    print(f"Expert Evaluation Complete. Avg Success: {expert_avg_success_history[-1]:.2f}")
else:
    print(f"Warning: Expert model not found at {expert_model_path}. Comparison will be skipped.")
    expert_success_history = [0.0] * n_episodes # Dummy filler
"""

# 3. Modified Training Loop
# Replaces the simple loop with one that collects data
training_loop_replacement = """
    all_results = []

    for i in range(max_runs):
        print(f"--- Training Apprentice Model {i} ---")
        apprentice = TD3Trainer(env=env, input_dims=obs_shape, agent_name=f'Apprentice_{i}', 
                                model_save_path=f'./Models/Apprentice_{i}/', 
                                exploration_period=exploration_period)
        
        # Train and capture histories
        # Note: td3_train returns (score_history, avg_score_history, success_history, avg_success_history)
        # make sure td3_train signature matches this unpacking!
        score_hist, avg_score_hist, success_hist, avg_success_hist = apprentice.td3_train(
            n_episodes=n_episodes, 
            opt_steps=opt_steps, 
            reward_weights=w,
            print_every=print_every
        )
        
        # Chunked Analysis for this model
        chunk_stats = calculate_chunked_stats(success_hist)
        total_success = np.sum(success_hist)
        total_rate = total_success / len(success_hist)
        
        print(f"Model {i} Finished. Total Success Rate: {total_rate*100:.1f}%")
        
        all_results.append({
            "id": i,
            "score_history": score_hist,
            "avg_score_history": avg_score_hist,
            "success_history": success_hist,
            "avg_success_history": avg_success_hist,
            "chunk_stats": chunk_stats,
            "total_success_rate": total_rate
        })
"""

# 4. Final Analysis & Plotting Block
final_analysis_code = """
# --- Final Comparative Analysis ---

# 1. Identify Notable Models
best_model_id, _ = find_fastest_stable_model(all_results)
closest_model_id, dist = find_closest_to_expert(all_results, expert_success_history)

print(f"Most Fast & Stable Model ID: {best_model_id}")
print(f"Model Closest to Expert ID: {closest_model_id} (MSE: {dist:.4f})")

# 2. Unified Plotting
plt.figure(figsize=(12, 6))

# Plot Expert Baseline
if any(expert_avg_success_history):
    plt.plot(expert_avg_success_history, label='Expert Baseline', color='black', linestyle='--', linewidth=2)

# Plot All Apprentices
for res in all_results:
    mid = res['id']
    label = f'Model {mid}'
    
    # Highlight special models
    style = '-'
    width = 1
    if mid == best_model_id:
        label += ' (Fastest Stable)'
        width = 2.5
    if mid == closest_model_id:
        label += ' (Closest to Expert)'
        style = ':'
        width = 2.5
        
    plt.plot(res['avg_success_history'], label=label, linestyle=style, linewidth=width, alpha=0.8)

plt.title("Comparison of Model Success Rates vs Expert")
plt.xlabel("Episode")
plt.ylabel("Average Success Rate")
plt.ylim(0, 1.05) # Unified Scale
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Print 50-Episode Chunk Stats for Best Model
if best_model_id != -1:
    print(f"\\nDetailed Stats for Best Model (ID {best_model_id}):")
    best_res = next(r for r in all_results if r['id'] == best_model_id)
    print(f"{'Chunk':<10} {'Range':<15} {'Success':<10} {'Fail':<10} {'Rate':<10}")
    print("-" * 60)
    for stat in best_res['chunk_stats']:
        range_str = f"{stat['start_episode']}-{stat['end_episode']}"
        print(f"{stat['chunk_idx']:<10} {range_str:<15} {stat['success_count']:<10} {stat['failure_count']:<10} {stat['success_rate']:.2%}")
    print("-" * 60)
    print(f"Overall Success Rate: {best_res['total_success_rate']:.2%}")
"""

def modify_notebook():
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        cells = nb['cells']
        new_cells = []
        
        # Insert Helper Functions at the beginning (after imports)
        # We'll put it after the first cell which usually has imports
        helpers_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": analysis_helpers_code.splitlines(keepends=True)
        }
        
        # Locate the main algorithm cell
        algo_cell_idx = -1
        for i, cell in enumerate(cells):
            if cell['cell_type'] == 'code' and 'def projection_method_algorithm' in "".join(cell['source']):
                algo_cell_idx = i
                break
        
        if algo_cell_idx == -1:
            print("Could not find projection_method_algorithm cell.")
            return

        # Modify the algo cell to replace the loop
        algo_cell = cells[algo_cell_idx]
        source_lines = algo_cell['source']
        
        # Find where to inject Expert Eval (before the loop)
        # We look for "max_runs = 10" or similar context, or just put it at start of function
        # But wait, this is inside a function 'projection_method_algorithm'.
        # We need to insert the Expert Eval logic code INSIDE the function, before the loop.
        # And replacements for the loop itself.
        
        # Converting raw string codes to list of lines with indentation
        def indent_lines(text, spaces=4):
            return [" " * spaces + line + "\\n" for line in text.strip().splitlines()]

        new_source = []
        loop_replaced = False
        
        for line in source_lines:
            # Detect start of apprentice loop
            if 'for i in range(max_runs):' in line and not loop_replaced:
                # Insert Expert Eval code before this loop
                new_source.extend(indent_lines(expert_eval_code))
                new_source.append("\\n")
                
                # Insert New Training Loop
                new_source.extend(indent_lines(training_loop_replacement))
                loop_replaced = True 
                
                # We skip lines until we see the end of the original loop or indentation change
                # But typically we can just consume lines until we find what we want to keep?
                # Actually, simpler approach: The original loop in this notebook (from previous view) was:
                #     for i in range(max_runs):
                #         apprentice = TD3Trainer(...)
                #         score_hist... = apprentice.td3_train(...)
                # We are replacing this WHOLE block.
                # So we need to skip the original lines corresponding to this loop.
                # However, iterating line by line is tricky if we don't know exact content.
                # Let's rely on the fact that we replace 'for i in range(max_runs):' and subsequent indented lines.
            
            elif loop_replaced:
                # We are now skipping the old loop body.
                # We stop skipping when indentation drops back to function level (4 spaces) or less?
                # The loop body has 8 spaces.
                # We need to be careful not to skip 'return ...' or other things after the loop.
                if len(line) - len(line.lstrip()) < 8 and line.strip() != "":
                    # Indentation decreased, loop ended. 
                    # Insert Final Analysis code here (inside function, before return)
                    # Or maybe return values change?
                    # The function returns w usually.
                    
                    # Wait, if we want to print analysis and plot, we should probably do it here.
                    new_source.extend(indent_lines(final_analysis_code))
                    new_source.append(line) # Add the current line (e.g. return w)
                    loop_replaced = False # Stop skipping, back to normal copying
                else:
                    # Inside old loop, skip it
                    pass
            else:
                new_source.append(line)
                
        algo_cell['source'] = new_source
        
        # Inject the helper cell before the algo cell
        cells.insert(algo_cell_idx, helpers_cell)
        
        # Save
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            
        print("Successfully updated notebook with analysis logic.")

    except Exception as e:
        print(f"Error updating notebook: {e}")

if __name__ == "__main__":
    modify_notebook()
