import json
import re
import os

# --- New Function Definitions (as strings to be injected) ---

# 1. Plot Individual Performance (for Apprentice 0)
PLOT_INDIVIDUAL_FUNC = r"""
def plot_individual_performance(agent_name, score_history, success_history, window_size=50):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Scores
    ax1 = axes[0]
    ax1.plot(score_history, alpha=0.6, label='Raw Score')
    if len(score_history) >= window_size:
        avg_scores = [np.mean(score_history[max(0, i-window_size):i+1]) for i in range(len(score_history))]
        ax1.plot(avg_scores, color='red', linewidth=2, label=f'Avg Score (w={window_size})')
    ax1.set_title(f"{agent_name} - Performance (Score)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Score")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Success Rate Moving Average
    ax2 = axes[1]
    if len(success_history) >= window_size:
        avg_success = [np.mean(success_history[max(0, i-window_size):i+1]) for i in range(len(success_history))]
        ax2.plot(avg_success, color='green', linewidth=2, label=f'Success Rate (w={window_size})')
    else:
        ax2.plot(success_history, alpha=0.5, label='Raw Success')
    ax2.set_title(f"{agent_name} - Moving Avg Success Rate")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Rate")
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Raster Plot
    ax3 = axes[2]
    # Create success events list for eventplot
    indices = [i for i, x in enumerate(success_history) if x >= 0.9] # Assuming 1 is success
    if indices:
        ax3.eventplot([indices], lineoffsets=[0], linelengths=0.8, colors=['blue'])
    ax3.set_title(f"{agent_name} - Binary Success Raster")
    ax3.set_xlabel("Episode")
    ax3.set_yticks([])
    ax3.set_xlim(0, len(success_history))
    ax3.grid(True, axis='x')

    plt.tight_layout()
    plt.show()
"""

# 2. Plot Comparative Dashboard (for Apprentice 1-10 vs Expert)
PLOT_COMPARATIVE_FUNC = r"""
def plot_comparative_dashboard(phase_name, expert_data, apprentices_data, window_size=50):
    # expert_data: {'name': 'Expert', 'scores': [], 'successes': []}
    # apprentices_data: list of {'name': 'Apprentice X', 'scores': [], 'successes': []}
    
    if not apprentices_data:
        print("No apprentice data to plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # --- Ax1: Performance (Scores) ---
    ax1 = axes[0]
    # Plot Expert
    if expert_data and expert_data.get('scores'):
        scores = expert_data['scores']
        # Smooth expert if long
        if len(scores) > window_size:
            smoothed = [np.mean(scores[max(0, i-window_size):i+1]) for i in range(len(scores))]
            ax1.plot(smoothed, color='black', linewidth=2, linestyle='--', label=f"Expert (Smoothed)")
        else:
            ax1.plot(scores, color='black', linewidth=2, linestyle='--', label=f"Expert")
    
    # Plot Apprentices
    cmap = plt.get_cmap('tab10')
    for i, app in enumerate(apprentices_data):
        scores = app['scores']
        if len(scores) > window_size:
            smoothed = [np.mean(scores[max(0, i-window_size):i+1]) for i in range(len(scores))]
            ax1.plot(smoothed, color=cmap(i % 10), label=app['name'])
        else:
            ax1.plot(scores, color=cmap(i % 10), alpha=0.7, label=app['name'])
            
    ax1.set_title(f"{phase_name}: Performance Comparison (Scores)")
    ax1.set_ylabel("Score")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)

    # --- Ax2: Moving Average Success Rate ---
    ax2 = axes[1]
    if expert_data and expert_data.get('successes'):
        succ = expert_data['successes']
        if len(succ) > window_size:
            smoothed = [np.mean(succ[max(0, i-window_size):i+1]) for i in range(len(succ))]
            ax2.plot(smoothed, color='black', linewidth=2, linestyle='--', label="Expert")
        else:
            # If short eval, plot simplified line or mean
            ax2.axhline(np.mean(succ), color='black', linestyle='--', label=f"Expert Mean ({np.mean(succ):.2f})")

    for i, app in enumerate(apprentices_data):
        succ = app['successes']
        if len(succ) > window_size:
            smoothed = [np.mean(succ[max(0, i-window_size):i+1]) for i in range(len(succ))]
            ax2.plot(smoothed, color=cmap(i % 10), label=app['name'])
        else:
             # Short eval - plot markers
             ax2.plot(succ, marker='.', linestyle='none', color=cmap(i % 10), alpha=0.3)
             ax2.axhline(np.mean(succ), color=cmap(i % 10), alpha=0.5, linestyle=':', label=f"{app['name']} Mean")

    ax2.set_title(f"{phase_name}: Success Rate Comparison")
    ax2.set_ylabel("Success Rate")
    ax2.set_ylim(-0.05, 1.05)
    # ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)

    # --- Ax3: Raster Plot (Binary Success Events) ---
    ax3 = axes[2]
    event_collection = []
    colors = []
    labels = []
    
    # Expert
    if expert_data and expert_data.get('successes'):
        indices = [idx for idx, val in enumerate(expert_data['successes']) if val >= 0.9]
        if indices:
            event_collection.append(indices)
            colors.append('black')
            labels.append('Expert')
            
    # Apprentices
    for i, app in enumerate(apprentices_data):
        succ = app['successes']
        indices = [idx for idx, val in enumerate(succ) if val >= 0.9]
        if indices:
            event_collection.append(indices)
            colors.append(cmap(i % 10))
            labels.append(app['name'])
    
    if event_collection:
        ax3.eventplot(event_collection, lineoffsets=range(len(event_collection)), 
                      linelengths=0.7, colors=colors)
        ax3.set_yticks(range(len(labels)))
        ax3.set_yticklabels(labels)
    
    ax3.set_title(f"{phase_name}: Raster Plot (Success Events)")
    ax3.set_xlabel("Episode")
    ax3.grid(True, axis='x')

    plt.tight_layout()
    plt.show()
"""

# 3. New Evaluate Agent Function (Returns Histories)
EVALUATE_AGENT_FUNC = r"""
def evaluate_agent(agent, env, episodes=50, steps=200):
    print(f"Evaluating agent for {episodes} episodes...")
    returns = []
    successes = []
    
    for _ in range(episodes):
        ep_ret, success = agent.test_model(env=env, steps=steps, render_save_path=None) # Ensure no render
        returns.append(ep_ret)
        successes.append(1 if success else 0)
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    success_rate = np.mean(successes) * 100
    
    return mean_return, std_return, success_rate, successes, returns
"""

# --- Notebook Processors ---

def update_evaluate_agent_cell(cell_source):
    if "def evaluate_agent" in cell_source and "eval_succ_hist" not in cell_source:
        return EVALUATE_AGENT_FUNC
    return cell_source

def update_plotting_cell(cell_source):
    if "def plot_intermediate_results" in cell_source:
        return PLOT_INDIVIDUAL_FUNC + "\n" + PLOT_COMPARATIVE_FUNC
    return cell_source

def update_irl_loop(cell_source):
    if "def projection_method_algorithm" in cell_source:
        # Check if already updated (avoid double replace)
        if "eval_score_hist" in cell_source:
             return cell_source
             
        new_source = cell_source.replace(
            "eval_mean, eval_std, eval_sr = evaluate_agent(apprentice, env, episodes=50, steps=200)",
            "eval_mean, eval_std, eval_sr, eval_succ_hist, eval_score_hist = evaluate_agent(apprentice, env, episodes=50, steps=200)"
        )
        new_source = new_source.replace(
            '"noise_free_success_rate": eval_sr,',
            '"noise_free_success_rate": eval_sr, "eval_success_history": eval_succ_hist, "eval_score_history": eval_score_hist,'
        )
        
        plot_block_old = r"""        try:
            plot_intermediate_results(expert_history, all_results, n_episodes=n_episodes)
        except Exception as e:
            print(f"Plotting failed: {e}")"""
            
        plot_block_new = r"""        # --- Updated Plotting Logic ---
        try:
            # 1. Pot Individual for Apprentice 0
            if i == 0:
                print("Plotting Individual Result for Apprentice 0...")
                plot_individual_performance("Apprentice 0 (Training)", success_hist, score_history)
            
            # 2. Accumulate Comparative Data (Apprentice 1+)
            else:
                app_train_list = []
                app_eval_list = []
                
                for r in all_results:
                    if r['id'] == 0: continue
                    app_train_list.append({
                        'name': f"Apprentice {r['id']}",
                        'scores': r.get('score_history', []), 
                        'successes': r['success_history']
                    })
                    app_eval_list.append({
                        'name': f"Apprentice {r['id']}",
                        'scores': r.get('eval_score_history', []),
                        'successes': r.get('eval_success_history', [])
                    })
                
                plot_comparative_dashboard("Training Phase (Apprentice 1+ vs Expert)", 
                                         {'name': 'Expert', 'scores': [], 'successes': []},
                                         app_train_list)
                                         
                plot_comparative_dashboard("Evaluation Phase (Apprentice 1+ vs Expert)", 
                                         {'name': 'Expert', 'scores': [], 'successes': expert_history}, 
                                         app_eval_list)

        except Exception as e:
            print(f"Plotting failed: {e}")
            import traceback
            traceback.print_exc()"""

        new_source = new_source.replace(
            "success_hist, avg_success_history = apprentice.irl_train(",
            "score_hist, avg_score_hist, success_hist, avg_success_history = apprentice.irl_train("
        )
        new_source = new_source.replace(
            '"success_history": success_hist,',
            '"success_history": success_hist, "score_history": score_hist,'
        )
        new_source = new_source.replace(plot_block_old, plot_block_new)
        return new_source
    return cell_source

def update_gail_loop(cell_source):
    if "for run_idx in range(max_runs):" in cell_source:
         # Check if already updated
        if "eval_score_hist" in cell_source:
             return cell_source

        new_source = cell_source.replace(
            "eval_mean, eval_std, eval_sr = evaluate_agent(agent, env, episodes=50, steps=200)",
            "eval_mean, eval_std, eval_sr, eval_succ_hist, eval_score_hist = evaluate_agent(agent, env, episodes=50, steps=200)"
        )
        new_source = new_source.replace(
            '"noise_free_success_rate": eval_sr,',
            '"noise_free_success_rate": eval_sr, "eval_success_history": eval_succ_hist, "eval_score_history": eval_score_hist,'
        )
        
        # CORRECTED SEARCH STRING FOR GAIL PLOTTING BLOCK
        # Note: Be very careful with whitespace.
        plot_block_old_gail = r"""    # Live Plotting
    try:
        plot_intermediate_results(expert_success_history, all_results, n_episodes=n_train_episodes)
    except Exception as e:
        print(f"Plotting failed: {e}")"""
            
        plot_block_new_gail = r"""    # Live Plotting
    try:
        # 1. Individual for Apprentice 0
        if run_idx == 0:
             if 'score_history' in locals():
                 plot_individual_performance("Apprentice 0", score_history, success_history)
             else:
                 plot_individual_performance("Apprentice 0", [], success_history)

        # 2. Comparative
        else:
            app_train_list = []
            app_eval_list = []
            for r in all_results:
                if r['id'] == 0: continue
                
                app_train_list.append({
                    'name': f"Apprentice {r['id']}",
                    'scores': r.get('score_history', []), 
                    'successes': r.get('success_history', [])
                })
                app_eval_list.append({
                    'name': f"Apprentice {r['id']}",
                    'scores': r.get('eval_score_history', []),
                    'successes': r.get('eval_success_history', [])
                })
            
            # Plot Training - Empty/Static Expert for GAIL
            plot_comparative_dashboard("Training Phase", None, app_train_list)
            
            # Plot Eval - Use expert_success_history
            expert_eval_data = {'name': 'Expert', 'scores': [], 'successes': expert_success_history} if 'expert_success_history' in locals() else None
            plot_comparative_dashboard("Evaluation Phase", expert_eval_data, app_eval_list)

    except Exception as e:
        print(f"GAIL Plotting failed: {e}")
        import traceback
        traceback.print_exc()"""

        new_source = new_source.replace(plot_block_old_gail, plot_block_new_gail)
        return new_source
    return cell_source

def modify_notebook(nb_path, is_irl=True):
    print(f"Processing {nb_path}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    
    for cell in cells:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # Disable Rendering globally
            if "render_save_path=" in source and "None" not in source:
                source = re.sub(r"render_save_path\s*=\s*['\"].*?['\"]", "render_save_path=None", source)
            
            # Update Functions
            source = update_evaluate_agent_cell(source)
            source = update_plotting_cell(source)
            
            # Update Loops
            if is_irl:
                source = update_irl_loop(source)
            else:
                source = update_gail_loop(source)
            
            cell['source'] = [line + "\n" for line in source.splitlines()]
            # Fix double newlines locally
            cell['source'] = [l.replace('\n\n', '\n') for l in cell['source']]

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Saved {nb_path}")

if __name__ == "__main__":
    # modify_notebook('/home/shimoiyusuke/manipulator-inverse-rl/IRL/Panda_Reach_v3_IRL.ipynb', is_irl=True)
    modify_notebook('/home/shimoiyusuke/manipulator-inverse-rl/GAIL/Panda_Reach_v3_GAIL.ipynb', is_irl=False)
