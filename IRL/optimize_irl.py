import json
import os

NOTEBOOK_PATH = "/home/shimoiyusuke/manipulator-inverse-rl/IRL/Panda_Reach_v3_IRL.ipynb"

# The logic inside the function is identical to previous plan
NEW_FUNCTION_CODE = """def projection_method_algorithm(env, expert_history=None, epsilon=0.001, m=2000, n_episodes=500,
                                opt_steps=50, batch_size=1024, exploration_period=100, save_render=True, fps=5, print_every=50):
    
    feature_expectation = []
    feature_expectation_bar = []
    weights = []
    margins = []
    rewards = []
    success_rates = []
    all_results = [] 
    
    obs_shape = env.observation_space['observation'].shape[0] + \\
            env.observation_space['achieved_goal'].shape[0] + \\
            env.observation_space['desired_goal'].shape[0]

    max_runs = 15
    
    for i in range(max_runs):
        print(f"\\n-------------------------------------- Iteration: {i} --------------------------------------\\n")
        
        # Warm Start Logic: Load model from previous apprentice if available
        # i=0: Apprentice 1 (Fresh)
        # i=1: Apprentice 2 (Load Apprentice 1)
        load_path = f'./Models/Apprentice {i}/' if i > 0 else None
        
        # Reduce exploration for transfer learning to exploit prior knowledge
        current_exploration = exploration_period if i == 0 else int(exploration_period * 0.5)
        
        # Instantiate IRLTrainer with optimized batch_size for GPU usage
        apprentice = IRLTrainer(env=env, input_dims=obs_shape, agent_name=f'Apprentice {i+1}', 
                                model_save_path=f'./Models/Apprentice {i+1}/', 
                                model_load_path=load_path,
                                exploration_period=current_exploration,
                                batch_size=batch_size)
                                
        if i > 0 and load_path:
             print(f"Warm Start: Loaded model from {load_path}")

        # Step 1: Initialization / First Run
        if i == 0:
            observation, info = env.reset()
            if isinstance(observation, dict):
                state = np.concatenate((observation['observation'], observation['achieved_goal'], observation['desired_goal']))
            else:
                state = observation
            sample_feature = torch.tensor(state, dtype=torch.float32)
            w_0 = torch.randn(sample_feature.size(0), 1).div_(torch.randn(1).norm())
            weights.append(w_0)
            
            score_hist, avg_score_hist, success_hist, avg_success_hist = apprentice.irl_train(
                n_episodes=n_episodes, opt_steps=opt_steps, 
                reward_weights=w_0, print_every=print_every, 
                plot_save_path=f'../Results/IRL/Apprentice_{i+1}_Performance.png'
            )
            apprentice_feature_expectation, apprentice_reward = compute_average_feature(agent=apprentice, m=m)
            rewards.append(apprentice_reward)
            margins.append(1.0)
            feature_expectation.append(apprentice_feature_expectation)
            # print("Expert Feature Expectation:", expert_feature_expectation) 
            
        else:
            # Step 2: IRL Projection
            if i == 1:
                feature_expectation_bar.append(feature_expectation[i - 1])
                weights.append((expert_feature_expectation - feature_expectation[i - 1]).view(-1, 1))
                margins.append((expert_feature_expectation - feature_expectation_bar[i - 1]).norm().item())
            else:
                A = feature_expectation_bar[i - 2]
                B = feature_expectation[i - 1] - A
                C = expert_feature_expectation - A
                numerator = (B.view(-1, 1).t() @ C.view(-1, 1))
                denominator = (B.view(-1, 1).t() @ B.view(-1, 1))
                if denominator == 0: denominator = 1e-8
                feature_expectation_bar.append(A + (numerator / denominator) * B)
                weight = (expert_feature_expectation - feature_expectation_bar[i - 1]).view(-1, 1)
                margin = (expert_feature_expectation - feature_expectation_bar[i - 1]).norm().item()
                weights.append(weight)
                margins.append(margin)
            print(f"Margin: {margins[i]}")
            if margins[i] <= epsilon:
                print(f"Converged with margin {margins[i]} <= {epsilon}")
                break

            score_hist, avg_score_hist, success_hist, avg_success_hist = apprentice.irl_train(
                n_episodes=n_episodes, opt_steps=opt_steps, 
                reward_weights=weights[-1], print_every=print_every, 
                plot_save_path=f'../Results/IRL/Apprentice_{i+1}_Performance.png'
            )
            apprentice.save_model()
            
            if save_render:
                try:
                    apprentice.test_model(steps=1000, render_save_path=f'../Results/IRL/Apprentice {i+1} Policy')
                except Exception as e:
                    print(f"Rendering failed: {e}")

            apprentice_feature_expectation, apprentice_reward = compute_average_feature(agent=apprentice, m=m)
            rewards.append(apprentice_reward)
            feature_expectation.append(apprentice_feature_expectation)
            
        # --- Live Plotting & Results Collection ---
        final_sr = np.mean(success_hist[-100:]) * 100 if len(success_hist) >= 100 else 0.0
        success_rates.append(final_sr)
        res = {
            "id": i+1,
            "success_history": success_hist,
            "avg_success_history": avg_success_hist,
            "chunk_stats": calculate_chunked_stats(success_hist, chunk_size=50)
        }
        all_results.append(res)
        try:
            plot_intermediate_results(expert_history, all_results, n_episodes=n_episodes)
        except Exception as e:
            print(f"Plotting failed: {e}")
        print(f"Apprentice {i+1} Final Avg Success Rate: {final_sr:.1f}%")
        print(f"\\nDetailed Stats for Apprentice {i+1}:")
        stats = calculate_chunked_stats(success_hist, chunk_size=50)
        print(f"{'Chunk':<6} {'Range':<15} {'Success':<10} {'Fail':<10} {'Rate':<10}")
        print("-" * 60)
        for s in stats:
            range_str = f"{s['start_episode']}-{s['end_episode']}"
            print(f"{s['chunk_idx']:<6} {range_str:<15} {s['success_count']:<10} {s['failure_count']:<10} {s['success_rate']*100:>6.2f}%")
        print("")
        
        # --- NEW: Post-Training Evaluation (Noise-Free) ---
        print(f"Running Noise-Free Evaluation for Apprentice {i+1}...")
        eval_mean, eval_std, eval_sr = evaluate_agent(apprentice, env, episodes=50, steps=200)
        print(f"Apprentice {i+1} Evaluation: Success Rate {eval_sr:.1f}%, Return {eval_mean:.1f}")
        
    return rewards, margins, success_rates"""

def optimze_notebook():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    found = False
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            # Check source (list of strings)
            source_text = "".join(cell['source'])
            if "def projection_method_algorithm" in source_text:
                print("Found projection_method_algorithm cell. Updating...")
                # Update source as a list of lines (split by \n for cleanliness, or just list of one string)
                # Jupyter standard is list of strings, usually one per line including \n
                cell['source'] = [line + '\n' for line in NEW_FUNCTION_CODE.split('\n')]
                # Remove last \n from last line if needed, but it's fine.
                found = True
                break
    
    if found:
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Notebook updated successfully.")
    else:
        print("Error: Could not find the function definition cell.")

if __name__ == "__main__":
    optimze_notebook()
