# td3_runner.py - TD3 Expert and Apprentice training/evaluation

import numpy as np
import torch
from pathlib import Path
import sys
import os

# Ensure current directory is in sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import config and utils
from config import (
    N_EPISODES_EXPERT, N_EPISODES_APPRENTICE, N_EPISODES_APPRENTICE_0, OPT_STEPS, BATCH_SIZE,
    EXPLORATION_PERIOD, EXPLORATION_PERIOD_EXPERT, PRINT_EVERY, NUM_APPRENTICES,
    EXPERT_CHECK_STEPS, FEATURE_CALC_STEPS,
    EXPERT_MODEL_PATH, TD3_MODELS_DIR, TD3_RESULTS_DIR,
    EXPERT_EVAL_EPISODES, FEATURE_EXPECTATION_EPISODES, FEATURE_EXPECTATION_EPISODES_APPRENTICE,
    ALPHA, BETA, GAMMA, TAU, REPLAY_SIZE, NOISE_FACTOR, UPDATE_ACTOR_EVERY, EPSILON
)
from irl_utils import (
    create_env, get_obs_shape, compute_average_feature,
    print_chunked_stats, evaluate_agent
)
from irl_algo import solve_projection_method
from plotting import plot_individual_performance, plot_comparative_dashboard

# Import TD3Trainer
from td3_algo import TD3Trainer


def train_expert(n_episodes=None, force_retrain=False):
    """
    Train the TD3 Expert agent.
    Returns training history and evaluation results.
    """
    n_episodes = n_episodes or N_EPISODES_EXPERT
    
    # Create directories
    TD3_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TD3_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    expert_path = EXPERT_MODEL_PATH
    expert_path.mkdir(parents=True, exist_ok=True)
    
    env = create_env()
    obs_shape = get_obs_shape(env)
    
    # Always retrain expert as requested
    if (expert_path / "actor.pth").exists():
        print(f"Expert model found at {expert_path}, but forcing retrain...")

    
    print("Training TD3 Expert...") # Create Expert Agent
    expert = TD3Trainer(
        env=env,
        input_dims=obs_shape,
        agent_name='Expert',
        model_save_path=str(expert_path) + "/",
        exploration_period=EXPLORATION_PERIOD_EXPERT,
        alpha=ALPHA, beta=BETA, gamma=GAMMA, tau=TAU,
        batch_size=BATCH_SIZE, replay_size=REPLAY_SIZE,
        update_actor_every=UPDATE_ACTOR_EVERY, noise_factor=NOISE_FACTOR
    )
    
    score_history, avg_score_history, success_history, avg_success_history = expert.td3_train(
        n_episodes=n_episodes,
        opt_steps=OPT_STEPS,
        print_every=PRINT_EVERY,
        plot_save_path=str(TD3_RESULTS_DIR / "TD3_Expert_Performance.png")
    )
    
    expert.save_model()
    print(f"Expert model saved to {expert_path}")
    
    # Print chunked stats
    print_chunked_stats("TD3 Expert", success_history)
    
    # Plot individual performance
    plot_individual_performance(
        "TD3 Expert",
        score_history, success_history,
        save_path=str(TD3_RESULTS_DIR / "TD3_Expert_Individual.png")
    )
    
    training_data = {
        'scores': score_history,
        'successes': success_history,
        'avg_scores': avg_score_history,
        'avg_successes': avg_success_history
    }
    
    return expert, training_data


def evaluate_expert(expert=None, episodes=None):
    """Evaluate the TD3 Expert agent."""
    env = create_env()
    obs_shape = get_obs_shape(env)
    
    if expert is None:
        expert_path = EXPERT_MODEL_PATH
        expert = TD3Trainer(
            env=env, input_dims=obs_shape, agent_name='Expert',
            model_load_path=str(expert_path) + "/"
        )
    
    if episodes is None:
        episodes = EXPERT_EVAL_EPISODES

    print(f"Evaluating TD3 Expert over {episodes} episodes...")
    results = evaluate_agent(expert, env, episodes=episodes, steps=EXPERT_CHECK_STEPS)
    
    success_count = int(np.sum(results['successes']))
    total_episodes = len(results['successes'])
    print(f"TD3 Expert Evaluation: Mean Return = {results['mean_return']:.3f}, "
          f"Std = {results['std_return']:.3f}, Success: {success_count}/{total_episodes} ({results['success_rate']:.1f}%)")
    
    print_chunked_stats("TD3 Expert (Eval)", results['successes'])

    # Plot evaluation performance
    plot_individual_performance(
        "TD3 Expert (Eval)",
        results['returns'], results['successes'],
        save_path=str(TD3_RESULTS_DIR / "TD3_Expert_Evaluation.png")
    )
    
    return expert, results


def compute_expert_features(expert, m=None):
    """
    Compute Expert's feature expectation.
    Returns: expert_feature_expectation, expert_mean_reward
    """
    env = create_env()
    
    m_expert = m
    if m_expert is None:
        m_expert = FEATURE_EXPECTATION_EPISODES
        
    print(f"Computing Expert feature expectation with {m_expert} episodes...")
    expert_feature_expectation, expert_mean_reward = compute_average_feature(
        expert, m=m_expert, steps=FEATURE_CALC_STEPS, env=env
    )
    return expert_feature_expectation, expert_mean_reward


def train_apprentice_0(m=None):
    """
    Train Apprentice 0 (Random Weights).
    Returns: result dictionary, w_0, margin_0, feature_expectation_0
    """
    env = create_env()
    obs_shape = get_obs_shape(env)
    
    print(f"\n{'='*70}")
    print(f"  TD3 Apprentice 0 Training")
    print(f"{'='*70}\n")
    
    # Create apprentice model directory
    apprentice_base_path = TD3_MODELS_DIR / "Apprentices"
    apprentice_base_path.mkdir(parents=True, exist_ok=True)
    
    # Initial: Random weights
    weights = []
    margins = []
    
    observation, info = env.reset()
    state = np.concatenate((
        observation['observation'],
        observation['achieved_goal'],
        observation['desired_goal']
    ))
    sample_feature = torch.tensor(state, dtype=torch.float32)
    w_0 = torch.randn(sample_feature.size(0), 1).div_(torch.randn(1).norm())
    weights.append(w_0)
    margins.append(1.0)
    
    # Save path
    i = 0
    iter_save_path = apprentice_base_path / f"Apprentice_{i}"
    iter_save_path.mkdir(parents=True, exist_ok=True)

    # Initialize Apprentice 0
    apprentice = TD3Trainer(
        env=env, input_dims=obs_shape, agent_name=f'Apprentice_{i}',
        model_save_path=str(iter_save_path) + "/",
        exploration_period=EXPLORATION_PERIOD,
        batch_size=BATCH_SIZE, replay_size=REPLAY_SIZE,
        alpha=ALPHA, beta=BETA, gamma=GAMMA, tau=TAU,
        update_actor_every=UPDATE_ACTOR_EVERY, noise_factor=NOISE_FACTOR
    )
    
    # Train
    score_hist, avg_score_hist, success_hist, avg_success_hist = apprentice.td3_train(
        n_episodes=N_EPISODES_APPRENTICE_0,
        opt_steps=OPT_STEPS,
        reward_weights=w_0,
        print_every=PRINT_EVERY,
        plot_save_path=str(TD3_RESULTS_DIR / f"TD3_Apprentice_{i}_Performance.png")
    )
    
    # Save model
    apprentice.save_model()
    
    # Compute feature expectation
    m_apprentice = m
    if m_apprentice is None:
            m_apprentice = FEATURE_EXPECTATION_EPISODES_APPRENTICE
    app_feature, app_reward = compute_average_feature(
        apprentice, m=m_apprentice, steps=FEATURE_CALC_STEPS
    )
    
    # Print chunked stats
    print_chunked_stats(f"TD3 Apprentice {i}", success_hist)
    
    # Plot individual performance
    plot_individual_performance(
        f"TD3 Apprentice {i}",
        score_hist, success_hist,
        save_path=str(TD3_RESULTS_DIR / f"TD3_Apprentice_{i}_Individual.png")
    )
    
    result = {
        'id': i,
        'name': f'TD3_Apprentice_{i}',
        'scores': score_hist,
        'successes': success_hist,
        'margin': margins[i]
    }
    
    return result, w_0, margins[0], app_feature


def train_single_apprentice(i, expert_feature_expectation, feature_expectation, feature_expectation_bar, weights, margins):
    """
    Train a single TD3 Apprentice (i) using projection method.
    Returns: result, new_fe, new_weight, new_margin, new_bar
    """
    print(f"\n{'='*70}")
    print(f"  TD3 Apprentice {i} Training")
    print(f"{'='*70}\n")
    
    apprentice_base_path = TD3_MODELS_DIR / "Apprentices"
    
    # Projection method
    weight, margin, new_bar = solve_projection_method(
        expert_feature_expectation, feature_expectation, feature_expectation_bar, i
    )
    
    print(f"Margin[{i}]: {margin:.6f}")

    # Check termination condition
    if margin <= EPSILON:
        print(f"converged at iteration {i} with margin {margin}")
        return None, None, weight, margin, new_bar
        
    # Define save path for this apprentice
    iter_save_path = apprentice_base_path / f"Apprentice_{i}"
    iter_save_path.mkdir(parents=True, exist_ok=True)

    env = create_env()
    obs_shape = get_obs_shape(env)

    # Initialize NEW Apprentice agent (Fresh Start)
    apprentice = TD3Trainer(
        env=env, input_dims=obs_shape, agent_name=f'Apprentice_{i}',
        model_save_path=str(iter_save_path) + "/",
        exploration_period=EXPLORATION_PERIOD,
        batch_size=BATCH_SIZE, replay_size=REPLAY_SIZE,
        alpha=ALPHA, beta=BETA, gamma=GAMMA, tau=TAU,
        update_actor_every=UPDATE_ACTOR_EVERY, noise_factor=NOISE_FACTOR
    )
    
    # Train
    score_hist, avg_score_hist, success_hist, avg_success_hist = apprentice.td3_train(
        n_episodes=N_EPISODES_APPRENTICE,
        opt_steps=OPT_STEPS,
        reward_weights=weight,
        print_every=PRINT_EVERY,
        plot_save_path=str(TD3_RESULTS_DIR / f"TD3_Apprentice_{i}_Performance.png")
    )
    
    # Save model
    apprentice.save_model()
    
    # Compute feature expectation
    app_feature, app_reward = compute_average_feature(
        apprentice, m=FEATURE_EXPECTATION_EPISODES_APPRENTICE, steps=FEATURE_CALC_STEPS
    )
    
    # Print chunked stats
    print_chunked_stats(f"TD3 Apprentice {i}", success_hist)
    
    # Plot individual performance
    plot_individual_performance(
        f"TD3 Apprentice {i}",
        score_hist, success_hist,
        save_path=str(TD3_RESULTS_DIR / f"TD3_Apprentice_{i}_Individual.png")
    )
    
    result = {
        'id': i,
        'name': f'TD3_Apprentice_{i}',
        'scores': score_hist,
        'successes': success_hist,
        'margin': margin
    }
    
    return result, app_feature, weight, margin, new_bar


def train_remaining_apprentices(expert_feature_expectation, initial_feature_expectation, initial_weights, initial_margins, m=None):
    """
    Train Apprentices 1 to NUM_APPRENTICES using projection method.
    Returns: List of all results (including app 0 which is NOT passed in, effectively, we return results for 1-N but function name implies logic)
    Wait, to match the original return, we should probably accumulate everything outside or return a list of results for 1-N.
    """
    # Note: 'env' and 'obs_shape' were created here but moved inside train_single_apprentice 
    # to avoid stale env issues across long runs
    
    feature_expectation = [initial_feature_expectation] # List of tensors
    feature_expectation_bar = []
    weights = [initial_weights]
    margins = [initial_margins]
    results = [] # Will store results for 1..N
    
    for i in range(1, NUM_APPRENTICES):
        
        res, new_fe, new_w, new_m, new_bar = train_single_apprentice(
            i, expert_feature_expectation, feature_expectation, feature_expectation_bar, weights, margins
        )
        
        if new_bar is not None:
            feature_expectation_bar.append(new_bar)
            
        weights.append(new_w)
        margins.append(new_m)
        
        # Check termination (if result is None)
        if res is None:
             break
             
        feature_expectation.append(new_fe)
        results.append(res)
        
    return results


def train_apprentices(expert=None, m=None):
    """
    Wrapper for backward compatibility or one-shot execution.
    """
    # 1. Expert Features
    expert_feat, _ = compute_expert_features(expert, m=m)
    
    # 2. Apprentice 0
    res0, w0, m0, feat0 = train_apprentice_0(m=m)
    
    all_results = [res0]
    
    # 3. Remaining Apprentices
    res_rest = train_remaining_apprentices(expert_feat, feat0, w0, m0, m=m)
    
    all_results.extend(res_rest)
    
    # Plot apprentice comparison after all training is complete
    from plotting import plot_apprentice_comparison
    # Plot apprentice comparison after all training is complete
    plot_apprentice_comparison(
        "TD3",
        all_results,
        save_dir=TD3_RESULTS_DIR
    )
    
    return all_results


def evaluate_apprentices():
    """Evaluate all trained TD3 Apprentices."""
    env = create_env()
    obs_shape = get_obs_shape(env)
    
    apprentice_base_path = TD3_MODELS_DIR / "Apprentices"
    all_eval_results = []
    
    for i in range(NUM_APPRENTICES):
        iter_path = apprentice_base_path / f"Apprentice_{i}"
        
        if not (iter_path / "actor.pth").exists():
            print(f"Apprentice {i} model not found at {iter_path}. Skipping.")
            continue
        
        print(f"\n--- Evaluating TD3 Apprentice {i} ---")
        apprentice = TD3Trainer(
            env=env, input_dims=obs_shape, agent_name=f'Apprentice_{i}',
            model_load_path=str(iter_path) + "/"
        )
        
        results = evaluate_agent(apprentice, env, episodes=N_EPISODES_APPRENTICE, steps=FEATURE_CALC_STEPS)
        
        success_count = int(np.sum(results['successes']))
        total_episodes = len(results['successes'])
        print(f"TD3 Apprentice {i} Evaluation: Mean Return = {results['mean_return']:.3f}, "
              f"Success: {success_count}/{total_episodes} ({results['success_rate']:.1f}%)")
        
        print_chunked_stats(f"TD3 Apprentice {i} (Eval)", results['successes'])
        
        all_eval_results.append({
            'id': i,
            'name': f'TD3_Apprentice_{i}',
            'scores': results['returns'],
            'successes': results['successes'],
            'mean_return': results['mean_return'],
            'success_rate': results['success_rate']
        })
    
    return all_eval_results


from plotting import plot_individual_performance, plot_comparative_dashboard, plot_apprentice_comparison

# ... (imports remain same) ...

def plot_all_comparisons(expert_train_data, expert_eval_data, apprentice_train_data, apprentice_eval_data):
    """Plot all comparative dashboards."""
    
    # Learning phase comparison
    if expert_train_data and apprentice_train_data:
        expert_dash = {
            'name': 'TD3 Expert',
            'scores': expert_train_data.get('scores', []),
            'successes': expert_train_data.get('successes', [])
        }
        apprentices_dash = [
            {'name': r['name'], 'scores': r['scores'], 'successes': r['successes']}
            for r in apprentice_train_data
        ]
        plot_comparative_dashboard(
            "TD3 Learning Phase Comparison",
            expert_dash, apprentices_dash,
            save_path=str(TD3_RESULTS_DIR / "TD3_Learning_Comparison.png")
        )
        
        # Within-algorithm comparison (Apprentice 1-3)
        plot_apprentice_comparison(
            "TD3",
            apprentice_train_data,
            save_dir=TD3_RESULTS_DIR
        )
    
    # Evaluation phase comparison
    if expert_eval_data and apprentice_eval_data:
        expert_dash = {
            'name': 'TD3 Expert',
            'scores': expert_eval_data.get('returns', []),
            'successes': expert_eval_data.get('successes', [])
        }
        apprentices_dash = [
            {'name': r['name'], 'scores': r['scores'], 'successes': r['successes']}
            for r in apprentice_eval_data
        ]
        plot_comparative_dashboard(
            "TD3 Evaluation Phase Comparison",
            expert_dash, apprentices_dash,
            save_path=str(TD3_RESULTS_DIR / "TD3_Evaluation_Comparison.png")
        )
