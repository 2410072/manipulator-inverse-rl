# gail_runner.py - GAIL Apprentice training/evaluation

import os
import numpy as np
import torch
from pathlib import Path
import sys

# Ensure current directory is in sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import config and utils
from config import (
    N_EPISODES_APPRENTICE, OPT_STEPS, BATCH_SIZE,
    EXPLORATION_PERIOD, PRINT_EVERY, NUM_APPRENTICES,
    FEATURE_CALC_STEPS,
    ALPHA, BETA, GAMMA, TAU, REPLAY_SIZE, NOISE_FACTOR, UPDATE_ACTOR_EVERY,
    EXPERT_MODEL_PATH, GAIL_MODELS_DIR, GAIL_RESULTS_DIR, EXPERT_TRAJECTORIES_PATH,
    TD3_MODELS_DIR, SEED
)
from irl_utils import create_env, get_obs_shape, print_chunked_stats, evaluate_agent, set_seed
from plotting import plot_individual_performance, plot_comparative_dashboard

# Import GAIL modules
from gail_algo import GAILTrainer, build_expert_loader
from collect_expert_trajectories import collect_expert_trajectories


def ensure_expert_trajectories():
    """Ensure expert trajectories file exists."""
    if not EXPERT_TRAJECTORIES_PATH.exists():
        print(f"Expert trajectories not found at {EXPERT_TRAJECTORIES_PATH}. Collecting...")
        # Use TD3 Expert model for trajectory collection
        expert_model_path = TD3_MODELS_DIR / "Expert"
        if not (expert_model_path / "actor.pt").exists():
            expert_model_path = EXPERT_MODEL_PATH
        
        collect_expert_trajectories(
            env_name="PandaReach-v3",
            episodes=200,
            steps_per_episode=300,
            expert_model_path=str(expert_model_path) + "/",
            save_path=str(EXPERT_TRAJECTORIES_PATH),
            render=False,
            seed=SEED
        )
    else:
        print(f"Using cached expert trajectories from {EXPERT_TRAJECTORIES_PATH}")


def train_gail_apprentice_i(i, seed=None):
    """Train a single GAIL Apprentice i."""
    current_seed = seed if seed is not None else SEED
    
    print(f"\n{'='*70}")
    print(f"  GAIL Apprentice {i} Training")
    print(f"{'='*70}\n")
    
    set_seed(current_seed)
    env = create_env()
    obs_shape = get_obs_shape(env)
    
    # Ensure directories exist
    GAIL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    GAIL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load expert data (Assumed ensured)
    expert_loader = build_expert_loader(
        str(EXPERT_TRAJECTORIES_PATH),
        batch_size=BATCH_SIZE,
        device=None,
        shuffle=True
    )
    
    # Create save path for this apprentice
    iter_save_path = GAIL_MODELS_DIR / f"Apprentice_{i}"
    iter_save_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize fresh GAIL agent
    agent = GAILTrainer(
        env=env,
        input_dims=obs_shape,
        agent_name=f'GAIL_Apprentice_{i}',
        model_save_path=str(iter_save_path) + "/",
        exploration_period=EXPLORATION_PERIOD,
        batch_size=BATCH_SIZE,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        tau=TAU,
        replay_size=REPLAY_SIZE,
        noise_factor=NOISE_FACTOR,
        update_actor_every=UPDATE_ACTOR_EVERY,
        disc_lr=3e-4,
        disc_updates=2,
        gail_reward_scale=1.0,
        expert_loader=expert_loader,
    )
    
    # Train
    score_hist, avg_score_hist, success_hist, avg_success_hist = agent.gail_train(
        n_episodes=N_EPISODES_APPRENTICE,
        opt_steps=OPT_STEPS,
        print_every=PRINT_EVERY,
        plot_save_path=str(GAIL_RESULTS_DIR / f"GAIL_Apprentice_{i}_Performance.png"),
        seed=current_seed
    )
    
    # Save model
    agent.save_model()
    print(f"GAIL Apprentice {i} saved to {iter_save_path}")
    
    # Print chunked stats
    print_chunked_stats(f"GAIL Apprentice {i}", success_hist)
    
    # Plot individual performance
    plot_individual_performance(
        f"GAIL Apprentice {i}",
        score_hist, success_hist,
        save_path=str(GAIL_RESULTS_DIR / f"GAIL_Apprentice_{i}_Individual.png")
    )
    
    return {
        'id': i,
        'name': f'GAIL_Apprentice_{i}',
        'scores': score_hist,
        'successes': success_hist
    }


def train_apprentices(seed=None):
    """
    Train GAIL Apprentices 0-10.
    Returns all training results.
    """
    current_seed = seed if seed is not None else SEED
    
    # Ensure expert trajectories exist
    ensure_expert_trajectories()
    
    all_results = []
    
    # Start from 1 to match TD3 (which uses Apprentice 0 for initial exploration)
    for i in range(1, NUM_APPRENTICES):
        result = train_gail_apprentice_i(i, seed=current_seed)
        all_results.append(result)
    
    # Plot apprentice comparison after all training is complete
    from plotting import plot_apprentice_comparison
    plot_apprentice_comparison(
        "GAIL",
        all_results,
        save_dir=GAIL_RESULTS_DIR
    )
    
    return all_results


def evaluate_apprentices(episodes=None):
    """Evaluate all trained GAIL Apprentices."""
    set_seed(SEED)
    env = create_env()
    obs_shape = get_obs_shape(env)
    
    if episodes is None:
        episodes = N_EPISODES_APPRENTICE
    
    # Ensure expert trajectories exist for loading
    ensure_expert_trajectories()
    expert_loader = build_expert_loader(
        str(EXPERT_TRAJECTORIES_PATH),
        batch_size=BATCH_SIZE,
        device=None,
        shuffle=True
    )
    
    all_eval_results = []
    
    # Start from 1 to match train_apprentices (skipping Apprentice 0)
    for i in range(1, NUM_APPRENTICES):
        iter_path = GAIL_MODELS_DIR / f"Apprentice_{i}"
        if not (iter_path / "actor.pth").exists():
            print(f"GAIL Apprentice {i} model not found at {iter_path}. Skipping.")
            continue
        
        print(f"\n--- Evaluating GAIL Apprentice {i} ---")
        agent = GAILTrainer(
            env=env,
            input_dims=obs_shape,
            agent_name=f'GAIL_Apprentice_{i}',
            model_load_path=str(iter_path) + "/",
            expert_loader=expert_loader
        )
        
        results = evaluate_agent(agent, env, episodes=episodes, steps=FEATURE_CALC_STEPS)
        
        success_count = int(np.sum(results['successes']))
        total_episodes = len(results['successes'])
        print(f"GAIL Apprentice {i} Evaluation: Mean Return = {results['mean_return']:.3f}, "
              f"Success: {success_count}/{total_episodes} ({results['success_rate']:.1f}%)")
        
        print_chunked_stats(f"GAIL Apprentice {i} (Eval)", results['successes'])
        
        all_eval_results.append({
            'id': i,
            'name': f'GAIL_Apprentice_{i}',
            'scores': results['returns'],
            'successes': results['successes'],
            'mean_return': results['mean_return'],
            'success_rate': results['success_rate']
        })
    
    return all_eval_results


from plotting import plot_individual_performance, plot_comparative_dashboard, plot_apprentice_comparison

# ... (imports remain same) ...

def plot_all_comparisons(apprentice_train_data, apprentice_eval_data):
    """Plot all GAIL comparative dashboards."""
    
    # Learning phase comparison (no Expert for GAIL, just compare apprentices)
    if apprentice_train_data:
        apprentices_dash = [
            {'name': r['name'], 'scores': r['scores'], 'successes': r['successes']}
            for r in apprentice_train_data
        ]
        plot_comparative_dashboard(
            "GAIL Learning Phase Comparison",
            None, apprentices_dash,
            save_path=str(GAIL_RESULTS_DIR / "GAIL_Learning_Comparison.png")
        )
        
        # Within-algorithm comparison (Apprentice 1-3)
        plot_apprentice_comparison(
            "GAIL",
            apprentice_train_data,
            save_dir=GAIL_RESULTS_DIR
        )
    
    # Evaluation phase comparison
    if apprentice_eval_data:
        apprentices_dash = [
            {'name': r['name'], 'scores': r['scores'], 'successes': r['successes']}
            for r in apprentice_eval_data
        ]
        plot_comparative_dashboard(
            "GAIL Evaluation Phase Comparison",
            None, apprentices_dash,
            save_path=str(GAIL_RESULTS_DIR / "GAIL_Evaluation_Comparison.png")
        )
