# gail_runner.py - GAIL training using imitation library

import os
import sys
import numpy as np
import torch
from pathlib import Path

import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.monitor import Monitor

from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.data.wrappers import RolloutInfoWrapper

# Ensure parent directory is in path
current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from config import (
    N_EPISODES_APPRENTICE, BATCH_SIZE, PRINT_EVERY, NUM_APPRENTICES,
    EXPERT_TRAJECTORIES_PATH, GAMMA
)
from trajectory_utils import load_trajectories_as_imitation_format, get_flattened_obs_space
from plotting import plot_individual_performance
from irl_utils import TrackingCallback

# GAIL-specific paths
GAIL_MODELS_DIR = current_dir / "Models" / "GAIL_lib"
GAIL_RESULTS_DIR = current_dir / "Results" / "GAIL_lib"

# Training parameters
SEED = 42
N_ENVS = 1
N_RL_TRAIN_STEPS = 100_000  # Increase for better performance


class FlattenGoalWrapper(gym.ObservationWrapper):
    """Flatten goal-conditioned Dict observation to Box."""
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = get_flattened_obs_space(env)
    
    def observation(self, observation):
        return np.concatenate([
            observation['observation'],
            observation['achieved_goal'],
            observation['desired_goal']
        ]).astype(np.float32)


def create_vec_env(n_envs=N_ENVS, seed=SEED):
    """Create vectorized environment with flattening wrapper."""
    def make_env() -> gym.Env:
        env = gym.make("PandaReach-v3")
        env = Monitor(env) # Add Monitor wrapper
        env = FlattenGoalWrapper(env)
        env = RolloutInfoWrapper(env)
        return env
    
    from stable_baselines3.common.vec_env import DummyVecEnv
    envs = [make_env for _ in range(n_envs)]
    venv = DummyVecEnv(envs)
    return venv


def train_gail(n_episodes=N_EPISODES_APPRENTICE, save_models=True):
    """
    Train an agent using GAIL from the imitation library.
    
    Args:
        n_episodes: Number of episodes to train for
        save_models: Whether to save trained models
    
    Returns:
        Dictionary with training results
    """
    
    # Create directories
    GAIL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    GAIL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    venv = create_vec_env()
    
    # Load expert trajectories
    if not EXPERT_TRAJECTORIES_PATH.exists():
        raise FileNotFoundError(
            f"Expert trajectories not found at {EXPERT_TRAJECTORIES_PATH}. "
            "Please run collect_expert_trajectories.py first."
        )
    
    # Create a single env for trajectory conversion
    single_env = gym.make("PandaReach-v3")
    single_env = FlattenGoalWrapper(single_env)
    
    print("Loading expert trajectories...")
    trajectories = load_trajectories_as_imitation_format(
        str(EXPERT_TRAJECTORIES_PATH), single_env
    )
    print(f"Loaded {len(trajectories)} expert trajectories")
    single_env.close()
    
    # Create learner (PPO agent)
    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0004,
        gamma=GAMMA,
        n_epochs=5,
        seed=SEED,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create reward network (discriminator)
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )
    
    # Create GAIL trainer
    gail_trainer = GAIL(
        demonstrations=trajectories,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )
    
    # Evaluate before training
    print("Evaluating before training...")
    rewards_before, _ = evaluate_policy(
        learner, venv, 50, return_episode_rewards=True
    )
    print(f"Mean reward before training: {np.mean(rewards_before):.2f}")
    
    # Setup callback for episode tracking
    callback = TrackingCallback(n_episodes=n_episodes, print_every=PRINT_EVERY)
    
    # Monkey-patch learner.learn to inject callback
    # This ensures our callback is called during PPO training steps
    original_learn = learner.learn
    def learn_with_callback(*args, **kwargs):
        # Merge our callback with any existing one if necessary, but typically none is passed by imitation
        # If kwargs['callback'] exists, we could chain them, but here we prioritize ours
        kwargs['callback'] = callback
        return original_learn(*args, **kwargs)
    learner.learn = learn_with_callback
    
    # Train
    print(f"Training GAIL for {n_episodes} episodes...")
    # Run for enough steps to cover episodes (controlled by callback)
    gail_trainer.train(total_timesteps=1_000_000)
    
    # Restore original learn (good practice)
    learner.learn = original_learn
    
    # Evaluate after training
    print("Evaluating after training...")
    rewards_after, _ = evaluate_policy(
        learner, venv, 100, return_episode_rewards=True
    )
    # Ensure rewards_after is a list (to satisfy type checkers)
    if not isinstance(rewards_after, list):
        rewards_after = [rewards_after]

    print(f"Mean reward after training: {np.mean(rewards_after):.2f}")
    
    # Save model
    if save_models:
        model_path = GAIL_MODELS_DIR / "gail_policy"
        learner.save(str(model_path))
        print(f"Model saved to {model_path}")
    
    # Plot results
    plot_individual_performance(
        "GAIL Agent",
        callback.scores,
        callback.successes,
        save_path=str(GAIL_RESULTS_DIR / "GAIL_Performance.png")
    )
    
    venv.close()
    
    return {
        'rewards_before': rewards_before,
        'rewards_after': rewards_after,
        'mean_reward': np.mean(rewards_after),
        'std_reward': np.std(rewards_after),
        'score_history': callback.scores,
        'success_history': callback.successes
    }


def evaluate_gail(model_path=None, n_episodes=100):
    """Evaluate a trained GAIL model."""
    if model_path is None:
        model_path = GAIL_MODELS_DIR / "gail_policy"
    
    venv = create_vec_env(n_envs=1)
    
    learner = PPO.load(str(model_path), env=venv, device="cuda" if torch.cuda.is_available() else "cpu")
    
    rewards, _ = evaluate_policy(
        learner, venv, n_episodes, return_episode_rewards=True
    )
    
    print(f"GAIL Evaluation: Mean={np.mean(rewards):.2f}, Std={np.std(rewards):.2f}")
    
    venv.close()
    
    return {
        'rewards': rewards,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards)
    }


if __name__ == "__main__":
    results = train_gail()
    print(f"Training complete. Final mean reward: {results['mean_reward']:.2f}")
