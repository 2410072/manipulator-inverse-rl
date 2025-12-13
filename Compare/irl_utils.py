# utils.py - Common utility functions

import numpy as np
import gymnasium as gym
import panda_gym
import torch
from tqdm import tqdm

from config import ENV_NAME


def create_env():
    """Create and return the PandaReach-v3 environment."""
    
    # Preventing "Only one local in-process GUI/GUI_SERVER connection allowed" error
    # If a previous environment wasn't closed properly, PyBullet might still be connected.
    # try:
    #     import pybullet
    #     if pybullet.isConnected():
    #         pybullet.disconnect()
    # except ImportError:
    #     pass

    env = gym.make(
        ENV_NAME,
        render_mode="rgb_array",
        renderer="OpenGL",
        render_target_position=[0, 0.15, 0.25],
        render_distance=0.85,
        render_yaw=135,
        render_pitch=-20,
    )
    return env


def get_obs_shape(env):
    """Get the observation shape from the environment."""
    return (
        env.observation_space['observation'].shape[0] +
        env.observation_space['achieved_goal'].shape[0] +
        env.observation_space['desired_goal'].shape[0]
    )


def compute_average_feature(agent, m=2000, steps=1000):
    """
    Compute average feature vector (feature expectation) and mean reward over m episodes.
    """
    with torch.inference_mode():
        feature_sum, reward_sum = None, None

        for i in tqdm(range(m), desc='Computing features and rewards'):
            reward, success, states = agent.test_model(steps=steps, save_states=True)

            episode_mean = torch.stack(states).mean(0)

            if feature_sum is None:
                feature_sum, reward_sum = episode_mean, reward
            else:
                feature_sum += episode_mean
                reward_sum += reward

        if feature_sum is not None and reward_sum is not None:
            feature_sum /= m
            reward_sum /= m

        print('\nFeature expectation: ', feature_sum)
        print('\nMean reward: ', reward_sum)

    return feature_sum, reward_sum


def calculate_chunked_stats(history, chunk_size=50):
    """Calculate success/failure stats for each chunk of episodes."""
    stats = []
    n = len(history)
    for i in range(0, n, chunk_size):
        chunk = history[i:i+chunk_size]
        success_count = sum(chunk)
        failure_count = len(chunk) - success_count
        success_rate = success_count / len(chunk) if chunk else 0
        stats.append({
            'chunk_idx': i // chunk_size,
            'start_episode': i,
            'end_episode': min(i + chunk_size, n) - 1,
            'success_count': success_count,
            'failure_count': failure_count,
            'success_rate': success_rate
        })
    return stats


def print_chunked_stats(agent_name, success_history, chunk_size=50):
    """Print formatted chunked stats for an agent."""
    stats = calculate_chunked_stats(success_history, chunk_size)
    print(f"\n--- {agent_name} Stats (per {chunk_size} episodes) ---")
    print(f"{'Chunk':<6} {'Range':<15} {'Success':<10} {'Fail':<10} {'Rate':<10}")
    print("-" * 60)
    for s in stats:
        range_str = f"{s['start_episode']}-{s['end_episode']}"
        print(f"{s['chunk_idx']:<6} {range_str:<15} {s['success_count']:<10} {s['failure_count']:<10} {s['success_rate']*100:>6.2f}%")
    print()


def evaluate_agent(agent, env, episodes=50, steps=200):
    """Run noise-free evaluation of an agent."""
    print(f"Evaluating agent for {episodes} episodes...")
    returns = []
    successes = []
    
    for _ in tqdm(range(episodes), desc="Evaluating"):
        ep_ret, success = agent.test_model(env=env, steps=steps, render_save_path=None)
        returns.append(ep_ret)
        successes.append(1 if success else 0)
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    success_rate = np.mean(successes) * 100
    
    return {
        'mean_return': mean_return,
        'std_return': std_return,
        'success_rate': success_rate,
        'successes': successes,
        'returns': returns
    }
