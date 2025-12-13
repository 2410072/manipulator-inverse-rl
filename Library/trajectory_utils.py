# trajectory_utils.py - Utilities to convert trajectories for imitation library

import numpy as np
import torch
import gymnasium as gym
from imitation.data.types import Trajectory


def flatten_goal_obs(obs_dict):
    """Flatten goal-conditioned observation dict to 1D array."""
    return np.concatenate([
        obs_dict['observation'],
        obs_dict['achieved_goal'],
        obs_dict['desired_goal']
    ])


def load_trajectories_as_imitation_format(pt_path, env):
    """
    Load .pt trajectories and convert to imitation library Trajectory format.
    
    Args:
        pt_path: Path to .pt file containing list of {"states": ..., "actions": ...}
        env: Gymnasium environment (used to get terminal state info)
    
    Returns:
        List of imitation.data.types.Trajectory objects
    """
    raw_trajectories = torch.load(pt_path, weights_only=False)
    
    imitation_trajectories = []
    for traj in raw_trajectories:
        states = traj["states"]  # (T, obs_dim)
        actions = traj["actions"]  # (T, act_dim)
        
        # imitation expects obs to have one more entry than acts
        # obs[0..T], acts[0..T-1], so we need T+1 observations
        # Since we only have T observations, we duplicate the last one
        obs = np.vstack([states, states[-1:]])  # (T+1, obs_dim)
        acts = actions  # (T, act_dim)
        
        # Infos: list of dicts, one per timestep
        infos = np.array([{}] * len(acts))
        
        # Terminal: True only for last step
        terminal = True
        
        trajectory = Trajectory(
            obs=obs,
            acts=acts,
            infos=infos,
            terminal=terminal
        )
        imitation_trajectories.append(trajectory)
    
    return imitation_trajectories


def get_flattened_obs_space(env):
    """
    Get a flattened Box observation space from a goal-conditioned Dict space.
    """
    obs_space = env.observation_space
    assert isinstance(obs_space, gym.spaces.Dict), "Expected Dict observation space"
    
    obs_box = obs_space['observation']
    achieved_box = obs_space['achieved_goal']
    desired_box = obs_space['desired_goal']
    
    assert isinstance(obs_box, gym.spaces.Box)
    assert isinstance(achieved_box, gym.spaces.Box)
    assert isinstance(desired_box, gym.spaces.Box)
    
    total_dim = obs_box.shape[0] + achieved_box.shape[0] + desired_box.shape[0]
    
    low = np.concatenate([obs_box.low, achieved_box.low, desired_box.low])
    high = np.concatenate([obs_box.high, achieved_box.high, desired_box.high])
    
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)
