import os
import sys
from pathlib import Path

import torch
import numpy as np
import gymnasium as gym
import panda_gym

# Assuming running from Compare directory or package
try:
    from td3_algo import TD3Trainer
    from config import EXPERT_MODEL_PATH
except ImportError:
    # Handle case where script is run from parent directory
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from td3_algo import TD3Trainer
    from config import EXPERT_MODEL_PATH


def collect_expert_trajectories(env_name="PandaReach-v3", episodes=100, steps_per_episode=300,
                                expert_model_path=None, save_path="./expert_trajectories.pt",
                                render=False, seed=None):
    
    if expert_model_path is None:
        expert_model_path = str(EXPERT_MODEL_PATH) + "/"
        
    # panda_gym requires "rgb_array" or "human" render_mode
    env = gym.make(env_name, render_mode="rgb_array")
    
    # Type hint for observation space to avoid lint errors
    # We know this environment uses Dict space
    observation_space = env.observation_space
    assert isinstance(observation_space, gym.spaces.Dict), "Observation space must be a Dict"
    
    obs_space = observation_space['observation']
    achieved_goal_space = observation_space['achieved_goal']
    desired_goal_space = observation_space['desired_goal']

    # Assert sub-spaces are Box to ensure shape is defined and not None for type checkers
    assert isinstance(obs_space, gym.spaces.Box)
    assert isinstance(achieved_goal_space, gym.spaces.Box)
    assert isinstance(desired_goal_space, gym.spaces.Box)

    obs_shape = obs_space.shape[0] + \
                achieved_goal_space.shape[0] + \
                desired_goal_space.shape[0]

    expert = TD3Trainer(env=env, input_dims=obs_shape, agent_name='Expert', model_load_path=expert_model_path)

    trajectories = []
    print(f"Collecting trajectories from {expert_model_path}...")
    
    for i in range(episodes):
        if i == 0 and seed is not None:
            obs, _ = env.reset(seed=seed)
        else:
            obs, _ = env.reset()
        done = False
        truncated = False
        states = []
        actions = []
        while not (done or truncated):
            current_observation = obs['observation']
            current_achieved_goal = obs['achieved_goal']
            current_desired_goal = obs['desired_goal']
            
            state = np.concatenate((current_observation, current_achieved_goal, current_desired_goal))
            action = expert.select_action(state)
            obs, reward, done, truncated, _ = env.step(np.array(action))
            states.append(state)
            actions.append(action)
            if len(states) >= steps_per_episode:
                break
        trajectories.append({"states": np.stack(states), "actions": np.stack(actions)})

    torch.save(trajectories, save_path)
    print(f"Expert trajectories saved to {save_path}")


if __name__ == "__main__":
    try:
        from config import SEED
    except ImportError:
         seed = 42 # Fallback
         SEED = 42
    
    # Also set global seed if possible? No, the function takes seed.
    collect_expert_trajectories(seed=SEED)
