import torch
import numpy as np
import gymnasium as gym
import panda_gym
from TD3.td3_algo import TD3Trainer


def collect_expert_trajectories(env_name="PandaReach-v3", episodes=100, steps_per_episode=300,
                                expert_model_path="../TD3/Models/Expert/", save_path="./expert_trajectories.pt",
                                render=False):
    env = gym.make(env_name, render_mode="rgb_array" if render else None)
    obs_shape = env.observation_space['observation'].shape[0] + \
                env.observation_space['achieved_goal'].shape[0] + \
                env.observation_space['desired_goal'].shape[0]

    expert = TD3Trainer(env=env, input_dims=obs_shape, agent_name='Expert', model_load_path=expert_model_path)

    trajectories = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        states = []
        actions = []
        while not (done or truncated):
            current_observation, current_achieved_goal, current_desired_goal = obs.values()
            state = np.concatenate((current_observation, current_achieved_goal, current_desired_goal))
            action = expert.select_action(state)
            obs, reward, done, truncated, _ = env.step(np.array(action))
            states.append(state)
            actions.append(action)
            if len(states) >= steps_per_episode:
                break
        trajectories.append({"states": np.stack(states), "actions": np.stack(actions)})

    torch.save(trajectories, save_path)
    print(f"エキスパート軌跡を {save_path} に保存しました")


if __name__ == "__main__":
    collect_expert_trajectories()
