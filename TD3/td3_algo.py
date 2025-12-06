import numpy as np
import gymnasium as gym
import panda_gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
from IPython import display

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 日本語フォント設定
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans CJK JP', 'IPAexGothic', 'Takao']
matplotlib.rcParams['axes.unicode_minus'] = False

import sys
sys.path.append('../utils/')
from networks import Actor, Critic
from replay import ExperienceReplayMemory

    
# エージェント
class TD3Trainer:
    def __init__(self, env, input_dims, alpha=0.001, beta=0.002, gamma=0.99, tau=0.05, 
                 batch_size=256, replay_size=10**6, update_actor_every=2, exploration_period=500, 
                 noise_factor=0.1, agent_name='agent', model_save_path=None, model_load_path=None):
        
        # ハイパーパラメータ
        self.alpha = alpha  # アクターの学習率
        self.beta = beta    # クリティックの学習率
        self.gamma = gamma  # 割引率
        self.tau = tau      # ソフトアップデート係数
        self.batch_size = batch_size  # 学習バッチサイズ
        self.time_step = 0
        self.input_dims = input_dims   # 入力次元
        self.exploration_period = exploration_period  # 探索期間
        self.training_step_count = 0
        self.update_actor_every = update_actor_every
        self.noise_factor = noise_factor   # 探索ノイズ係数
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_score = 0
        self.agent_name = agent_name
        self.is_trained = False
        if model_save_path is None:
            self.model_save_path = f'../Data/{agent_name}'
        else:
            self.model_save_path = model_save_path

        # 環境
        self.env = env
        self.n_actions = env.action_space.shape[0]
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        # リプレイバッファ
        self.memory = ExperienceReplayMemory(replay_size, input_dims, self.n_actions)

        # initialize actor and critic networks
        if model_load_path:
            self.initialize_networks(self.n_actions, checkpoints_dir=model_load_path)
            self.load_model()
        else:
            self.initialize_networks(self.n_actions)
            self.update_target_parameters(tau=1)


    def initialize_networks(self, n_actions, checkpoints_dir=None):
        """
        TD3エージェントのアクターとクリティックを初期化する。
        """
        if checkpoints_dir is None:
            checkpoints_dir=self.model_save_path
            
        self.actor = Actor(state_shape=self.input_dims, num_actions=n_actions, 
                           name="actor", checkpoints_dir=checkpoints_dir).to(self.device)
        self.critic_1 = Critic(state_action_shape=self.input_dims+self.n_actions,
                               name="critic_1", checkpoints_dir=checkpoints_dir).to(self.device)
        self.critic_2 = Critic(state_action_shape=self.input_dims+self.n_actions,
                               name="critic_2", checkpoints_dir=checkpoints_dir).to(self.device)

        self.target_actor = Actor(state_shape=self.input_dims, num_actions=n_actions, 
                                  name="target_actor", checkpoints_dir=checkpoints_dir).to(self.device)
        self.target_critic_1 = Critic(state_action_shape=self.input_dims+self.n_actions, 
                                      name="target_critic_1", checkpoints_dir=checkpoints_dir).to(self.device)
        self.target_critic_2 = Critic(state_action_shape=self.input_dims+self.n_actions, 
                                      name="target_critic_2", checkpoints_dir=checkpoints_dir).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.alpha)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.beta)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.beta)

        self.target_actor_optimizer = optim.Adam(self.target_actor.parameters(), lr=self.alpha)
        self.target_critic_1_optimizer = optim.Adam(self.target_critic_1.parameters(), lr=self.beta)
        self.target_critic_2_optimizer = optim.Adam(self.target_critic_2.parameters(), lr=self.beta)
    
    
    def soft_update(self, target_network, source_network, tau):
        """
        ソフトアップデート則に従いターゲットネットワークの重みを更新する:
            new_weight = tau * old_weight + (1 - tau) * old_target_weight
            θ′ ← τ θ + (1 −τ )θ′
        """
        target_params = target_network.state_dict()
        source_params = source_network.state_dict()

        for key in source_params:
            target_params[key] = tau * source_params[key] + (1.0 - tau) * target_params[key]

        target_network.load_state_dict(target_params)
        
        
    def update_target_parameters(self, tau=None):
        """
        ターゲットアクターと2つのターゲットクリティックをソフトアップデートで更新する。
        """
        if tau is None:
            tau = self.tau

        # update weights of the target actor
        self.soft_update(self.target_actor, self.actor, tau)

        # update weights of the first target critic network
        self.soft_update(self.target_critic_1, self.critic_1, tau)

        # update weights of the second target critic network
        self.soft_update(self.target_critic_2, self.critic_2, tau)
        
        
    def select_action(self, observation):
        """
        エージェントの行動を選択する。
        
        """
        # exploration_periodの間はランダム行動で探索を促す
        if self.time_step < self.exploration_period and self.is_trained==False:
            mu = np.random.normal(scale=self.noise_factor, size=(self.n_actions,))
        else:
            state = torch.tensor([observation], dtype=torch.float32).to(self.device)
            mu = self.actor(state).detach().cpu().numpy()[0]
            
        mu_star = mu + np.random.normal(scale=self.noise_factor, size=self.n_actions)   # add noise
        mu_star = np.clip(mu_star, self.min_action, self.max_action)   # clip action
        self.time_step += 1

        return mu_star
    
    
    def optimize_model(self):
        """
        TD3アルゴリズムによる学習処理。

        ・過去の経験をリプレイバッファからランダムサンプリング
        ・2つのクリティックに対して勾配降下
        ・クリティック2回に1回の頻度でアクターを更新する遅延更新
        """
        # バッファに十分な経験があるか確認
        if self.memory.size < self.batch_size:
            return

        # バッファから経験をサンプリング
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # 2つのクリティックに対して勾配降下
        target_actions = self.target_actor(next_states) + torch.clamp(torch.randn_like(actions) * 0.2, -0.5, 0.5)
        target_actions = torch.clamp(target_actions, self.min_action, self.max_action)

        with torch.no_grad():
            q1_new = self.target_critic_1(next_states, target_actions).squeeze(1)
            q2_new = self.target_critic_2(next_states, target_actions).squeeze(1)
            target = rewards + self.gamma * torch.min(q1_new, q2_new) * (1 - dones)

        q1 = self.critic_1(states, actions).squeeze(1)
        q2 = self.critic_2(states, actions).squeeze(1)
        
        # クリティックの損失
        critic_1_loss = F.mse_loss(q1, target)
        critic_2_loss = F.mse_loss(q2, target)

        # 勾配降下
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # クリティック2更新につき1回だけアクターを更新
        self.training_step_count += 1
        if self.training_step_count % self.update_actor_every != 0:
            return

        actor_loss = -torch.mean(self.critic_1(states, self.actor(states)))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # アクターとクリティックのターゲットをソフトアップデート
        self.update_target_parameters()
        
        
    def td3_train(self, n_episodes=1500, opt_steps=64, reward_weights=None, 
                  print_every=100, render_save_path=None, plot_save_path=None):
        
        if render_save_path:
            env = gym.wrappers.RecordVideo(self.env, video_folder=render_save_path, 
                              episode_trigger=lambda t: t % (n_episodes//10) == 0, disable_logger=True)
        else:
            env = self.env
        
        score_history = []
        avg_score_history = []

        for i in tqdm(range(n_episodes), desc='学習中..'):
            done = False
            truncated = False
            score = 0
            step = 0

            obs_array = []
            actions_array = []
            next_obs_array = []

            observation, info = env.reset()

            while not (done or truncated):
                current_observation, achieved_goal, desired_goal = observation.values()
                state = np.concatenate((current_observation, achieved_goal, desired_goal))
                # print(state)

                # 行動を選択
                action = self.select_action(state)

                # 行動を実行
                next_observation, reward, done, truncated, _ = env.step(np.array(action))
                next_obs, next_achieved_goal, next_desired_goal = next_observation.values()
                next_state = np.concatenate((next_obs, next_achieved_goal, next_desired_goal))
                # print(next_observation)
                
                if reward_weights is not None:
                    features = self.construct_feature_vector(observation).to(self.device)
                    reward_weights = reward_weights.to(self.device)
                    reward = (reward_weights.t()) @ features                 # w^T ⋅ φ
                    
                    #print(reward_weights.t().shape, features.shape, reward.shape)

                # 経験をリプレイバッファに保存
                self.memory.push(state, action, reward, next_state, done)

                obs_array.append(observation)
                actions_array.append(action)
                next_obs_array.append(next_observation)

                observation = next_observation
                if reward_weights is not None:
                    score += reward.cpu().numpy()[0]
                else:
                    score += reward
                step += 1

            # HERでリプレイバッファを拡張
            self.her_augmentation(obs_array, actions_array, next_obs_array)

            # 複数ステップでエージェントを最適化
            for _ in range(opt_steps):
                self.optimize_model()

            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            avg_score_history.append(avg_score)

            if avg_score > self.best_score:
                self.best_score = avg_score

            if i % print_every==0 and i!=0:
                print(f"エピソード: {i} \t ステップ: {step} \t スコア: {score:.1f} \t 平均スコア: {avg_score:.1f}")
            
            # モデルを保存
            if self.model_save_path and i % (n_episodes//10)==0:
                self.save_model()
                
        # 学習性能をプロット
        self.plot_scores(scores=score_history, avg_scores=avg_score_history, plot_save_path=plot_save_path)

        return score_history, avg_score_history
    
            
    def her_augmentation(self, observations, actions, next_observations, k = 4):
        """
        Hindsight Experience Replay (HER) を用いてリプレイバッファを拡張する。

        観測・行動・次状態を走査し、各経験から複数の学習サンプルを生成する。
        """
        # リプレイバッファを拡張
        num_samples = len(actions)
        for index in range(num_samples):
            for _ in range(k):
                # エピソード後半から未来の観測とゴールをサンプリング
                future_index = np.random.randint(index, num_samples)
                future_observation, future_achieved_goal, _ = next_observations[future_index].values()

                # 現在の観測と行動を取り出す
                observation, _, _ = observations[future_index].values()
                
                # 未来の達成ゴールを目的ゴールとして状態を構成
                state = torch.tensor(np.concatenate((observation, future_achieved_goal, future_achieved_goal)), 
                                     dtype=torch.float32).to(self.device)

                next_observation, _, _ = next_observations[future_index].values()
                
                # 同じゴールで次状態を構成
                next_state = torch.tensor(np.concatenate((next_observation, future_achieved_goal, 
                                                          future_achieved_goal)), dtype=torch.float32).to(self.device)

                # 行動を取り出す
                action = torch.tensor(actions[future_index], dtype=torch.float32).to(self.device)
                
                # 未来ゴールを達成したと仮定した報酬を計算
                reward = self.env.unwrapped.compute_reward(future_achieved_goal, future_achieved_goal, 1.0)

                # 生成した経験をバッファへ保存
                state = state.cpu().numpy()
                action = action.cpu().numpy()
                next_state = next_state.cpu().numpy()

                self.memory.push(state, action, reward, next_state, True)
                
                
    def construct_feature_vector(self, observation):
        """
        観測を正規化し、特徴ベクトルを組み立てる。
        """
        # 観測要素を正規化
        obs = observation['observation']
        achieved_goal = observation['achieved_goal']
        desired_goal = observation['desired_goal']

        normalized_obs = (obs - self.env.observation_space['observation'].low) / \
                         (self.env.observation_space['observation'].high - self.env.observation_space['observation'].low)
        normalized_achieved_goal = (achieved_goal - self.env.observation_space['achieved_goal'].low) / \
                                    (self.env.observation_space['achieved_goal'].high - self.env.observation_space['achieved_goal'].low)
        normalized_desired_goal = (desired_goal - self.env.observation_space['desired_goal'].low) / \
                                   (self.env.observation_space['desired_goal'].high - self.env.observation_space['desired_goal'].low)

        # 特徴ベクトルを構築
        feature_vector = np.concatenate((normalized_obs, normalized_achieved_goal, normalized_desired_goal))

        return torch.tensor(feature_vector, dtype=torch.float32)
                
                
    def test_model(self, steps, env=None, save_states=False, render_save_path=None, fps=30):
        """
        学習済みエージェントを環境で実行する。
        """
        if env is None:
            env = self.env
        episode_score = 0
        state_list = []     # 状態特徴ベクトルを蓄積するリスト
        
        observation, info = env.reset()
        current_observation, current_achieved_goal, current_desired_goal = observation.values()
        state = np.concatenate((current_observation, current_achieved_goal, current_desired_goal))
                        
        if save_states:
            state_list.append(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device))
        
        images = []
        done = False
        truncated = False
        
        # 環境を指定ステップ実行し、報酬（必要なら状態）を収集
        with torch.inference_mode():
            for i in range(steps):
                if render_save_path:
                    images.append(env.render())

                action = self.select_action(state)

                observation, reward, done, truncated, _ = env.step(np.array(action))
                
                current_observation, current_achieved_goal, current_desired_goal = observation.values()
                state = np.concatenate((current_observation, current_achieved_goal, current_desired_goal))

                if save_states:
                    state_list.append(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device))
                
                episode_score += reward

                if done or truncated:
                    if render_save_path:
                        images.append(env.render())
                    break

        if render_save_path:
            # env.close()
            imageio.mimsave(f'{render_save_path}.gif', images, fps=fps, loop=0)
            with open(f'{render_save_path}.gif', 'rb') as f:
                display.display(display.Image(data=f.read(), format='gif'))
                
        if not save_states:
            return episode_score
        else:
            return episode_score, state_list
                
                
    def save_model(self):
        """
        学習済みモデルを保存する。
        """
        torch.save(self.actor.state_dict(), self.actor.checkpoints_file)
        torch.save(self.critic_1.state_dict(), self.critic_1.checkpoints_file)
        torch.save(self.critic_2.state_dict(), self.critic_2.checkpoints_file)
        torch.save(self.target_actor.state_dict(), self.target_actor.checkpoints_file)
        torch.save(self.target_critic_1.state_dict(), self.target_critic_1.checkpoints_file)
        torch.save(self.target_critic_2.state_dict(), self.target_critic_2.checkpoints_file)

    def load_model(self):
        """
        学習済みモデルを読み込む。
        """
        self.is_trained = True
        self.actor.load_state_dict(torch.load(self.actor.checkpoints_file))
        self.critic_1.load_state_dict(torch.load(self.critic_1.checkpoints_file))
        self.critic_2.load_state_dict(torch.load(self.critic_2.checkpoints_file))
        self.target_actor.load_state_dict(torch.load(self.target_actor.checkpoints_file))
        self.target_critic_1.load_state_dict(torch.load(self.target_critic_1.checkpoints_file))
        self.target_critic_2.load_state_dict(torch.load(self.target_critic_2.checkpoints_file))
        
        
    def plot_scores(self, scores, avg_scores, plot_save_path):
        """
        エージェントの性能をプロットする。
        """
        plt.figure(figsize=(10,8))
        plt.plot(scores)
        plt.plot(avg_scores)
        plt.title(f'{self.agent_name} のパフォーマンス')
        plt.xlabel('エピソード')
        plt.ylabel('スコア')
        if plot_save_path:
            plt.savefig(plot_save_path, bbox_inches='tight')
            plt.show()
        else:
            plt.show()
