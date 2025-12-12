import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# アクターとクリティックのネットワーク
class Actor(nn.Module):
    def __init__(self, state_shape, num_actions, name, checkpoints_dir="../Data/"):
        super(Actor, self).__init__()
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        self.checkpoints_file = os.path.join(checkpoints_dir, name + ".pth")

        self.hidden1 = nn.Linear(in_features=state_shape, out_features=512)
        self.hidden2 = nn.Linear(in_features=512, out_features=256)
        self.hidden3 = nn.Linear(in_features=256, out_features=256)
        self.action_output = nn.Linear(in_features=256, out_features=num_actions)

    def forward(self, state):
        x = torch.relu(self.hidden1(state))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        action = torch.tanh(self.action_output(x))
        return action

class Critic(nn.Module):
    def __init__(self, state_action_shape, name, checkpoints_dir="../Data/"):
        super(Critic, self).__init__()
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        self.checkpoints_file = os.path.join(checkpoints_dir, name + ".pth")

        self.hidden1 = nn.Linear(in_features=state_action_shape, out_features=512)
        self.hidden2 = nn.Linear(in_features=512, out_features=256)
        self.hidden3 = nn.Linear(in_features=256, out_features=256)
        self.q_value = nn.Linear(in_features=256, out_features=1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        q_value = self.q_value(x)
        return q_value


# 敵対的模倣学習（GAIL/AIRL）用の識別器
class Discriminator(nn.Module):
    def __init__(self, state_action_shape, name="discriminator", checkpoints_dir="../Data/"):
        super(Discriminator, self).__init__()
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        self.checkpoints_file = os.path.join(checkpoints_dir, name + ".pth")

        self.hidden1 = nn.Linear(in_features=state_action_shape, out_features=256)
        self.hidden2 = nn.Linear(in_features=256, out_features=128)
        self.output = nn.Linear(in_features=128, out_features=1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        prob = torch.sigmoid(self.output(x))
        return prob