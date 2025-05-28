import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=400, dropout_p=0.2):
        super(MLPActor, self).__init__()
        self.max_action = max_action

        # Input to first hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, action_dim),
            nn.Tanh()

        )


    def forward(self, state):
        action = self.fc1(state)

        return action * self.max_action


class DoubleCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DoubleCritic, self).__init__()

        # Q1 architecture
        self.q1_fc1 = nn.Linear(state_dim + action_dim, 400)
        self.q1_fc2 = nn.Linear(400, 300)
        self.q1_out = nn.Linear(300, 1)

        # Q2 architecture
        self.q2_fc1 = nn.Linear(state_dim + action_dim, 400)
        self.q2_fc2 = nn.Linear(400, 300)
        self.q2_out = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        return self.q1_out(q1)
