import torch
import torch.nn as nn
import torch.nn.functional as F


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
