import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class TD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        actor_class,
        critic_class,
        actor_lr=3e-4,         # Lowered LR for stability
        critic_lr=3e-3,
        tau=0.005,
        gamma=0.99,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        action_reg=1e-3,       # L2 penalty coefficient
        device=None
    ):
        # Device configuration
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize actor and target actor
        self.actor = actor_class(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Initialize critic and target critic
        self.critic = critic_class(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Hyperparameters
        self.max_action = max_action
        self.tau = tau
        self.gamma = gamma
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        self.action_reg = action_reg

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
      self.total_it += 1

      # Skip training if not enough samples (optional safety check)
      if replay_buffer.size < batch_size:
          return {"critic_loss": None, "actor_loss": None}

      # Sample from replay buffer
      state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)
      state      = state.to(self.device)
      action     = action.to(self.device)
      reward     = reward.to(self.device)
      next_state = next_state.to(self.device)
      not_done   = not_done.to(self.device)

      # -------------------- Critic Update --------------------
      with torch.no_grad():
          # Add clipped noise to target actions
          noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
          next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

          # Target Q-values
          target_Q1, target_Q2 = self.critic_target(next_state, next_action)
          target_Q = torch.min(target_Q1, target_Q2)
          target_Q = reward + not_done * self.gamma * target_Q

          # Optional: clip target Q to prevent explosion
          target_Q = target_Q.clamp(-1e3, 1e3)  

      # Current Q estimates
      current_Q1, current_Q2 = self.critic(state, action)
      critic_loss = F.mse_loss(current_Q1, target_Q)

      self.critic_optimizer.zero_grad()
      critic_loss.backward()

      #  Clip critic gradients to prevent instability
      torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
      self.critic_optimizer.step()

      # -------------------- Actor Update --------------------
      actor_loss = None
      if self.total_it % self.policy_freq == 0:
          # Get actions from actor and corresponding Q-values
          pi = self.actor(state)
          Q_val = self.critic.Q1(state, pi)

          #  Compute actor loss (maximize Q)
          actor_loss = -Q_val.mean()

          #  Add L2 regularization (to control action size)
          actor_loss += self.action_reg * (pi.pow(2).mean())  # usually set action_reg=1e-2

          self.actor_optimizer.zero_grad()
          actor_loss.backward()

          #  Clip actor gradients to avoid explosion
          torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
          self.actor_optimizer.step()

          #  Soft update of target networks (actor and critic)
          self._soft_update(self.critic, self.critic_target)
          self._soft_update(self.actor, self.actor_target)

      #  Return loss values (with fallback for actor_loss)
      return {
        'critic_loss': critic_loss.item(),
        'actor_loss': actor_loss.item() if actor_loss is not None else None
      }

    def _soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
