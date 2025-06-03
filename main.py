import numpy as np
import torch
import os

from enviroment import Environment      # Your UAV environment
from model import  DoubleCritic
from TransformerActor import TransformerActor
from policy import TD3                  # Or whatever your agent class is called
from memory import ReplayBuffer
import pandas as pd

# ==== Initialize environment and agent ====
env = Environment()
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = 1.0
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = TD3(
    state_dim=state_dim,
    action_dim=action_dim,
    max_action=max_action,
    actor_class=TransformerActor,
    critic_class=DoubleCritic,
    device=device
)

replay_buffer = ReplayBuffer(state_dim, action_dim)

# ==== Training settings ====
episodes          = 10000
max_steps         = 150
batch_size        = 512
epsilon_start     = 1.0     # start fully random
epsilon_end       = 0.01    # end mostly greedy
epsilon_decay     = episodes  # linear decay over all episodes
policy_noise      = 0.05    # small Gaussian noise on policy actions
noise_clip        = 0.1
episode_rewards   = []
energy_consumption=[]
aoi_list =[]
train_call =0
actor_losses=[]
critic_losses=[]
total_episode=[]

# ==== Training loop ====
for episode in range(episodes):
    state   = env.reset()
    #print("State form en",state)
    ep_reward = 0
    ep_energy =0
    ep_aoi=0
    ep_actor=0
    ep_critic=0

   
    epsilon = epsilon_end+(epsilon_start-epsilon_end)*math.exp(-1.0 * episode/30)

    for step in range(max_steps):
        # ε-greedy action:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
            #print("Action from En ",action)
        else:
            action = agent.select_action(np.array(state))
            #print("Action from Po ",action)
            # optional smoothing noise around the greedy action
            noise = np.random.normal(0, policy_noise, size=action_dim)
            action = (action + noise).clip(-max_action, max_action)

        # interact with env
        next_state, reward, done, energy,aoi = env.step(action)
        #print("Reward",step,reward)
        #print(energy)
        #print(aoi)

        # store in buffer
        replay_buffer.add(state, action, reward, next_state, float(done))

        state      = next_state
        ep_reward += reward
        ep_energy+=energy
        ep_aoi+=aoi


        # train once buffer has enough samples
        if len(replay_buffer) >= batch_size*5:
            info_dict =agent.train(replay_buffer, batch_size)
            ep_critic +=info_dict["critic_loss"]
            if info_dict["actor_loss"] is None :
                ep_actor +=0
            else:
                ep_actor+=info_dict["actor_loss"]

            #print(info_dict["actor_loss"],info_dict["critic_loss"])
            #train_call +=1


        if done:
            break

    episode_rewards.append(ep_reward / max_steps)
    energy_consumption.append(ep_energy/max_steps)
    critic_losses.append(ep_critic/max_steps)
    actor_losses.append(ep_actor/max_steps)
    aoi_list.append(ep_aoi)
    total_episode.append(episode)
    print(f"Ep {episode+1:3d} | Avg Reward: {ep_reward/max_steps:8.2f} | epsilon = {epsilon:.3f}   | Avg Critic Loss {ep_critic/max_steps:.2f}  |  Avg Actor Loss {ep_actor/max_steps:.2f}")
    print("--------------------------------------------------------------------------------------------------------------------")

pd.DataFrame({
    "Episode": total_episode,
    "AvgReward": episode_rewards,
    "Actor Loss":actor_losses,
    "Critic Loss":critic_losses,
    "Energy":energy_consumption,
    "AoI":aoi_list

}).to_csv("training_transfomer_log.csv", index=False)


