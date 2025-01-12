from agents import PPOAgent
from utils import ReplayBuffer
import gym
import torch
import numpy as np


env = gym.make("InvertedPendulum-v4")
# env = gym.make("CartPole-v1")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
print(act_dim)
# act_dim = env.action_space.n
# low, high = env.action_space.low, env.action_space.high
# print("Low", low, "high ", high)

# Create PPOAgent in continuous mode
ppo_agent = PPOAgent(
    observation_dim=obs_dim,
    action_dim=act_dim,
    # high=high,
    # low=low,
    discrete=False  # <--- indicates continuous
)

replay_buffer = ReplayBuffer()
episode = 32
batch = 32

cur_ob = np.zeros((episode, *env.observation_space.shape), dtype=np.float32)
next_ob = np.zeros((episode, *env.observation_space.shape), dtype=np.float32)
cur_rew = np.zeros((episode,), dtype=np.float32)
cur_done = np.zeros((episode,), dtype=np.bool_)
cur_ac = np.zeros((episode,), dtype=np.int64)
cur_log_prob = np.zeros((episode,), dtype=np.float32)

num_episodes = 1000000
num_update = 1
past_reward = []

for ep in range(num_episodes):
    done = 0
    observation, _ = env.reset()
    episode_reward = 0
    
    # You might store transitions for your agent’s update
    transitions = []
    n = 1
    total_rew = 0
    while not done:
        # Select an action (using your agent/policy)
        action, log_prob= ppo_agent.get_action(observation)
        # print("action", action)
        
        # Step the environment
        next_observation, reward, done, truncated, info = env.step(action)

        done = done or truncated
        total_rew += reward
                
        cur_ob[n-1] = observation
        next_ob[n-1] = next_observation
        cur_rew[n-1] = reward
        cur_done[n-1] = done
        cur_ac[n-1] = action
        cur_log_prob[n-1] = log_prob
        
        # Move to the next observation
        observation = next_observation

        if n % episode == 0:
            cur_ad = ppo_agent.calculate_advantage(cur_ob, next_ob, cur_rew, cur_done)
            replay_buffer.add_batch(cur_ob, next_ob, cur_ac, cur_rew, cur_done, cur_ad, cur_log_prob)
            # for a in range(num_update):
            #     samples = replay_buffer.sample(batch)
            #     val_l, ac_loss = ppo_agent.learn(samples)
            n = 0
        n += 1
    if n != 1:
        cur_ad = ppo_agent.calculate_advantage(cur_ob[:n-1], next_ob[:n-1], cur_rew[:n-1], cur_done[:n-1])
        replay_buffer.add_batch(cur_ob[:n-1], next_ob[:n-1], cur_ac[:n-1], cur_rew[:n-1], cur_done[:n-1], cur_ad, cur_log_prob[:n-1])
    # print("total reward ", total_rew)
    past_reward.append(total_rew)
    for a in range(num_update):
        samples = replay_buffer.sample(batch)
        val_l, ac_loss = ppo_agent.learn(samples)

    
    # print("most recent lsos ", val_l, ac_loss)
    if ep % 50 ==0:
        print(np.mean(past_reward[-20:]))
        print("most recent loss ", val_l.item(), "ac loss ", ac_loss.item())



      
    
    # In a real setting, update your agent using transitions from this episode
env.close()
