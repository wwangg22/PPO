import numpy as np
import torch
import torch.nn as nn

class ReplayBuffer():
    def __init__(self, size = 32):
        self.observation = None
        self.next_observation = None
        self.reward = None
        self.action = None
        self.done = None
        self.advantage = None
        self.log_probs = None
        self.maxsize = size
        self.size = 0

    def sample(self, batch_size):
        rand_indices = np.random.randint(0, self.size, size=(batch_size,)) % self.maxsize
        return {
            "observations": self.observation[rand_indices],
            "actions": self.action[rand_indices],
            "rewards": self.reward[rand_indices],
            "next_observations": self.next_observation[rand_indices],
            "dones": self.done[rand_indices],
            "advantages": self.advantage[rand_indices],
            "log_probs": self.log_probs[rand_indices],
        }

    def __len__(self):
        return self.size

    def add(self, observation, next_observation, action, reward, done, advantage, log_probs):
        if isinstance(reward, (float, int)):
            reward = np.array(reward)
        if isinstance(done, bool):
            done = np.array(done)
        if isinstance(action, int):
            action = np.array(action, dtype=np.int64)
        if isinstance(advantage, (float, int)):
            advantage= np.array(advantage)
        # print('log probs ', log_probs)
        if self.observation is None:
            self.observation = np.empty((self.maxsize, *observation.shape), dtype= observation.dtype)
            self.next_observation = np.empty((self.maxsize, *next_observation.shape), dtype= next_observation.dtype)
            self.reward = np.empty((self.maxsize, *reward.shape), dtype = reward.dtype)
            self.action = np.empty((self.maxsize, *action.shape), dtype=action.dtype)
            self.done = np.empty((self.maxsize, *done.shape), dtype=done.dtype)
            self.advantage = np.empty((self.maxsize, *advantage.shape), dtype=advantage.dtype)
            self.log_probs = np.empty((self.maxsize, *log_probs.shape), dtype=log_probs.dtype)

        assert self.observation.shape[1:] == observation.shape
        assert self.next_observation.shape[1:] == next_observation.shape
        assert self.action.shape[1:] == action.shape

        self.observation[self.size % self.maxsize] = observation
        self.action[self.size % self.maxsize] = action
        self.reward[self.size % self.maxsize] = reward
        self.next_observation[self.size % self.maxsize] = next_observation
        self.done[self.size % self.maxsize] = done
        self.advantage[self.size%self.maxsize] = advantage
        self.log_probs[self.size % self.maxsize] = log_probs

        self.size += 1
    
    def add_batch(self, observation, next_observation, action, reward, done, advantage, log_probs):
        #batches need to be given in [batch_size, data_shape]
        size = len(observation)
        if self.observation is None:
            self.observation = np.empty((self.maxsize, *observation[0].shape), dtype= observation.dtype)
            self.next_observation = np.empty((self.maxsize, *next_observation[0].shape), dtype= next_observation.dtype)
            self.reward = np.empty((self.maxsize, *reward[0].shape), dtype = reward.dtype)
            self.action = np.empty((self.maxsize, *action[0].shape), dtype=action.dtype)
            self.done = np.empty((self.maxsize, *done[0].shape), dtype=done.dtype)
            self.advantage = np.empty((self.maxsize, *advantage[0].shape), dtype=advantage.dtype)
            self.log_probs = np.empty((self.maxsize, *log_probs[0].shape), dtype=log_probs.dtype)



        start_idx = self.size % self.maxsize
        end_idx   = start_idx + size

        if end_idx <= self.maxsize:
            # ----- (A) NO WRAP-AROUND -----
            self.observation[start_idx:end_idx]      = observation
            self.action[start_idx:end_idx]           = action
            self.reward[start_idx:end_idx]           = reward
            self.next_observation[start_idx:end_idx] = next_observation
            self.done[start_idx:end_idx]             = done
            self.advantage[start_idx:end_idx]        = advantage
            self.log_probs[start_idx:end_idx]        = log_probs
        else:
            # ----- (B) WRAP-AROUND -----
            # how many items fit until the end of the buffer
            first_part_len = self.maxsize - start_idx
            
            # fill from start_idx to the end of the buffer
            self.observation[start_idx:self.maxsize]      = observation[:first_part_len]
            self.action[start_idx:self.maxsize]           = action[:first_part_len]
            self.reward[start_idx:self.maxsize]           = reward[:first_part_len]
            self.next_observation[start_idx:self.maxsize] = next_observation[:first_part_len]
            self.done[start_idx:self.maxsize]             = done[:first_part_len]
            self.advantage[start_idx:self.maxsize]        = advantage[:first_part_len]
            self.log_probs[start_idx:self.maxsize]        = log_probs[:first_part_len]

            
            # the remainder wraps to the beginning of the buffer
            second_part_len = end_idx % self.maxsize  # = N - first_part_len
            self.observation[0:second_part_len]      = observation[first_part_len:]
            self.action[0:second_part_len]           = action[first_part_len:]
            self.reward[0:second_part_len]           = reward[first_part_len:]
            self.next_observation[0:second_part_len] = next_observation[first_part_len:]
            self.done[0:second_part_len]             = done[first_part_len:]
            self.advantage[0:second_part_len]        = advantage[first_part_len:]
            self.log_probs[0:second_part_len]        = log_probs[first_part_len:]

        
        # 4. Update total insertion count
        self.size += size

        
def buildMLP(input_dim, output_dim, network_shape):
    layers = []
    in_size = input_dim
    for a in network_shape:
        layers.append(nn.Linear(in_size, a))
        layers.append(nn.ReLU())
        in_size = a
    layers.append(nn.Linear(in_size, output_dim))

    return nn.Sequential(*layers)