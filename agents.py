import numpy as np
import itertools
import torch
import torch.nn as nn
from torch import optim
from utils import buildMLP

cuda_available = torch.cuda.is_available()

device = torch.device("cuda" if cuda_available else "cpu")

class ValueNetwork(nn.Module):

    def __init__(self, observation_dim, lr = 3e-4, network_shape = [256, 256], tau = 1.0, target = False, update_target_step = 500):
        super().__init__()
        self.use_target = target
        self.tau = tau
        self.model = buildMLP(observation_dim, 1, network_shape).to(device)
        if self.use_target:
            self.target = buildMLP(observation_dim, 1, network_shape).to(device)
            self.updateTarget(self.tau)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr = lr
        )
        self.loss_fn = nn.MSELoss()
        self.step = 0
        self.gamma = 0.9
        self.target_update = update_target_step

    def updateTarget(self, tau):
        for param, target_param in zip(self.model.parameters(), self.target.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )


    def forward(self, observation) -> torch.Tensor:
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        
        observation = observation.to(device)
        
        assert isinstance(observation, torch.Tensor) == True

        pred = self.model(observation)

        return pred
    
    def update(self, observations, advantages, next_observation, reward, done):
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).float()
        observations = observations.to(device)

        if isinstance(advantages, np.ndarray):
            advantages = torch.from_numpy(advantages).float()
        advantages = advantages.to(device)

        if isinstance(next_observation, np.ndarray):
            next_observation = torch.from_numpy(next_observation).float()
        next_observation = next_observation.to(device)

        if isinstance(reward, np.ndarray):
            reward = torch.from_numpy(reward).float()
        reward = reward.to(device)

        done = torch.from_numpy(done).float().to(device)

        value_pred = self.model(observations)

        with torch.no_grad():
            if self.use_target:
                value_target = self.target(observations) + advantages
            else:
                value_target = self.model(observations) + advantages

        # with torch.no_grad():
        #     value_target = (1-done) * self.gamma * self.model(next_observation) + reward
        
        loss = self.loss_fn(value_pred, value_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.use_target:
            if self.step % self.target_update == 0:
                self.updateTarget()
        self.step+=1

        return loss


class Actor(nn.Module):

    def __init__(self, observation_dim, action_dim, lr=3e-4, discrete = True, network_shape = [256, 256], epsilon = 0.2, high = None, low = None):
        super().__init__()
        if discrete:
            self.logits = buildMLP(observation_dim, action_dim, network_shape).to(device)
            params = self.logits.parameters()
        else:
            self.mean = buildMLP(observation_dim, action_dim, network_shape).to(device)
            self.logstd = nn.Parameter(
                    torch.zeros(action_dim, dtype=torch.float32, device=device)            
                    )
            params = itertools.chain([self.logstd], self.mean.parameters())

        self.optimizer = optim.Adam(
            params,
            lr
        )

        self.discrete = discrete
        self.epsilon = epsilon
        if high is not None:
            self.high = torch.from_numpy(high).float()
            self.low = torch.from_numpy(low).float()
    
    @torch.no_grad()
    def get_action(self, observation):
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        assert observation.dim() == 1
        observation = observation.to(device)
        if self.discrete:
            logits = self.logits(observation)
            distribution = torch.distributions.Categorical(logits=logits)
            
        else:
            mean = self.mean(observation)
            distribution = torch.distributions.Normal(loc=mean, scale=torch.exp(self.logstd))
        sampled_action = distribution.sample()
        # if self.low is not None:
        #     sampled_action = torch.clamp(sampled_action, min=self.low, max=self.high)
        log_prob = distribution.log_prob(sampled_action).sum(dim=-1)
        return sampled_action.cpu().numpy(), log_prob.cpu().numpy()
    
    def forward(self, observation):
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        observation = observation.to(device)

        if self.discrete:
            logits = self.logits(observation)
            distribution = torch.distributions.Categorical(logits = logits)
        else:
            mean = self.mean(observation)
            # print("mean shape", mean.shape)
            distribution = torch.distributions.Normal(loc = mean, scale = torch.exp(self.logstd))
        
        return distribution



class PPOAgent(nn.Module):

    def __init__(self, observation_dim, action_dim, discrete = True, lr = 3e-4, gamma=0.99, lamb = 0.95, epsilon = 0.2, tau=1.0, value_network_shape = [256, 256], high=None, low= None):
        super().__init__()
        self.value = ValueNetwork(observation_dim=observation_dim, lr=lr, network_shape=value_network_shape)
        self.actor = Actor(observation_dim=observation_dim, lr=lr, action_dim=action_dim, discrete=discrete, high = high, low= low)
        self.tau = tau
        self.lamb = lamb
        self.gamma = gamma
        self.high = high
        self.low = low
        self.num_val = 10
        self.discrete = discrete

        self.epsilon = epsilon

        self.prev_prob = None

    def calculate_advantage(self, observation, next_observation, rewards, dones):
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        observation = observation.to(device)
        size = observation.shape[0]
        

        advantages = np.zeros(size+1)
        
        for i in reversed(range(size)):
            if dones[i]:
                delta = rewards[i] - self.value.forward(observation[i]).detach().cpu().numpy().squeeze()
            else:
                delta = rewards[i] + self.gamma*self.value.forward(next_observation[i]).detach().cpu().numpy().squeeze() - self.value.forward(observation[i]).detach().cpu().numpy().squeeze()

            advantages[i] = delta + self.gamma*self.lamb * advantages[i+1]

        advantages = advantages[:-1]
        return advantages

    def get_action(self, observation):
        return self.actor.get_action(observation)
    
    def update_actor(self, observation, actions, advantages, old_log_probs):
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        observation = observation.to(device)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()
        actions = actions.to(device)
        if isinstance(advantages, np.ndarray):
            advantages = torch.from_numpy(advantages).float()
        advantages = advantages.to(device)
        if isinstance(old_log_probs, np.ndarray):
            old_log_probs = torch.from_numpy(old_log_probs).float()
        old_log_probs = old_log_probs.to(device)

        if actions.dim() != 2:
            actions = actions.unsqueeze(1)

        dist = self.actor.forward(observation)
        if self.discrete:
            log_probs = dist.log_prob(actions)
        else:
            log_probs = dist.log_prob(actions).sum(-1)
        
        r = (log_probs - old_log_probs).exp()

        clipped = torch.clamp(r, min = 1.0 - self.epsilon, max=1.0 + self.epsilon)

        clipped_obj = clipped * advantages
        unclipped_obj = r * advantages

        min_obj = torch.min(clipped_obj, unclipped_obj)
        loss = -min_obj.mean()

        self.actor.optimizer.zero_grad()
        loss.backward()
        self.actor.optimizer.step()
        return loss

    def learn(self, sample):
        """
        passed as 
         "observations": self.observation[rand_indices],
            "actions": self.action[rand_indices],
            "rewards": self.reward[rand_indices],
            "next_observations": self.next_observation[rand_indices],
            "dones": self.done[rand_indices],
            "advantages": self.advantage[rand_indices],
            "log_probs": self.log_probs[rand_indices],
        """
        observations = sample["observations"]
        actions = sample["actions"]
        rewards = sample["rewards"]
        next_observations = sample["next_observations"]
        dones = sample["dones"]
        advantages = sample["advantages"]
        log_probs = sample["log_probs"]
        if isinstance(advantages, np.ndarray):
            advantages = torch.from_numpy(advantages).float()
        advantages = advantages.to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # for a in range(self.num_val):
        val_loss = self.value.update(observations, advantages, next_observations, rewards, dones)

        actor_loss = self.update_actor(observations,actions, advantages, log_probs)

        return val_loss, actor_loss

    

    
        