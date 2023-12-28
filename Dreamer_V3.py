import torch
from torch import nn
from torch.distributions import Categorical
from numba.cuda import njit

class Sym_net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
        self.symlog = nn.Sequential(
            nn.Linear(in_features=1, out_features=10),
            nn.LeakyReLU(),
            nn.Linear(in_features=10, out_features=15),
            nn.LeakyReLU(),
            nn.Linear(in_features=15, out_features=10),
            nn.LeakyReLU(),
            nn.Linear(in_features=10, out_features=1)
        )
        
        self.symexp = nn.Sequential(
            nn.Linear(in_features=1, out_features=10),
            nn.LeakyReLU(),
            nn.Linear(in_features=10, out_features=15),
            nn.LeakyReLU(),
            nn.Linear(in_features=15, out_features=10),
            nn.LeakyReLU(),
            nn.Linear(in_features=10, out_features=1)
        )
        
    def symlog(self, input: torch.Tensor):
        return self.symlog(input)
    
    def symexp(self, input: torch.Tensor):
        return self.symexp(input)

class Dreamer_V3(nn.Module):
    def __init__(self, input_dim, output_dim, sym_net, length) -> None:
        super().__init__()
        
        self.input_size: int = input_dim[0] * input_dim[1]
        self.output_size: int = output_dim
        self.sym_net: Sym_net = sym_net
        self.simul_len = length
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=800)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(in_features=1200, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=self.input_size)
        )
        
        self.distinct_distribution = nn.Sequential(
            nn.Linear(in_features=40, out_features=45),
            nn.LeakyReLU(),
            nn.Linear(in_features=45, out_features=50),
            nn.LeakyReLU(),
            nn.Linear(in_features=50, out_features=55),
            nn.LeakyReLU(),
            nn.Linear(in_features=55, out_features=60),
            nn.Softmax(dim=-1)
        )
        
        self.reset()
        
        self.memory_module = nn.LSTM(input_size=1201, hidden_size=2048, num_layers=5, batch_first=True)
        
        self.disc_from_memory = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1536),
            nn.LeakyReLU(),
            nn.Linear(in_features=1536, out_features=1536),
            nn.LeakyReLU(),
            nn.Linear(in_features=1536, out_features=1200)
        )
        
        self.value = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=1)
        )
        
        self.reward = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=1)
        )
        
        self.policy = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=self.output_size),
            nn.Softmax()
        )
        
        self.end = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=2),
            nn.Sigmoid()
        )
    
    def reset(self):
        self.prev_h = torch.zeros((1, 1, 2048))
        self.prev_c = torch.zeros((1, 1, 2048))
        self.prev_disc_dist = torch.zeros((1, 1200))
        self.prev_action = torch.zeros((1, 1))
    
    def symlog_pass(self, input: torch.Tensor):
        return self.sym_net.symlog(input.unsqueeze(dim=-1)).squeeze()
    
    def symexp_pass(self, input: torch.Tensor):
        return self.sym_net.symexp(input.unsqueeze(dim=-1)).squeeze()
    
    def encoder_pass(self, input:torch.Tensor):
        input: torch.Tensor = self.symlog_pass(input.flatten(start_dim=-2))
        return self.encoder(input)
    
    def decoder_pass(self, disc_dist: torch.Tensor):
        return self.decoder(disc_dist.flatten(start_dim=-2))
    
    @njit
    def create_dist_from_probs(self, input: torch.Tensor):
        disc_dist_probs: torch.Tensor = self.distinct_distribution(input.reshape((input.size(0), 20, 40))) # Output shape (input.size(0), 20, 60)
        dist = Categorical(probs=disc_dist_probs)
        index = dist.sample()
        base = torch.zeros(disc_dist_probs.size(), dtype=torch.float32)
        for temp in range(0, base.size(-2)):
            base[0,temp, index[temp]] = 1
        return base.flatten(dim=-2).unsqeeze(dim=1)
        
    def lstm_pass(self, input: torch.Tensor, action: torch.Tensor):
        input: torch.Tensor = torch.cat((input, action), dim=-1)
        lstm_out, (self.prev_h, self.prev_c) = self.memory_module(input, (self.prev_h, self.prev_c))
        return lstm_out
    
    def dist_from_hidden(self, lstm_out:torch.Tensor):
        linear_dist:torch.Tensor = self.disc_from_memory(lstm_out)
        linear_dist.reshape((linear_dist.size(0), 20, 60))
        linear_dist = torch.softmax(linear_dist)
        return self.create_dist_from_probs(linear_dist)
    
    def dist_from_encoder(self, input:torch.Tensor):
        disc_dist_probs: torch.Tensor = self.distinct_distribution(input.reshape((input.size(0), 20, 40)))
        return self.create_dist_from_probs(disc_dist_probs)
    
    def reward_pass(self, input: torch.Tensor):
        return self.reward(input)
    
    def value_pass(self, input: torch.Tensor):
        return self.value(input)
    
    def end_pass(self, input: torch.Tensor):
        return torch.argmax(self.end(input), dim=-1)
    
    def policy_pass(self, input: torch.Tensor):
        policy_logit = self.policy(input) # Shape (1, 1, X)
        dist = Categorical(probs=policy_logit)
        action = dist.sample()
        return action, dist
    
    def forward_eval(self, input: torch.Tensor):
        encoded: torch.Tensor = self.encoder_pass(input) # Shape: (1, 1, X)
        disc_dist: torch.Tensor = self.dist_from_encoder(encoded) # Shape: (1, 1, X)
        hidden_state = self.lstm_pass(self.prev_disc_dist, self.prev_action) # Shape: (1, 1, 2048)
        action, _ = self.policy_pass(hidden_state) # Shape: (1, 1, 1)
        self.prev_action = action
        self.prev_disc_dist = disc_dist
        return action
    
    def forward_train_model(self, observation: torch.Tensor, action: torch.Tensor):
        encoded: torch.Tensor = self.encoder_pass(observation) # Shape: (1, 1, X)
        disc_dist: torch.Tensor = self.dist_from_encoder(encoded) # Shape: (1, 1, X)
        decoded: torch.Tensor = self.decoder_pass(disc_dist) # Shape: (1, 1, observation_size)
        hidden_state = self.lstm_pass(self.prev_disc_dist, action) # Shape: (1, 1, 2048)
        disc_dist_from_hidden = self.dist_from_hidden(hidden_state) # Shape (1, 1, X)
        reward: torch.Tensor = self.reward_pass(hidden_state) # Shape (1, 1, X)
        end: torch.Tensor = self.end_pass(hidden_state) # (1, 1, X)
        
        self.prev_disc_dist = disc_dist
        return disc_dist, disc_dist_from_hidden, decoded, reward, end
    
    @njit
    def forward_train_actor_critic(self, observation: torch.Tensor):
        
        encoded: torch.Tensor = self.encoder_pass(observation)
        disc_dist: torch.Tensor = self.dist_from_encoder(encoded)
        hidden_state: torch.Tensor = self.lstm_pass(self.prev_disc_dist, self.prev_action)
        temp_action, temp_action_dist: torch.Tensor = self.policy_pass(hidden_state)
        
        actions: list[torch.Tensor] = [temp_action]
        action_dists: list[torch.Tensor] = [temp_action_dist]
        values: list[torch.Tensor] = [self.value_pass(hidden_state)]
        ends: list[torch.Tensor] = [self.end_pass(hidden_state)]
        rewards: list[torch.Tensor] = [self.reward_pass(hidden_state)]

        for index in range(1, self.simul_len):
            self.prev_disc_dist = disc_dist
            self.prev_action = actions[index]
            if ends[index] == 0:
                return torch.cat(actions, dim=1), torch.cat(action_dists, dim=1), torch.cat(values, dim=1), torch.cat(rewards, dim=1), torch.cat(ends, dim=1)
            
            hidden_state = self.lstm_pass(self.prev_disc_dist, self.prev_action)
            temp_action, temp_action_dist = self.policy_pass(hidden_state)
            actions.append(temp_action)
            action_dists.append(temp_action_dist)
            values.append(self.value_pass(hidden_state))
            ends.append(self.end_pass(hidden_state))
            rewards.append(self.reward_pass(hidden_state))
        
        return torch.cat(actions, dim=1), torch.cat(action_dists, dim=1), torch.cat(values, dim=1), torch.cat(rewards, dim=1), torch.cat(ends, dim=1)