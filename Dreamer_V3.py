import torch
from torch import nn
from torch.distributions import Categorical

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
    def __init__(self, input_dim, output_dim, sym_net) -> None:
        super().__init__()
        
        self.input_size: int = input_dim[0] * input_dim[1]
        self.output_size: int = output_dim
        self.sym_net: Sym_net = sym_net
        
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
        
        self.prev_h = torch.zeros((1, 1, 2048))
        self.prev_c = torch.zeros((1, 1, 2048))
        
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
            nn.Linear(in_features=1024, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=1)
        )
        
        self.policy = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=3)
        )
        
        self.end = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )
    
    def encoder_pass(self, input:torch.Tensor):
        return self.encoder(input.flatten(start_dim=-2))
    
    def decoder_pass(self, disc_dist: torch.Tensor):
        return self.decoder(disc_dist.flatten(start_dim=-2))
    
    def create_dist_from_probs(self, input: torch.Tensor):
        disc_dist_probs: torch.Tensor = self.distinct_distribution(input.reshape((input.size(0), 20, 40))) # Output shape (input.size(0), 20, 60)
        dist = Categorical(probs=disc_dist_probs)
        index = dist.sample()
        base = torch.zeros(disc_dist_probs.size, dtype=torch.float32)
        for temp in range(0, base.size(-2)):
            base[0,temp, index[temp]] = 1
        return base
        
    def lstm_pass(self, input: torch.Tensor, action: torch.Tensor):
        input: torch.Tensor = torch.cat((input, action), dim=-1)
        lstm_out, (self.prev_h, self.prev_c): torch.Tensor = self.memory_module(input, (self.prev_h, self.prev_c))
        return lstm_out
    
    def dist_from_hidden(self, lstm_out:torch.Tensor):
        linear_dist:torch.Tensor = self.disc_from_memory(lstm_out)
        linear_dist.reshape((linear_dist.size(0), 20, 60))
        linear_dist = torch.softmax(linear_dist, dim=-1)
        return self.create_dist_from_probs(linear_dist)
    
    def dist_from_encoder(self, input:torch.Tensor):
        disc_dist_probs: torch.Tensor = self.distinct_distribution(input.reshape((input.size(0), 20, 40)))
        return self.create_dist_from_probs(disc_dist_probs)
    
    def forward(self, input):
        ...
