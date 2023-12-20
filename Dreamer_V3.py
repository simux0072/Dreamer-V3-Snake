import torch
from torch import nn

class Dreamer_V3(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        
        self.input_size: int = input_dim[0] * input_dim[1]
        self.output_size: int = output_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=800)
        )
        
        self.distinct_distribution = nn.Sequential(
            nn.Linear(in_features=40, out_features=45),
            nn.LeakyReLU(),
            nn.Linear(in_features=45, out_features=50),
            nn.LeakyReLU(),
            nn.Linear(in_features=50, out_features=55),
            nn.LeakyReLU(),
            nn.Linear(in_features=55, out_features=60)
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
    
    def disc_distribution_pass(self, input: torch.Tensor):
        disc_dist = self.distinct_distribution(input.reshape((input.size(0), 20, 40)))
        
        
        
    def lstm_pass(self, input: torch.Tensor, action: torch.Tensor):
        input: torch.Tensor = torch.cat((input, action), dim=-1)
        lstm_out, (self.prev_h, self.prev_c): torch.Tensor = self.memory_module(input, (self.prev_h, self.prev_c))
        return lstm_out
    
    def forward(self, input):
        ...
