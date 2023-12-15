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
        
        self.prev_hidden = torch.zeros(())
        
        self.memory_module = nn.LSTM(input_size=1201, hidden_size=2048, num_layers=5, batch_first=True)
