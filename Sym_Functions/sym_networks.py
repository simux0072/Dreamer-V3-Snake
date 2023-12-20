import torch
from torch import nn

class sym_net(nn.Module):
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
        
    def forward_symlog(self, input: torch.Tensor):
        return self.symlog(input)
    
    def forward_symexp(self, input: torch.Tensor):
        return self.symexp(input)