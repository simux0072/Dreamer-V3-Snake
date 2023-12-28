import dreamer_V3
import torch
import numba

class Agent:
    def __init__(self, Dreamer_model, memory, lr):
        self.net: dreamer_V3.Dreamer_V3 = Dreamer_model
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.memory = memory