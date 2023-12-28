import numpy

class Memory:
    def __init__(self):
        self.observations = None
        self.rewards = None
        self.ends = None
    
    def store_state(self, observation, reward: int, end: bool):
        self.observations