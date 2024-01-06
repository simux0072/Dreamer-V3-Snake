import numpy

class Memory:
    def __init__(self):
        self.reset()
    
    def store_state(self, observation, reward: int, end: bool):
        self.observations.append(observation)
        self.rewards.append(reward)
        self.ends.append(end)
    
    def reset(self):
        self.observations: list(numpy.ndarray) = []
        self.rewards: list(numpy.ndarray) = []
        self.ends: list(numpy.ndarray) = []
    
    def get_memory(self):
        return numpy.array(self.observations), numpy.array(self.rewards), numpy.array(self.ends)