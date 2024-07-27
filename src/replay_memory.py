
from collections import deque

import random

class Replay_Memory:
    
    def __init__(self, maxlen):
        
        self.memory = deque([], maxlen)
        
    def append(self, x):
        self.memory.append(x)
    
    def sample(self, size):
        return random.sample(self.memory, size)
    
    def __len__(self):
        return len(self.memory)
        
        