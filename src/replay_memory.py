
from collections import deque
from threading import Lock

import random


class Replay_Memory:
    
    def __init__(self, maxlen):
        
        self.memory = deque([], maxlen)
        self.lock = Lock()
        
    def append(self, x):
        with self.lock:
            self.memory.append(x)
    
    def sample(self, size):
        if size > len(self.memory):
            return self.memory
        
        return random.sample(self.memory, size)
    
    def __len__(self):
        return len(self.memory)
        
        