
from src.networking import Connection
from src.replay_memory import Replay_Memory

import time
import io

from PIL import Image

import random
import numpy as np

class Agent:
    
    def __init__(self, ip, port, max_steps, max_replay_len):
        self.connection = Connection(ip, int(port))
        self.connection.sendData(str(max_steps))
        
        self.memory = Replay_Memory(int(max_replay_len))
       
    
    def run(self):

        state = self.connection.recieveScreenShot()
        
        while True:
            
            # actions
            actions = self.sampleActions()
            
            action = np.argmax(actions)
            self.connection.sendData(str(action))
            
            next_state = self.connection.recieveScreenShot()
            
            image = Image.open(io.BytesIO(next_state))
            
            msg = self.connection.recieveData(5)
            termination, reward = msg.split()
            
            self.memory.append((state, action ,next_state, int(reward), bool(int(termination))))
            
            state = next_state
            self.connection.sendData("FRAME DONE!")
    
        
    def sampleActions(self):
        actions = [0, 0, 0]
        actions[random.randint(0, 2)] = 1
        return actions