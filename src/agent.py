
from src.networking import Connection
from src.replay_memory import Replay_Memory
from src.model import KartModel

import time
import io
import random
import itertools

from PIL import Image

import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:
    
    def __init__(self, ip, ports, config):
        
        MAX_STEPS = config.get("Training", "MAX_STEPS")
        REPLAY_MEMORY_SIZE = config.get("Training", "REPLAY_MEMORY_SIZE")
        
         
        self.epslion = float(config.get("Training", "EPSILON_INIT"))
        self.epslion_decay = float(config.get("Training", "EPSILON_DECAY"))
        self.epslion_min = float(config.get("Training", "EPSILON_MIN"))
        
        self.discount_factor = float(config.get("Training", "DISCOUNT_FACTOR"))
        self.mini_batch_size = int(config.get("Training", "MINI_BATCH_SIZE"))
        self.network_sync_rate = int(config.get("Training", "NETWORK_SYNC_RATE"))
        
        
        self.connections = []
        
        for port in ports: 
            connection = Connection(ip, int(port))
            connection.sendData(str(MAX_STEPS))
            self.connections.append(connection)
            
        self.memory = Replay_Memory(int(REPLAY_MEMORY_SIZE))
        
        self.model = KartModel().to(device)
        self.target_model = KartModel().to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimiser = optim.Adam(self.model.parameters(), lr=float(config.get("Training", "LEARNING_RATE")))
        self.loss_fn = nn.MSELoss()
        
        self.writer = SummaryWriter()
       
    
    
    def run(self):
        
        # Lowest possible reward * 1000 is Max_steps
        highest_reward = -15 * 1000
        
        for episode in itertools.count():
            
          
            termination = False
            
            step = 0
            
            total_reward = 0
            
            while not termination:
                
                
                if random.random() < self.epslion:
                    actions = self.sampleActions()
                else:
                    actions = self.model(state.unsqueeze(0)).squeeze()

                action = torch.argmax(actions).to(device)
                
                self.connection.sendData(str(action.item()))
                
                image = self.connection.recieveScreenShot()
                next_state = self.convertImage(image).to(device)
                
                msg = self.connection.recieveData(5)
                termination, reward = msg.split()
                
                termination = bool(int(termination))
                reward = int(reward)
                
                total_reward += reward
                
                
                self.memory.append((state, action, next_state, reward, termination))
                
                # train this frame
                self.optimise([(state, action, next_state, reward, termination)])
                
                if step % self.network_sync_rate == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
                    step = 0
                
                state = next_state
                self.connection.sendData("FRAME DONE!")
            
            
            # decay epsilon
            self.epslion = max(self.epslion*self.epslion_decay, self.epslion_min)
            
            mini_batch = self.memory.sample(self.mini_batch_size)
            self.optimise(mini_batch)
            
            if episode % 10 == 0: 
                self.writer.add_scalar("Reward", total_reward, episode)
                self.writer.add_scalar("Epslion", self.epslion, episode)
            
            if total_reward > highest_reward:
                highest_reward = total_reward
                torch.save(self.model.state_dict(), "models/best.pth")
    
    def getState(self, connection):
        image = connection.recieveScreenShot()
            
        return self.convertImage(image).to(device)

    def sampleActions(self):
        actions = [0, 0, 0]
        actions[random.randint(0, 2)] = 1
        return torch.tensor(actions)
    
    def convertImage(self, data):
        
        img = Image.open(io.BytesIO(data)).convert('L').resize((96, 96), Image.Resampling.NEAREST)
        img = transforms.ToTensor()(img)
        
        return img
    
    def optimise(self, mini_batch):
        
        states, actions, next_states, rewards, terminations = zip(*mini_batch)
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards, dtype=torch.int64).to(device)
        terminations = torch.tensor(terminations, dtype=torch.int64).to(device)
        
        current_q = self.model(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.discount_factor * self.target_model(next_states).max(dim=1)[0]
            
        loss = self.loss_fn(current_q, target_q)
        
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()