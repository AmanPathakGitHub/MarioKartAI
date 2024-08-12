
from src.networking import Connection
from src.replay_memory import Replay_Memory
from src.model import KartModel

import time
import io
import random
import itertools
import traceback
from threading import Thread, Lock
from multiprocessing.pool import ThreadPool

from PIL import Image

import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:
    
    def __init__(self, ip, port, num_env, config):
        
        MAX_STEPS = config.get("Training", "MAX_STEPS")
        REPLAY_MEMORY_SIZE = config.get("Training", "REPLAY_MEMORY_SIZE")
        
         
        self.epslion = float(config.get("Training", "EPSILON_INIT"))
        self.epslion_decay = float(config.get("Training", "EPSILON_DECAY"))
        self.epslion_min = float(config.get("Training", "EPSILON_MIN"))
        
        self.discount_factor = float(config.get("Training", "DISCOUNT_FACTOR"))
        self.mini_batch_size = int(config.get("Training", "MINI_BATCH_SIZE"))
        self.network_sync_rate = int(config.get("Training", "NETWORK_SYNC_RATE"))
        
        
        self.connections = []
        
        #TODO: Needs to open on seperate threads this is too slow
        for i in range(num_env):
            connection = Connection(ip, int(port) + i)
            self.connections.append(connection)

        
        for connection in self.connections:
            connection.thread.join()
            connection.sendData(str(MAX_STEPS))
            
        self.memory = Replay_Memory(int(REPLAY_MEMORY_SIZE))
        
        self.model_lock = Lock()
        self.model = KartModel().to(device)
        # self.model.load_state_dict(torch.load("models/last.pth"))
        
        self.target_model = KartModel().to(device)
        # self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimiser = optim.Adam(self.model.parameters(), lr=float(config.get("Training", "LEARNING_RATE")))
        self.optimiser.load_state_dict(torch.load("models/last-optim.pth"))
        self.loss_fn = nn.MSELoss()
      
        self.writer = SummaryWriter()
        
        self.total_reward = 0
        self.num_env = num_env
        
        self.starting_episode = 0
        
        
    
    
    def run(self):
        
        # Lowest possible reward * 1000 is Max_steps
        highest_reward = -15 * 1000
        
        pool = ThreadPool(processes=self.num_env)
        
        for episode in itertools.count(self.starting_episode):
            self.total_reward = 0
            
            pool.map(self.startEnvLoop, [c for c in self.connections])
            
            # decay epsilon
            self.epslion = max(self.epslion*self.epslion_decay, self.epslion_min)
            
            mini_batch = self.memory.sample(self.mini_batch_size)
            self.optimise(mini_batch, episode=episode)
            
            if episode % 10 == 0:
                self.writer.add_scalar("Reward", self.total_reward / self.num_env, episode)
                self.writer.add_scalar("Epslion", self.epslion, episode)
                
            # if episode % self.network_sync_rate == 0:
            #     self.target_model.load_state_dict(self.model.state_dict())
            
            if self.total_reward > highest_reward:
                highest_reward = self.total_reward
                # torch.save(self.model.state_dict(), "models/best.pth")
                # torch.save(self.optimiser.state_dict(), "models/best-optim.pth")
                self.saveCheckpoint(episode)
                

        
    # TODO this needs a better name
    def startEnvLoop(self, connection):
        
        state = self.getState(connection)
          
        termination = False
        step = 0 

            
        while not termination:
            
            
            if random.random() < self.epslion:
                actions = self.sampleActions()
            else:
                with self.model_lock:
                    self.model.eval()
                    with torch.no_grad():
                        actions = self.model(state.unsqueeze(0)).squeeze()
                    self.model.train()
                
            action = torch.argmax(actions).to(device)
            
            connection.sendData(str(action.item()))
            
            next_state = self.getState(connection)
            
            msg = connection.recieveData(5)
            termination, reward = msg.split()
            
            termination = bool(int(termination))
            reward = int(reward)
            
            
            
            self.memory.append((state, action, next_state, reward, termination))
            
            # train this frame
            with self.model_lock:
                self.total_reward += reward
                self.optimise([(state, action, next_state, reward, termination)])
            
            step += 1
            
            if step % self.network_sync_rate == 0:
                with self.model_lock:
                    self.target_model.load_state_dict(self.model.state_dict())
                    step = 0
            
            state = next_state
            connection.sendData("FRAME DONE!")


                        
    
    def getState(self, connection):
        image, err = connection.recieveScreenShot()
        
        if err:
            # locking the model so the weights don't get changed from another thread while saving
            with self.model_lock:
                torch.save(self.model.state_dict(), "models/last.pth")
                torch.save(self.optimiser.state_dict(), "models/last-optim.pth")
            exit(-1)
            
        return self.convertImage(image).to(device)

    def sampleActions(self):
        actions = [0, 0, 0]
        actions[random.randint(0, 2)] = 1
        return torch.tensor(actions)
    
    def convertImage(self, data):
        
        # halving the image size
        # also colors are already normalised
        img = Image.open(io.BytesIO(data)).resize((200, 66), Image.NEAREST)
        img = transforms.ToTensor()(img)
        
        return img
    
    def optimise(self, mini_batch, episode=0):
        
        states, actions, next_states, rewards, terminations = zip(*mini_batch)
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards, dtype=torch.int64).to(device)
        terminations = torch.tensor(terminations, dtype=torch.int64).to(device)
        
        current_q = self.model(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.discount_factor * self.target_model(next_states).max(dim=1)[0]
            
        loss = self.loss_fn(current_q, target_q)
        
        if episode != 0:
            self.writer.add_scalar("Loss", loss.item(), episode)
        
        self.optimiser.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimiser.step()
    
    # episode number
    # epsilon
    # optimiser weights
    # model weights
    
    def saveCheckpoint(self, epsiode):
        torch.save({'episode' : epsiode,
                    'model_weights' : self.model.state_dict(),
                    'optimiser_weights' : self.optimiser.state_dict(),
                    'epsilon' : self.epslion}, "./checkpoints/recent-checkpoint.pth")
    
    def loadCheckpoint(self, filepath):
        try:
            checkpoint = torch.load(filepath)
        
            self.starting_episode = checkpoint['episode']
            self.epslion = checkpoint['epsilon']
            self.model.load_state_dict(checkpoint['model_weights'])
            self.optimiser.load_state_dict(checkpoint['optimiser_weights'])
        except:
            print("Invalid or no checkpoint provided")
                