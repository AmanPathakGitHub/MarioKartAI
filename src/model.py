import torch
import torch.nn as nn


# 224, 256
class KartModel(nn.Module):
    
    def __init__(self):
        super(KartModel, self).__init__()
        
        # Nvidia driving model
        self.convLayer = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ReLU(),
           
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ReLU(),
           
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ReLU(),
            
            
            nn.Conv2d(48, 64, 3),
            nn.ReLU(),
           
           
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
           
            nn.Flatten()
        )
        
        self.net = nn.Sequential(
            nn.Linear(1152, 100),
            nn.ReLU(),
            
            nn.Linear(100, 50),
            nn.ReLU(),
            
            nn.Linear(50, 10),
            nn.ReLU(),
            
            nn.Linear(10, 3)
        ) 
       
        
    def forward(self, x):
        out = self.convLayer(x)
        out = self.net(out)
        return out