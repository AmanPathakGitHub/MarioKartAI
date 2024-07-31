import torch
import torch.nn as nn


# 224, 256
class KartModel(nn.Module):
    
    def __init__(self):
        super(KartModel, self).__init__()
        
        self.convLayer = nn.Sequential(
            nn.Conv2d(1, 6, 9),
            nn.ReLU(),
            nn.MaxPool2d(3, 3), 
            nn.Conv2d(6, 16, 9),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
            nn.Flatten(),
        )
        
        self.net = nn.Sequential(
            nn.Linear(7*7*16, 350),
            nn.ReLU(),
            nn.Linear(350, 100),
            nn.ReLU(),
            nn.Linear(100, 3)
        ) 
       
        
    def forward(self, x):
        out = self.convLayer(x)
        out = self.net(out)
        return out