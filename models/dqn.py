import torch
import torch.nn as nn

class DQN(nn.Module):

    def __init__(self, num_actions):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),

            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU(),

            nn.Flatten()
        )

        self.fc = nn.Sequential(

            nn.Linear(64*7*7,512),
            nn.ReLU(),

            nn.Linear(512,num_actions)
        )

    def forward(self,x):

        x=self.cnn(x)
        return self.fc(x)