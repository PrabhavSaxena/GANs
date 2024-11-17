import torch
import torch.nn as nn
from matplotlib import pyplot
import numpy as np

img_shape = [1, 28, 28]

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        noise_shape = (100,)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=100, out_features=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(256, momentum=0.8))   
        
        self.fc2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(512, momentum=0.8))    

        self.fc3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(1024, momentum=0.8))       
        
        self.fc4 = nn.Sequential(
            nn.Linear(1024, np.prod(img_shape)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = x.reshape(x.size(0), 1, 28, 28) 
        return x

model = Generator().cpu()
PATH = '/home/prabhav/ml_tut/generator_model.pth'
model.load_state_dict(torch.load(PATH)) 
model.eval()

# Generate vector
vector = torch.rand(1, 100) 

X = model(vector)

# Plot the result
pyplot.imshow(X[0, 0, :, :].detach().cpu().numpy(), cmap='gray_r') 
pyplot.show()
