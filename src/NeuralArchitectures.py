import torch, torch.nn
import numpy as np

class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, input):
        return input.view(input.size(0), -1)   

class regressionHead(torch.nn.Module):

    def __init__(self, shape_embedding):
        super(regressionHead, self).__init__()
        self.shape_emb = np.prod(shape_embedding)
        #self.input_shape = input_shape # pos0 = #channels, pos1 = #diagonal comps, or viseversa


        self.FFNN = torch.nn.Sequential(
            Flatten(),
            torch.nn.Linear(self.shape_emb, 512),
            torch.nn.Sigmoid(), #nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.Sigmoid(), #nn.ReLU(),
            torch.nn.Linear(512, 1),
        )
        
    def forward(self, x):
        z = self.FFNN(x)
        return z