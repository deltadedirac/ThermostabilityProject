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
    
class CNNpooling(torch.nn.Module):
    
    def __init__(self, dim_seq, dim_embed):
        self.channel_emb = dim_embed
        self.seq_dim = dim_seq
        
        self.cnn_pooling = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.channel_emb, out_channels=self.channel_emb, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.channel_emb, out_channels=1, kernel_size=1),
            torch.nn.ReLu()           
        )
    
    def forward(self,x):
        x = self.cnn_pooling(x)
        return x
    
    
#class tunning_pool_and_FFNN(torch.nn.Module):
    
#    def __init__(self, dim_seq, dim_embed):


#for padding
def padding_tensor( a, size_end ,val_pad=0):
    a_size = a.shape
    f_size = torch.zeros(a_size[0], size_end[1], size_end[2])
    
    f_size[:, 0:a_size[1]] = a
    return f_size
    
    