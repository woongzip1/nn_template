import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

class generator(nn.Module):
    def __init__(   self, 
                    **kwargs):
        super().__init__()
        print('model initialize')
        
