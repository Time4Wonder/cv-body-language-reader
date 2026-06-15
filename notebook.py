import torch
import torch.nn as nn 

linear = nn.Linear(1,1)
with torch.no_grad():
    linear.weight.fill_(2.0)
    linear.bias.fill_(1.0)



