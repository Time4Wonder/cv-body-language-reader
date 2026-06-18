import torch
import torch.nn as nn 

linear = nn.Linear(1,1)
with torch.no_grad():
    linear.weight.fill_(2.0)
    linear.bias.fill_(1.0)
    
x = torch.tensor([5.0])
loss = linear(x).sum()

print(f"Vor backward: {linear.weight.grad}")
loss.backward()
print(f"Nach backward: {linear.weight.grad}")


print(loss.grad_fn)
print(linear.weight.grad_fn)

print(linear.parameters())
