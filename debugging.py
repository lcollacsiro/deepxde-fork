import torch
import torch.nn as nn

device = "cuda"

model = nn.Sequential(
    nn.Linear(10, 100),
    nn.ReLU(),
    nn.Linear(100, 1)
).to(device)

x = torch.randn(1000, 10, device=device)

y = model(x)

loss = y.pow(2).mean()
loss.backward()

print("Success")