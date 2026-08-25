import torch
from pv_model import LinithPVNet  # your model class

model = LinithPVNet()
torch.save(model.state_dict(), "epsilon_0.1.pt")
print("Created blank model - epsilon_0.1.pt")