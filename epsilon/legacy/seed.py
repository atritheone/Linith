from pv_model import LinithPVNet
import torch

net = LinithPVNet.from_value_only("linith_from_hard_200.pt")
torch.save(net.state_dict(), "linith_pv_init.pt")
print("saved linith_pv_init.pt")