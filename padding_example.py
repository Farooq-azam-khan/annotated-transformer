from torch.nn import functional as F
import torch

t = torch.randn(4, 2)
print(F.pad(t, [1, 1, 0, 1]))
