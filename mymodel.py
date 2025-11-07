import torch
from torch import nn

class SimpleMLP(nn.Module):
    """
    简单的MLP
    """
    def __init__(self, num_hidden):
        super().__init__()
        self.model = nn.Sequential(
            nn.LazyLinear(num_hidden),
            nn.Tanh(),
            nn.LazyLinear(1)
        )

    def forward(self, X):
        return self.model(X)
    
class WhatNet(nn.Module):
    """
    三MLP并联
    """
    def __init__(self):
        super().__init__()
        self.sub1 = nn.Sequential(
            nn.LazyLinear(32),
            nn.Tanh(),
            nn.LazyLinear(1)
        )

        self.sub2 = nn.Sequential(
            nn.LazyLinear(64),
            nn.Tanh(),
            nn.LazyLinear(32),
            nn.Tanh(),
            nn.LazyLinear(1)
        )

        self.sub3 = nn.Sequential(
            nn.LazyLinear(96),
            nn.Tanh(),
            nn.LazyLinear(64),
            nn.Tanh(),
            nn.LazyLinear(32),
            nn.Tanh(),
            nn.LazyLinear(1)
        )

    def forward(self, X):
        y1 = self.sub1(X)
        y2 = self.sub2(X)
        y3 = self.sub3(X)
        return (y1 + y2 + y3) / 3
    
if __name__ == "__main__":
    net = WhatNet()
    X = torch.randn((10, 11))
    y = net(X)
    print(y.shape)