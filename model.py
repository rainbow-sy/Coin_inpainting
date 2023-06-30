import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.inconv = nn.Sequential(  # 输入层网络
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.midconv = nn.Sequential(  # 中间层网络
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.outconv = nn.Sequential(  # 输出层网络
            nn.Conv2d(64, 3, 3, 1, 1),
        )

    def forward(self, x):
        x = self.inconv(x)
        x = self.midconv(x)
        x = self.outconv(x)

        return x

