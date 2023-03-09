import torch.nn as nn
import torch.nn.functional as F


class DiscriminatorTrain(nn.Module):
    def __init__(self):
        super(DiscriminatorTrain, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.Sequential(
            nn.Conv2d(128, 1, 1, 1, 0),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        # B H W -> B 1 H W
        x = x.unsqueeze(1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = x.squeeze(1).squeeze(1).squeeze(1)
        x = F.sigmoid(x)
        return x


class DiscriminatorFinetuning(nn.Module):
    def __init__(self):
        super(DiscriminatorFinetuning, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.Sequential(
            nn.Conv2d(128, 1, 1, 1, 0),
        )

    def forward(self, x):
        # B H W -> B 1 H W
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = F.sigmoid(x)
        return x
