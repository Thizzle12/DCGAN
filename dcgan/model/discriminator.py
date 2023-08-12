from torch import nn
from torchvision.models import resnet50
import torchvision


class Discriminator2(nn.Module):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.resnet = resnet50()

        self.resnet.fc = nn.Linear(
            512 * torchvision.models.resnet.BasicBlock.expansion,
            2,
        )

    def forward(self, inputs):
        return self.resnet(inputs)


class Discriminator(nn.Module):
    def __init__(
        self,
        n_features: int = 64,
        n_channels: int = 3,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_features,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=n_features,
                out_channels=n_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(n_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=n_features * 2,
                out_channels=n_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(n_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=n_features * 4,
                out_channels=n_features * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(n_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=n_features * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)
