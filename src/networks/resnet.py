from src.networks._network import AbstractCNN
from torch import nn, Tensor


class ResudialBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, kernel_size: int = 3):
        super().__init__()

        self.downsample = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1, stride=stride)

        self.resudial_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        indentity = x
        x = self.resudial_block(x)
        if self.downsample is not None:
            indentity = self.downsample(indentity)
        return nn.functional.relu(x + indentity)


class ResNet34(AbstractCNN):
    def __init__(self, n_channels: int = 3,
                 image_shape: tuple[int, int] = (440, 440)):
        super().__init__(n_channels, image_shape)
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=7,
                      stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *[ResudialBlock(in_channels=64, out_channels=64)
              for _ in range(3)],
            ResudialBlock(in_channels=64, out_channels=128, stride=2),
            *[ResudialBlock(in_channels=128, out_channels=128)
              for _ in range(3)],
            ResudialBlock(in_channels=128, out_channels=256, stride=2),
            *[ResudialBlock(in_channels=256, out_channels=256)
              for _ in range(5)],
            ResudialBlock(in_channels=256, out_channels=512, stride=2),
            *[ResudialBlock(in_channels=512, out_channels=512)
              for _ in range(2)],
            nn.AdaptiveAvgPool2d(1)
        )
        flatten_out_size = 512
        self.linear = nn.Sequential(
            nn.Linear(in_features=flatten_out_size + 1, out_features=1)
        )


class ResNet18(AbstractCNN):
    def __init__(self, n_channels: int = 3,
                 image_shape: tuple[int, int] = (440, 440)):
        super().__init__(n_channels, image_shape)
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=7,
                      stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *[ResudialBlock(in_channels=64, out_channels=64)
              for _ in range(2)],
            ResudialBlock(in_channels=64, out_channels=128, stride=2),
            ResudialBlock(in_channels=128, out_channels=128),
            ResudialBlock(in_channels=128, out_channels=256, stride=2),
            ResudialBlock(in_channels=256, out_channels=256),
            ResudialBlock(in_channels=256, out_channels=512, stride=2),
            ResudialBlock(in_channels=512, out_channels=512),
            nn.AdaptiveAvgPool2d(1)
        )
        flatten_out_size = 512
        self.linear = nn.Sequential(
            nn.Linear(in_features=flatten_out_size + 1, out_features=1)
        )
