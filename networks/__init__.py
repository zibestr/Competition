from abc import ABC, abstractmethod
from torch import nn, cat, Tensor


class AbstractCNN(ABC, nn.Module):
    @abstractmethod
    def __init__(self, n_channels: int, filters_shape: tuple[int, int],
                 image_shape: tuple[int, int]):
        self.normalization = nn.BatchNorm2d(n_channels, affine=False)
        self.convolution = nn.Sequential()
        self.flatten = nn.Flatten()
        # add 1 percentron for distance parameter
        self.linear = nn.Sequential()

    def forward(self, x: Tensor):
        image, distance = self.normalization(x[0]), x[1]
        conv_outp = self.flatten(self.convolution(image))
        linear_input = cat((conv_outp, distance), dim=1)
        return self.linear(linear_input)
