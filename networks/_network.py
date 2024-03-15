from abc import ABC, abstractmethod
from torch import nn, cat, Tensor


class AbstractCNN(ABC, nn.Module):
    @abstractmethod
    def __init__(self, n_channels: int = 3,
                 image_shape: tuple[int, int] = (440, 440)):
        super().__init__()
        self.normalization = nn.BatchNorm2d(n_channels, affine=False)
        self.convolution = nn.Sequential()
        self.flatten = nn.Flatten()
        # add 1 percentron for distance parameter
        self.linear = nn.Sequential()

    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        image, distance = self.normalization(x[0]), x[1].view(-1, 1)
        conv_outp = self.flatten(self.convolution(image))
        linear_input = cat((conv_outp, distance), dim=1)
        return self.linear(linear_input)
