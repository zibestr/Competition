from networks._network import AbstractCNN
from torch import nn


class AlexNet(AbstractCNN):
    def __init__(self, n_channels: int = 3,
                 image_shape: tuple[int, int] = (400, 400)):
        super().__init__(n_channels, image_shape)
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=96, kernel_size=11,
                      stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        flatten_out_size = 30976
        self.linear = nn.Sequential(
            nn.Linear(in_features=flatten_out_size + 1, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1)
        )
