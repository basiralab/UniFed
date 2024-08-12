import torch.nn as nn

"""
The reason for puting 512*1*1 as our input: The first linear layer has 512*7*7 input features. But how did we reach here? Note that VGG takes an input size of 224*224 (height x width) for images.
And we have 5 max-pool layers with a stride of 2 which are going to halve the features maps each time. Also, the final convolutional layer has 512 output channels. To get the number of input
features for the first Linear() layer, we just need to calculate it using the following formula.
224 / (2^5) = 7. And because the final convolutional layer has 512 output channels, the first linear layer has 512*7*7 input features.
"""

# the VGG11 architecture
class VGG11(nn.Module):
    def __init__(self, in_channels=1, num_classes=23):
        super(VGG11, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # fully connected linear layers
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features= 512 * 1 * 1, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        # flatten to prepare for the fully connected layers
        # h = x.view(x.size(0), -1)
        x = x.view(x.shape[0], -1)
        # print("x;", x.shape[0])
        x = self.linear_layers(x)
        return x