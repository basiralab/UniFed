import torch.nn as nn
import torch.nn.functional as F

class SimCNN(nn.Module):
    """
    Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
    Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
    We use `zero`-padding instead of  `same`-padding used in
     https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.
    """
    def __init__(self, num_classes):
        super(SimCNN, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)

        self.fc1 = nn.Linear(64 * 5 * 5, 2048)
        self.output = nn.Linear(2048, self.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1, 64 * 5 * 5)
        # 
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x