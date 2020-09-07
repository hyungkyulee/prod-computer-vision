import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        output_size1 = int((28 - 5) / 1 + 1)

        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 5)
        output_size2 = int((output_size1 - 5) / 1 + 1)

        in_features = 64 * output_size2 * output_size2
        out_features = int((in_features + 13) / 2)
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc1_drop = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(out_features, 13)

    def forward(self, x):
        # x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)

        # one linear layer
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(F.relu(self.fc1(x)))
        x = self.fc2(x)

        # # a softmax layer to convert the 10 outputs into a distribution of class scores
        x = F.log_softmax(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x

