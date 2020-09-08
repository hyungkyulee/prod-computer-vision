import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 28, 5, padding=2) # 1 grey scale, 28x28 image, 5x5 filter
        # output size formula : ((W - F + 2P) / S) + 1
        output_size1 = int(((32 - 5 + (2*2)) / 1) + 1) # output size = 32 (C1 = 32 x 32 x feature map)

        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool = nn.MaxPool2d(2, 2) # by pooling, feature map size = 14*14

        self.conv2 = nn.Conv2d(28, 10, 5, padding=2)
        output_size2 = int((10 - 5 + (2*2)) / 1 + 1) # output size = 10 (C2 = 10 x 10 x feature map)
        self.pool = nn.MaxPool2d(2, 2) # by pooling, feature map size = 7*7

        in_features = output_size2 * 7 * 7 # output_size2 * feature map size
        out_features = 1024 # why??
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, output_size2)

    def forward(self, x):
        # x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # flatten the inputs into a vector
        # x = x.view(x.size(0), -1)
        x = x.view(-1, 10 * 7 * 7) # reshape variable

        # one linear layer
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)


        # # a softmax layer to convert the 10 outputs into a distribution of class scores
        x = F.log_softmax(x, dim=1)

        # a modified x, having gone through all the layers of your model, should be returned
        return x

