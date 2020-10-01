import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 16, 5)  # 1 grey scale input, 16 features map, 5x5 filter
        # output size formula : ((W - F + 2P) / S) + 1
        output_img_size = int(((224 - 5 + 0) / 1) + 1)  # output size = 220x220, 16 feature map

        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool1 = nn.MaxPool2d(2, 2)  # by pooling, output size = 110x110, 16

        self.conv2 = nn.Conv2d(16, 32, 5)
        output_img_size = int(((output_img_size/2) - 5 + 0) / 1 + 1)  # output size = 106x106, 32
        self.pool2 = nn.MaxPool2d(2, 2)  # by pooling, 53x53, 32

        self.conv3 = nn.Conv2d(32, 64, 5)
        output_img_size = int(((output_img_size/2) - 5 + 0) / 1 + 1)  # output size = 53x53, 64
        self.pool3 = nn.MaxPool2d(2, 2)  # by pooling, 26x26, 64

        self.conv4 = nn.Conv2d(64, 128, 3)
        output_img_size = int(((output_img_size/2) - 3 + 0) / 1 + 1)  # output size = 24x24, 128
        self.pool4 = nn.MaxPool2d(2, 2)  # by pooling, 12x12, 128

        self.conv5 = nn.Conv2d(128, 256, 3)
        output_img_size = int(((output_img_size/2) - 3 + 0) / 1 + 1)  # output size = 10x10, 256
        self.pool5 = nn.MaxPool2d(2, 2)  # by pooling, 5x5, 256
        
        self.conv6 = nn.Conv2d(256, 512, 3)
        output_img_size = int(((output_img_size/2) - 3 + 0) / 1 + 1)  # output size = 3x3, 512
        self.pool6 = nn.MaxPool2d(2, 2)  # by pooling, 1x1, 512

        final_output_image_size = (output_img_size/2)
        in_features = 512 * final_output_image_size * final_output_image_size
        out_features = 1024
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, 136)

        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.15)
        self.drop3 = nn.Dropout(p=0.2)
        self.drop4 = nn.Dropout(p=0.25)
        self.drop5 = nn.Dropout(p=0.3)
        self.drop6 = nn.Dropout(p=0.35)
        self.drop7 = nn.Dropout(p=0.4)

    def forward(self, x):
        # x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop3(x)
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.drop4(x)
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.drop5(x)
        x = self.pool6(F.relu(self.conv6(x)))
        x = self.drop6(x)

        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)

        # one linear layer
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.drop7(x)

        x = self.fc2(x)

        # # a softmax layer to convert the 10 outputs into a distribution of class scores
        # x = F.log_softmax(x, dim=1)

        # a modified x, having gone through all the layers of your model, should be returned
        return x