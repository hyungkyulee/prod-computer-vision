# import the usual resources
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2

from models import Net

# Get the test image
# image_path = 'data/training/claire_Danes_51.jpg'
#
# # Load color image, and convert it to grayscale
# image_bgr = cv2.imread(image_path)
# image_grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#
# # normalise, rescale entry
# image_grey = image_grey.astype("float32") / 255

#### --------> display
# plt.imshow(image_grey, cmap='gray')
# plt.show()


## TODO: Once you've define the network, you can instantiate it
# one example conv layer has been provided for you

net = Net()
print(net)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor

## TODO: define the data_transform using transforms.Compose([all tx's, . , .])
# order matters! i.e. rescaling should come before a smaller crop
# data_transform = transforms.Compose([transforms.ToTensor()])
# data_transform = transforms.Compose([transforms.ToPILImage()])
# data_transform = transforms.ToPILImage()
# data_transform = transforms.Compose([
#                         transforms.ToPILImage(),
#                         transforms.Resize(),
#                         transforms.ToTensor(),
#         ])

data_transform = transforms.Compose([
    Rescale(256),
    RandomCrop(224),
    ToTensor()
])

# testing that you've defined a transform
assert (data_transform is not None), 'Define a data_transform'

# create the transformed dataset
transformed_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                             root_dir='data/training/',
                                             transform=data_transform)

print('Number of images: ', len(transformed_dataset))

# iterate through the transformed dataset and print some stats about the first few samples
for i in range(4):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['keypoints'].size())

# load training data in batches
batch_size = 10

train_loader = DataLoader(transformed_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

# create the test dataset
test_dataset = FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv',
                                      root_dir='data/test/',
                                      transform=data_transform)

# load test data in batches
batch_size = 10

test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)


# test the model on a batch of test images

def net_sample_output():
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):

        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']

        # convert images to FloatTensors
        images = images.type(torch.FloatTensor)

        # forward pass to get net output
        output_pts = net(images)

        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)

        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts


from multiprocessing import Process, freeze_support

# call the above function
# returns: test images, test predicted keypoints, test ground truth keypoints
if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    freeze_support()
    test_images, test_outputs, gt_pts = net_sample_output()

# print out the dimensions of the data to see if they make sense
print(test_images.data.size())
print(test_outputs.data.size())
print(gt_pts.size())
