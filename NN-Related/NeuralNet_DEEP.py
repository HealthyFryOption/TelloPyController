import torch.nn as nn
import torch.nn.functional as F

# Width and Height is 140 pixels
IMG_WIDTH = 140
IMG_HEIGHT = 140

CLASSES_COUNT = 6

DRONE_LABELS = {
    0:"PALM", 1:"FIST", 2:"PEACE", 3:"RIGHT", 4:"LEFT", 5:"THUMB"
}

class NeuralNet(nn.Module):

    KERNEL_SIZE = 3
    CONV_STRIDE = 1
    FINAL_CONV_OUT_CHANNEL = 280
    CONVO_COUNT = 4

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, 
                                out_channels=32, 
                                kernel_size=self.KERNEL_SIZE,
                                stride=self.CONV_STRIDE)

        self.conv2 = nn.Conv2d(in_channels=32, 
                                out_channels=64, 
                                kernel_size=self.KERNEL_SIZE,
                                stride=self.CONV_STRIDE)

        self.conv3 = nn.Conv2d(in_channels=64, 
                                out_channels= 128, 
                                kernel_size=self.KERNEL_SIZE,
                                stride=self.CONV_STRIDE)

        self.conv4 = nn.Conv2d(in_channels=128, 
                                out_channels= self.FINAL_CONV_OUT_CHANNEL, 
                                kernel_size=self.KERNEL_SIZE,
                                stride=self.CONV_STRIDE)
        

        # convo number, final channel output, initial image pixel width size, universal kernel size for convulation, universal kernel stride value for convulation
        self.toLinearSize= self.getLinearSize(self.CONVO_COUNT, self.FINAL_CONV_OUT_CHANNEL, IMG_WIDTH, self.KERNEL_SIZE, 0, self.CONV_STRIDE)

        self.fc1 = nn.Linear(self.toLinearSize, 1200)
        self.fc2 = nn.Linear(1200, 120)
        self.fc3 = nn.Linear(120, CLASSES_COUNT)

    def convolution(self, X):
        X = F.max_pool2d(F.relu(self.conv1(X)), (2,2))
        X = F.max_pool2d(F.relu(self.conv2(X)), (2,2))
        X = F.max_pool2d(F.relu(self.conv3(X)), (2,2))
        X = F.max_pool2d(F.relu(self.conv4(X)), (2,2))

        return X

    def forward(self, X):
        X = self.convolution(X)
        
        X = X.view(-1, self.toLinearSize)
        
        # pass flattened convolutional output to first connected layer
        X = F.relu(self.fc1(X)) 
        X = F.relu(self.fc2(X))

        X = self.fc3(X) 
        
        return X

    def showNetworkDetails(self):
        print("================ DETAILS ================")
        print(self.conv1)
        print(self.conv2)
        print(self.conv3)
        print(self.conv4)
        print(self.fc1)
        print(self.fc2)
        print(self.fc3)
        print("================ DETAILS ================")

    def getLinearSize(self, num_convo, last_out_channel, size, kernel, padding, stride, max2pool=True):
        #[(W−K+2P)/S] + 1

        for i in range(num_convo):
            size = ((size-kernel+(2*padding))/stride)+1
            if max2pool:
                size //= 2 #assuming maxpool stride is 2 and size is 2x2

        return int(size*size*last_out_channel)