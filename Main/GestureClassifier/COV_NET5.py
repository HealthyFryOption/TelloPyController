import torch.nn as nn
import torch.nn.functional as F

# Width and Height is 150 pixels
IMG_WIDTH = 150
IMG_HEIGHT = 150

CLASS_LABELS = {
    0:"PALM", 1:"FIST", 2:"PEACE", 3:"RIGHT", 4:"LEFT", 5:"THUMB"
}

NUM_CLASSES = 6
FLATTEN_NUM = 17*17*156

class Neural(nn.Module):

    def __init__(self):
        super().__init__()

       # Since our images are small, downsample don't need to be aggresive
        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1
            ),
            nn.ReLU(),
                nn.MaxPool2d(
                kernel_size=(2,2),
                stride=(2,2)
            ),
            # 75W

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3,3),
                stride=(1,1),
                padding=0
            ),
            # 73 width

            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2,2),
                stride=(2,2)
            ),
            # 36 width

            nn.Conv2d(
                in_channels=64,
                out_channels=156,
                kernel_size=(3,3),
                stride=(1,1),
                padding=0
            ),
            nn.ReLU(),
            # 34 width

            nn.MaxPool2d(
                kernel_size=(2,2),
                stride=(2,2)
            ),
            # 17 width
        )

        self.classification = nn.Sequential(
            nn.Linear(FLATTEN_NUM, 1350),
            nn.ReLU(True),
            

            nn.Linear(1350, 120),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            
            nn.Linear(120, NUM_CLASSES),
        )

    def forward(self, X):
        X = self.conv_layer(X)
        X = X.view(-1, FLATTEN_NUM)

        classes = self.classification(X)
        probability = F.softmax(classes, dim=1)

        return probability

    def showNetworkDetails(self):
        print("================ DETAILS ================")
        print(self.conv_layer)
        print(self.classification)
        print("================ DETAILS ================")