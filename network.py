import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, image_size):
        super(Net, self).__init__()
        self.image_size = image_size
        # common layers for feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 16 * 16, 256) # Assuming input images are 128x128
        self.dropout = nn.Dropout2d(0.5)
        
        # fully connected layers for classification
        self.fc2_clsf = nn.Linear(256, 10) # 10 classes
        
        # fully connected layers for bbox regression
        self.obj_x1_out = nn.Linear(256, 1)
        self.obj_y1_out = nn.Linear(256, 1)
        self.obj_x2_out = nn.Linear(256, 1)
        self.obj_y2_out = nn.Linear(256, 1)

    def forward(self, x):
        # shared convolutional layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 128*16*16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # classification path
        clsf = F.log_softmax(self.fc2_clsf(x), dim=1)

        # bbox regression path
        x1 = F.tanh(self.obj_x1_out(x))
        y1 = F.tanh(self.obj_y1_out(x))
        x2 = F.tanh(self.obj_x2_out(x))
        y2 = F.tanh(self.obj_y2_out(x))
        
        return clsf, [x1.squeeze(), y1.squeeze(), x2.squeeze(), y2.squeeze()]
