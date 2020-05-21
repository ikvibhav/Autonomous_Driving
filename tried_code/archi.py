import torch
import torch.nn as nn
import torch.nn.functional as F

data_dim = 3            #keep 3 for 3D (RGB) data and 1 for grayscale data
input_width = 66
input_height = 200

class Net(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv2d(data_dim, 24, 5, 2) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(24, 36, 5, 2) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv3 = nn.Conv2d(36, 48, 5, 2)
        self.conv4 = nn.Conv2d(48, 64, 3, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1)

        x = torch.randn(data_dim,input_width,input_height).view(-1,data_dim,input_width,input_height) #-1 indicates its any size of data that can come. Data comes as a batch of data.
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 100) #flattening.
        self.fc2 = nn.Linear(100, 50) # 512 in, 2 out bc we're doing 2 classes (dog vs cat).
        self.fc3 = nn.Linear(50, 10)

    def convs(self, x):
        # max pooling over 2x2
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))

        print(x[0].shape)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)				 #Pass through all the convolutional layers
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)

net = Net()
print(net)