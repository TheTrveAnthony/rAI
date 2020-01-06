import torch
import torch.nn as nn
from torch.nn import Sequential, Conv2d, Dropout2d, MaxPool2d, ReLU, UpsamplingNearest2d, Module
#from torchsummary import summary



""" implemetation of the neural network model, with an encoder-decoder structure 
	it will take 3840*2160*1 images in input and throw 3840*2160*2 masks as outputs

"""



class Net(Module):

    def __init__(self):

        super(Net, self).__init__()


        self.block1 = Sequential(
            Conv2d(1, 32, 3, padding=1),
            ReLU(),
            Dropout2d(0.2),
            Conv2d(32, 32, 3, padding=1),
            ReLU(),
        )

        self.block2 = Sequential(
            Conv2d(32, 64, 3, padding=1),
            ReLU(),
            Dropout2d(0.2),
            Conv2d(64, 64, 3, padding=1),
            ReLU(),
        )


        self.pool = nn.MaxPool2d((2, 2)) 


        self.block3 = Sequential(
            Conv2d(96, 32, 3, padding=1),
            ReLU(),
            Dropout2d(0.2),
            Conv2d(32, 32, 3, padding=1),
            ReLU()
        )

        self.up = UpsamplingNearest2d(scale_factor=2)

        ###### Final layer
        self.conv2d = Conv2d(32, 3, kernel_size=1)
        

        


    def forward(self, x):
       
        out1 = self.block1(x)
        out_pool1 = self.pool(out1)

        out2 = self.block2(out_pool1)
        out_up1 = self.up(out2)
       
        out3 = torch.cat((out_up1, out1), dim=1)
        out3 = self.block3(out3)

        out = self.conv2d(out3)

        return out

#### Let's see how it looks like :

#md = Net()
#summary(md, input_size=(1, 480, 270), batch_size=50, device='cpu')  ##### I divided the input shape my 10, otherwise my laptop doesn't survive to it
