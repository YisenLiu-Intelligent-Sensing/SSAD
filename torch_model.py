"""
 @file   torch_model.py
 @brief  Script for CNN model
 @author Yisen Liu
 Copyright (C) 2022 Institute of Intelligent Manufacturing, Guangdong Academy of Sciences. All right reserved.
"""


########################################################################
# import python-library
########################################################################\
import torch
import torch.nn as nn

########################################################################
# torch model
########################################################################

class ss_model(nn.Module):
    def __init__(self):
        super(ss_model, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1,padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 16, 3, stride=1,padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 16, 3, stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 4, 3, stride=1,padding=1),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=15*15*4, out_features=16),
            nn.Tanh()
        )

        self.fc2 = nn.Linear(in_features=16, out_features=5)

    def forward(self,input):
        x = self.conv_layer(input)
        x = self.fc1(x)
        
        output = self.fc2(x)
        
        return output,x

    

