import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_

try:
    from submodules import compact_conv_layer as conv
    from mdn_modules import *
except:
    from net_module.submodules import compact_conv_layer as conv
    from net_module.mdn_modules import *

class ConvMDN(nn.Module):
    # batch x channel x height x width
    def __init__(self, input_channel, dim_output, fc_input, num_components, with_batch_norm=True):
        super(ConvMDN,self).__init__()

        dim_fea = num_components * dim_output * 3

        self.conv1  = conv(with_batch_norm, input_channel, 16)
        self.conv2  = conv(with_batch_norm, 16,            32)
        self.conv3  = conv(with_batch_norm, 32,            64)
        self.pool   = nn.MaxPool2d(2, 2)

        self.fc1    = nn.Linear(fc_input,64)
        self.fc2    = nn.Linear(64,dim_fea)
        self.relu   = nn.ReLU()

        self.mdn = ClassicMixtureDensityModule(dim_fea, dim_output, num_components)

    def forward(self, x):
        out_conv = self.pool(self.conv1(x))
        out_conv = self.pool(self.conv2(out_conv))
        out_conv = self.pool(self.conv3(out_conv))

        x = out_conv.view(out_conv.size(0), -1) # batch x -1
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        x = self.mdn(x)

        return x



