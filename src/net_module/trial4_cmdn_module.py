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

        self.conv1  = conv(with_batch_norm, input_channel, 16, kernel_size=5)
        self.conv2  = conv(with_batch_norm, 16,            16, padding=1)
        self.conv3  = conv(with_batch_norm, 16,            32)
        self.conv4  = conv(with_batch_norm, 32,            32)
        self.pool   = nn.MaxPool2d(2, 2)
        self.apool  = nn.AvgPool2d(2, 2)

        self.fc1    = nn.Linear(fc_input,64)
        self.fc2    = nn.Linear(64,dim_fea)
        self.relu   = nn.ReLU()

        # self.fc_sampling = nn.Linear(dim_fea, dim_output)
        self.mdn = ClassicMixtureDensityModule(dim_fea, dim_output, num_components) # model 1
        # self.mdn = SplitMixtureDensityModule(dim_fea, dim_output, num_components) # model 2
        # self.mdn = SamplingMixtureDensityModule(dim_output, num_components) # model 3

    def forward(self, x):
        out_conv = self.pool(self.conv1(x))
        out_conv = self.conv2(out_conv) + out_conv # res_net
        out_conv = self.pool(self.conv3(out_conv))
        out_conv = self.pool(self.conv4(out_conv))

        x = out_conv.view(out_conv.size(0), -1) # batch x -1
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        # x = self.fc_sampling(self.relu(x)) # model 3

        x = self.mdn(x)

        return x



