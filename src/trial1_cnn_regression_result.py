import os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import tensor as ts
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from net_module.trial1_conv_module import ConvNN
from net_module.loss_functions import loss_MSE
from network_manager import NetworkManager

from data_handle.segdata_handler import *
from data_handle.sad_object import *

print("Program: animation\n")

### Customize
input_channel = 1
dim_output = 2
fc_input = 386080

model_path = os.path.join(Path(__file__).parent.parent, 'Model/trial1_model')
csv_path   = os.path.join(Path(__file__).parent.parent, 'Data/SimpleRegression/all_data.csv')
data_dir   = os.path.join(Path(__file__).parent.parent, 'Data/SimpleRegression/')

idx_start = 0
idx_end = 127
pause_time = 0.1

### Prepare data
composed = transforms.Compose([ToTensor()])
dataset = ImageStackDataset(csv_path=csv_path, root_dir=data_dir, channel_per_image=1, transform=composed, T_channel=False)
print("Data prepared. #Samples:{}.".format(len(dataset)))
print('Sample: {\'image\':',dataset[0]['image'].shape,'\'label\':',dataset[0]['label'],'}')

### Initialize the model
net = ConvNN(input_channel, dim_output, fc_input, with_batch_norm=True)
myCNN = NetworkManager(net, loss_MSE)
myCNN.build_Network()
model = myCNN.model
model.load_state_dict(torch.load(model_path))
model.eval() # with BN layer, must run eval first

### Visualize
boundary_coords, obstacle_list = return_Map()
graph = Graph(boundary_coords, obstacle_list)

fig, ax = plt.subplots()
idc = np.linspace(idx_start,idx_end,num=idx_end-idx_start).astype('int') 
first_loop = 1
for idx in idc:
    plt.cla()
    graph.plot_map(ax, clean=1)
    
    img   = dataset[idx]['image']
    label = dataset[idx]['label']

    est = model(img.unsqueeze(0).float().cuda()).cpu()
    est = est[0].detach().numpy()

    plt.plot(label[0], label[1], 'bo', label="ground truth")
    plt.plot(est[0],   est[1],   'rx', label="estimation")
    plt.xlabel("x [m]", fontsize=14)
    plt.ylabel("y [m]", fontsize=14)
    plt.legend()
    plt.legend(prop={'size': 14}, loc='upper right')

    if idx == idc[-1]:
        plt.text(5,5,'Done!',fontsize=20)
    plt.pause(pause_time)

plt.show()

