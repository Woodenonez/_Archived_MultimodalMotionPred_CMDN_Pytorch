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

from net_module.trial3_cmdn_module import ConvMDN
from net_module.loss_functions import loss_NLL, loss_MAE
from network_manager import NetworkManager

from data_handle.segdata_handler import *
from data_handle.sad_object import *

print("Program: animation\n")

### Customize
input_channel = 4
dim_output = 2
num_components = 2
fc_input = 182528

model_path = os.path.join(Path(__file__).parent.parent, 'Model/trial3_model')
csv_path   = os.path.join(Path(__file__).parent.parent, 'Data/SimpleAvoidTwoModeOneChannel/all_data.csv')
data_dir   = os.path.join(Path(__file__).parent.parent, 'Data/SimpleAvoidTwoModeOneChannel/')

idx_start = 0
idx_end = 127
pause_time = 0.1

### Prepare data
composed = transforms.Compose([ToTensor()])
dataset = ImageStackDataset(csv_path=csv_path, root_dir=data_dir, channel_per_image=1, transform=composed, T_channel=False)
print("Data prepared. #Samples:{}.".format(len(dataset)))
print('Sample: {\'image\':',dataset[0]['image'].shape,'\'label\':',dataset[0]['label'],'}')

### Initialize the model
net = ConvMDN(input_channel, dim_output, fc_input, num_components)
myNet = NetworkManager(net, loss_NLL)
myNet.build_Network()
model = myNet.model
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
    traj  = np.array(dataset[idx]['traj'])

    alp, mu, sigma = model(img.unsqueeze(0).float().cuda())

    alp = alp[0].cpu().detach().numpy()
    mu  = mu[0].cpu().detach().numpy()
    sigma = sigma[0].cpu().detach().numpy()

    plt.plot(traj[-1,0], traj[-1,1], 'ko', label='current')
    plt.plot(traj[:-1,0], traj[:-1,1], 'k.') # past
    plt.plot(label[0], label[1], 'bo', label="ground truth")
    plt.plot(mu[0,0],   mu[0,1],   'rx', label="estimation")
    plt.plot(mu[1,0],   mu[1,1],   'rx', label="estimation")
    plt.xlabel("x [m]", fontsize=14)
    plt.ylabel("y [m]", fontsize=14)
    plt.legend()
    plt.legend(prop={'size': 14}, loc='upper right')

    if idx == idc[-1]:
        plt.text(5,5,'Done!',fontsize=20)
    plt.pause(pause_time)

plt.show()

