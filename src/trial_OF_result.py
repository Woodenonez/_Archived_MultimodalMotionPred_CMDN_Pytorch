import os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from torch import tensor as ts
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from net_module.trial5_cmdn_module import ConvMDN
from net_module.loss_functions import loss_NLL, loss_MAE
from network_manager import NetworkManager

from data_handle.segdata_handler_OF import *
from data_handle.sad_object import *

print("Program: animation\n")

### Customize
input_channel = 2
dim_output = 2
num_components = 2
fc_input = 45632

model_path = os.path.join(Path(__file__).parent.parent, 'Model/new')
csv_path   = os.path.join(Path(__file__).parent.parent, 'Data/SimpleAvoid2m1cOF/all_data.csv')
data_dir   = os.path.join(Path(__file__).parent.parent, 'Data/SimpleAvoid2m1cOF/')

# idx_start = 100
# idx_end = 227
idx_start = 0
idx_end = 127
pause_time = 0.1

### Prepare data
composed = transforms.Compose([MaxNormalize(),ToTensor()])
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
    label = dataset[idx]['label'] * 10
    traj  = np.array(dataset[idx]['traj'])

    alp, mu, sigma = model(img.unsqueeze(0).float().cuda())

    alp = alp[0].cpu().detach().numpy()
    mu  = mu[0].cpu().detach().numpy() * 10
    sigma = sigma[0].cpu().detach().numpy() * 10

    plt.plot(traj[-1,0], traj[-1,1], 'ko', label='current')
    plt.plot(traj[:-1,0], traj[:-1,1], 'k.') # past

    patch = patches.Ellipse(mu[0,:], sigma[0,0], sigma[0,1], fc='y')
    ax.add_patch(patch)
    patch = patches.Ellipse(mu[1,:], sigma[1,0], sigma[1,1], fc='y')
    ax.add_patch(patch)

    # patch = patches.Ellipse(mu[2,:], sigma[2,0], sigma[2,1], fc='y')
    # ax.add_patch(patch)
    # patch = patches.Ellipse(mu[3,:], sigma[3,0], sigma[3,1], fc='y')
    # ax.add_patch(patch)

    plt.plot(label[0], label[1], 'bo', label="ground truth")
    plt.plot(mu[0,0],   mu[0,1],   'rx', label="estimation")
    plt.plot(mu[1,0],   mu[1,1],   'rx', label="estimation")
    plt.text(mu[0,0],   mu[0,1],str(round(alp[0],2)),fontsize=20)
    plt.text(mu[1,0],   mu[1,1],str(round(alp[1],2)),fontsize=20)

    # plt.plot(mu[2,0],   mu[2,1],   'rx', label="estimation")
    # plt.plot(mu[3,0],   mu[3,1],   'rx', label="estimation")
    # plt.text(mu[2,0],   mu[2,1],str(round(alp[2],2)),fontsize=20)
    # plt.text(mu[3,0],   mu[3,1],str(round(alp[3],2)),fontsize=20)
    plt.xlabel("x [m]", fontsize=14)
    plt.ylabel("y [m]", fontsize=14)
    plt.legend()
    plt.legend(prop={'size': 14}, loc='upper right')

    if idx == idc[-1]:
        plt.text(5,5,'Done!',fontsize=20)
    # plt.pause(pause_time)

    plt.show()

