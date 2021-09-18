import os, sys, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch import tensor as ts
from torchvision import transforms

from net_module.trial2_conv_module import ConvNN
from net_module.loss_functions import loss_MSE
from network_manager import NetworkManager

from data_handle.segdata_handler import *

print("Program: training\n")
if torch.cuda.is_available():
    print(torch.cuda.current_device(),torch.cuda.get_device_name(0))
else:
    print(f'CUDA not working! Pytorch: {torch.__version__}.')
    sys.exit(0)
torch.cuda.empty_cache()

### Customize
input_channel = 4
dim_output = 2
fc_input = 182528

epoch = 2
validation_prop = 0.2
batch_size = 10

save_path = os.path.join(Path(__file__).parent.parent, 'Model/trial2_model') # if None, don't save
csv_path  = os.path.join(Path(__file__).parent.parent, 'Data/SimpleAvoidOneModeOneChannel/all_data.csv')
data_dir  = os.path.join(Path(__file__).parent.parent, 'Data/SimpleAvoidOneModeOneChannel/')

### Prepare data
composed = transforms.Compose([ToTensor()])
dataset = ImageStackDataset(csv_path=csv_path, root_dir=data_dir, channel_per_image=1, transform=composed, T_channel=False)
myDH = DataHandler(dataset, batch_size=batch_size, shuffle=True, validation_prop=validation_prop, validation_cache=2)
print("Data prepared. #Samples(training, val):{}, #Batches:{}".format(myDH.return_length_ds(), myDH.return_length_dl()))

### Initialize the model
net = ConvNN(input_channel, dim_output, fc_input, with_batch_norm=True)
myCNN = NetworkManager(net, loss_MSE)
myCNN.build_Network()
model = myCNN.model

### Training
start_time = time.time()
myCNN.train(myDH, batch_size, epoch, val_after_batch=10)
total_time = round((time.time()-start_time)/3600, 4)
if (save_path is not None) & myCNN.complete:
    torch.save(model.state_dict(), save_path)
nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\nTraining done: {} parameters. Cost time: {}h.".format(nparams, total_time))

### Visualize the training process
fig = plt.figure()
myCNN.plot_history_loss()