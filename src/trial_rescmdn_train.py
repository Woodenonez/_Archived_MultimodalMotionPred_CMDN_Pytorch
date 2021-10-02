import os, sys, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch import tensor as ts
from torchvision import transforms

from net_module.trial_cmdn_module import ConvMDN
from net_module.loss_functions import loss_NLL, loss_MAE
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
num_components = 2
fc_input = 98304

epoch = 10
validation_prop = 0.2
batch_size = 10
early_stopping = 5

save_path = os.path.join(Path(__file__).parent.parent, 'Model/new') # if None, don't save
csv_path  = os.path.join(Path(__file__).parent.parent, 'Data/SimpleAvoid2m1cBig/all_data.csv')
data_dir  = os.path.join(Path(__file__).parent.parent, 'Data/SimpleAvoid2m1cBig/')

### Prepare data
composed = transforms.Compose([ToTensor()])
dataset = ImageStackDataset(csv_path=csv_path, root_dir=data_dir, channel_per_image=1, transform=composed, T_channel=False)
myDH = DataHandler(dataset, batch_size=batch_size, shuffle=True, validation_prop=validation_prop, validation_cache=2)
print("Data prepared. #Samples(training, val):{}, #Batches:{}".format(myDH.return_length_ds(), myDH.return_length_dl()))

### Initialize the model
net = ConvMDN(input_channel, dim_output, fc_input, num_components=num_components, with_batch_norm=True)
myNet = NetworkManager(net, loss_NLL, extra_metric=None, early_stopping=early_stopping)
myNet.build_Network()
model = myNet.model

### Training
start_time = time.time()
myNet.train(myDH, batch_size, epoch, val_after_batch=10)
total_time = round((time.time()-start_time)/3600, 4)
if (save_path is not None) & myNet.complete:
    torch.save(model.state_dict(), save_path)
nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\nTraining done: {} parameters. Cost time: {}h.".format(nparams, total_time))

### Visualize the training process
myNet.plot_history_loss()

try:
    myNet.plot_mdn_tracker()
    myNet.plot_mdn_grad_tracker()
except:
    pass

plt.show()

