import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import tensor as ts
import torch.optim as optim

from net_module.loss_functions import *

class NetworkManager():
    """ 
    
    """
    def __init__(self, net, loss_function, extra_metric=None, early_stopping=3, verbose=True):
        self.vb = verbose
        
        self.lr = 1e-4      # learning rate
        self.w_decay = 1e-4 # L2 regularization
        self.Loss = []      # track the loss
        self.Val_loss= []   # track the validation loss
        self.es = early_stopping

        self.net = net
        self.loss_function0 = loss_NLL_fix
        self.loss_function = loss_function

        self.complete = False
        self.tracker = []
        self.grad_tracker = []

    def build_Network(self):
        self.gen_Model()
        self.gen_Optimizer(self.model.parameters())
        if self.vb:
            print(self.model)

    def gen_Model(self):
        self.model = nn.Sequential()
        self.model.add_module('Net', self.net)
        return self.model.float().cuda()

    def gen_Optimizer(self, parameters):
        self.optimizer = optim.Adam(parameters, lr=self.lr, weight_decay=self.w_decay, betas=(0.99, 0.999))
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)
        return self.optimizer

    def validate(self, data, labels, loss_function, val=False):
        outputs = self.model(data)
        if not val:
            self.record_mdn_tracker(outputs)
        loss = loss_function(outputs, labels)
        # if not val:
        #     print(loss)
        return loss

    def train_batch(self, batch, label, loss_function):
        self.model.zero_grad()
        loss = self.validate(batch, label, loss_function)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        # print(self.net.mdn.layer_mapping.weight.grad)
        # print(self.net.mdn.p)
        # print(self.net.mdn.mu.grad)
        # print(self.net.mdn.sigma.grad)
        # weight_map = self.get_parameter_map(self.net.mdn.layer_mapping.weight.grad)
        # p_grad = self.get_parameter_map(self.net.mdn.p.grad)
        # p_map = self.get_parameter_map(self.net.mdn.p)

        return loss

    def train(self, data_handler, batch_size, epoch, val_after_batch=1):
        print('\nTraining...')
        data_val = data_handler.dataset_val
        max_cnt_per_epoch = data_handler.return_length_dl()
        min_val_loss = np.Inf
        min_val_loss_epoch = np.Inf
        epochs_no_improve = 0
        cnt = 0 # counter for batches over all epochs
        self.epoch_nodes = []
        for ep in range(epoch):
            cnt_per_epoch = 0 # counter for batches within the epoch

            # loss_epoch = self.loss_function0
            if ep < 2:
                loss_epoch = self.loss_function0
            else:
                loss_epoch = self.loss_function

            self.epoch_nodes.append(cnt)
            while (cnt_per_epoch<max_cnt_per_epoch):
                cnt += 1
                cnt_per_epoch += 1
                batch, label = data_handler.return_batch()
                batch, label = batch.float().cuda(), label.float().cuda()
                
                # a_channel = batch[0,0,:,:]
                # a_unique = torch.unique(a_channel)
                # print(a_unique)
                # print(label)
                # print(batch.shape) # TEST
                # print(batch[0,1,:,:])
                # plt.imshow(batch[0,0,:,:].cpu())
                # plt.show()
                # sys.exit()

                loss = self.train_batch(batch, label, loss_function=loss_epoch) # train here
                self.Loss.append(loss.item())
                try:
                    self.record_mdn_grad()
                except:
                    pass

                # if cnt_per_epoch%5==0:
                #     self.plot_parameter_maps()
                #     plt.pause(0.1)

                if len(data_val)>0 & (cnt_per_epoch%val_after_batch==0):
                    del batch
                    del label
                    val_data, val_label = data_handler.return_val()
                    val_data, val_label = val_data.float().cuda(), val_label.float().cuda()
                    val_loss = self.validate(val_data, val_label, loss_function=self.loss_function, val=True)
                    self.Val_loss.append((cnt, val_loss.item()))
                    del val_data
                    del val_label
                    if val_loss < min_val_loss_epoch:
                        min_val_loss_epoch = val_loss

                if np.isnan(loss.item()): # assert(~np.isnan(loss.item())),("Loss goes to NaN!")
                    print(f"Loss goes to NaN! Fail after {cnt} batches.")
                    self.complete = False
                    return

                if (cnt_per_epoch%20==0 or cnt_per_epoch==max_cnt_per_epoch) & (self.vb):
                    if len(data_val)>0:
                        prt_loss = f'Loss/Val_loss: {round(loss.item(),4)}/{round(val_loss.item(),4)}'
                    else:
                        prt_loss = f'Training loss: {round(loss.item(),4)}'
                    prt_num_samples = f'{cnt_per_epoch*batch_size/1000}k/{max_cnt_per_epoch*batch_size/1000}k'
                    prt_num_epoch = f'Epoch {ep+1}/{epoch}'
                    print('\r'+prt_loss+', '+prt_num_samples+', '+prt_num_epoch+'     ', end='')
            if min_val_loss_epoch < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = min_val_loss_epoch
            else:
                epochs_no_improve += 1
            if (self.es > 0) & (epochs_no_improve >= self.es):
                print(f'\nEarly stopping after {self.es} epochs with no improvement.')
                break
            print() # end while
        self.complete = True
        # plt.show()
        print('\nTraining Complete!')


    def plot_history_loss(self):
        plt.figure()
        plt.plot(self.epoch_nodes, np.array(self.Loss)[self.epoch_nodes], 'r|', markersize=20)
        plt.plot(np.linspace(1,len(self.Loss),len(self.Loss)), self.Loss, '.', label='loss')
        if len(self.Val_loss):
            plt.plot(np.array(self.Val_loss)[:,0], np.array(self.Val_loss)[:,1], '.', label='val_loss')
        plt.xlabel('#batch')
        plt.ylabel('Loss')
        plt.legend()

    ### Inspection functions
    def record_mdn_tracker(self, x):
        alp, mu, sigma = x[0], x[1], x[2]
        alp   = alp.cpu().detach().numpy()
        mu    = mu.cpu().detach().numpy()
        sigma = sigma.cpu().detach().numpy()

        alp_list = []
        mu_list = []
        sigma_list = []
        for i in range(alp.shape[1]):
            alp_list.append(np.mean(alp[:,i]))
            mu_list.append(list(np.mean(mu[:,i],axis=0)))
            sigma_list.append(list(np.mean(sigma[:,i],axis=0)))
        mu_list = np.array(mu_list).reshape(1,-1).squeeze(0).tolist()
        sigma_list = np.array(sigma_list).reshape(1,-1).squeeze(0).tolist()
        self.tracker.append([alp_list, mu_list, sigma_list])

    def plot_mdn_tracker(self):
        all_alp = []
        all_mu = []
        all_sigma = []
        for t in self.tracker:
            all_alp.append(t[0])
            all_mu.append(t[1])
            all_sigma.append(t[2])
        all_alp = np.array(all_alp)
        all_mu = np.array(all_mu)
        all_sigma = np.array(all_sigma)

        M = all_alp.shape[1]
        D = int(all_mu.shape[1]/M)

        plt.figure()
        for i in range(M):
            plt.plot(all_alp[:,i], label=f'alpha {i}')
        plt.legend()

        plt.figure()
        for i in range(D):
            plt.subplot(D,1,i+1)
            for j in range(M):
                plt.plot(all_mu[:,i+D*j], label=f'mu m{j},{i}')
            plt.legend()

        plt.figure()
        for i in range(D):
            plt.subplot(D,1,i+1)
            for j in range(M):
                plt.plot(all_sigma[:,i+D*j], label=f'sigma m{j},{i}')
            plt.legend()

    def get_parameter_map(self, parameter):
        return torch.clone(parameter).cpu().detach()

    def record_mdn_grad(self):
        p_grad = self.get_parameter_map(self.net.mdn.p.grad)
        p_map = self.get_parameter_map(self.net.mdn.p)
        p_grad = torch.mean(p_grad, axis=0).tolist()
        p_map  = torch.mean(p_map, axis=0).tolist()
        self.grad_tracker.append([p_grad, p_map])

    def plot_mdn_grad_tracker(self):
        all_grad = []
        all_p = []
        for t in self.grad_tracker:
            all_grad.append(t[0])
            all_p.append(t[1])
        all_grad = np.array(all_grad)
        all_p = np.array(all_p)

        N = all_grad.shape[1]

        plt.figure()
        for i in range(N):
            plt.plot(all_grad[:,i], label=f'grad-{i}')
        plt.legend()

        plt.figure()
        plt.plot(all_grad[:,0]/all_grad[:,1], label=f'grad alp1/alp2')
        plt.legend()

        plt.figure()
        for i in range(N):
            plt.plot(all_p[:,i], label=f'p-{i}')
        plt.legend()

    def plot_parameter_maps(self):
        weight_map = self.get_parameter_map(self.net.mdn.layer_mapping.weight.grad)
        p_grad = self.get_parameter_map(self.net.mdn.p.grad)
        p_map = self.get_parameter_map(self.net.mdn.p)
        plt.subplot(131)
        plt.imshow(weight_map, cmap='gray')
        plt.title('weight_grad before p')
        plt.subplot(132)
        plt.imshow(p_grad, cmap='gray')
        plt.title('p_grad')
        plt.subplot(133)
        plt.imshow(p_map, cmap='gray')
        plt.title('p')
