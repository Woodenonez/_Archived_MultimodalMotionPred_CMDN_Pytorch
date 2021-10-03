import sys, math

import torch
from torch import tensor as ts
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

def cal_GauProb(mu, sigma, x):
    """
    Return the probability of "data" given MoG parameters "mu" and "sigma".
    
    Arguments:
        mu    (BxGxC) - The means of the Gaussians. 
        sigma (BxGxC) - The standard deviation of the Gaussians.
        x     (BxC)   - A batch of data points (coordinates of position).

    Return:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding mu/sigma index.
            (Assume the dimensions of the output are independent to each other.)
    """
    x = x.unsqueeze(1).expand_as(mu) # BxC -> Bx1xC -> BxGxC
    prob = torch.rsqrt(torch.tensor(2*math.pi)) * torch.exp(-((x - mu) / sigma)**2 / 2) / sigma
    return torch.prod(prob, dim=2) # overall probability for all output's dimensions in each component, BxG

def cal_multiGauProb(alp, mu, sigma, x):
    """
    Arguments:
        alp   (BxG)   - (alpha) Component's weight
    """
    prob = alp * cal_GauProb(mu, sigma, x) # BxG
    prob = torch.sum(prob, dim=1) # Bx1
                                  # overall prob for each batch (sum is for all compos)
    return prob

def loss_NLL(x, data):
    """
    Calculates the error, given the MoG parameters and the data.
    The loss is the negative log likelihood of the data given the MoG parameters.
    """
    alp, mu, sigma = x[0], x[1], x[2]
    # alp_avg = torch.mean(alp, axis=0)
    # sigma_avg = torch.mean(sigma.view(sigma.shape[0],-1), axis=0)
    nll = -torch.log(cal_multiGauProb(alp, mu, sigma, data) + 1e-6)
    return torch.mean(nll)  #+ 2*torch.linalg.norm(alp_avg) #+ torch.linalg.norm(sigma_avg)

def loss_adaptive_NLL(x, data):
    sfx = nn.Softmax()
    prob = cal_GauProb(mu, sigma, data) # (BxG)
    adaptive_alpha = sfx(prob)
    x.insert(0, adaptive_alpha)
    return loss_NLL(x, data)

def loss_NLL_fix(x, data):
    alp, mu, sigma = x[0], x[1], x[2]
    alp_fix = (torch.ones(alp.shape)/alp.shape[1]).cuda()
    sigma_fix = (torch.ones(sigma.shape)).cuda()
    nll = -torch.log(cal_multiGauProb(alp_fix, mu, sigma_fix, data) + 1e-6)
    return torch.mean(nll)

def loss_MaDist(alp, mu, sigma, data): # Mahalanobis distance
    '''
    mu    (GxC) - The means of the Gaussians. 
    sigma (GxC) - The standard deviation of the Gaussians.
    '''
    md = []
    alp = alp/sum(alp) #normalization
    for i in range(mu.shape[0]): # do through every component
        mu0 = (data-mu[i,:]).unsqueeze(0) # (x-mu)
        S_inv = ts([[1/sigma[i,0],0],[0,1/sigma[i,1]]]) # S^-1 inversed covariance matrix
        md0 = torch.sqrt( S_inv[0,0]*mu0[0,0]**2 + S_inv[1,1]*mu0[0,1]**2 )
        md.append(md0)
    return ts(md), sum(ts(md)*alp)

def loss_MSE(preds, labels):
    mse_loss = nn.MSELoss()
    loss = mse_loss(preds, labels)
    return loss

def loss_MAE(preds, labels): # for batch
    mae_loss = nn.L1Loss()
    loss = mae_loss(preds, labels)
    return loss

def loss_weightedMSE(alpha, preds, labels):
    mse = torch.zeros(alpha.shape[0],1)
    for i in range(preds.shape[1]):
        weight = alpha[:,i]
        mse += torch.mul(weight, torch.sum((preds[:,i,:]-labels)**2, axis=1)).reshape(-1,1) # #rows=#batches
    return torch.mean(mse)

def loss_weightedMAE(alpha, preds, labels):
    mae = torch.zeros(alpha.shape[0],1)
    for i in range(preds.shape[1]):
        weight = alpha[:,i]
        mae += torch.mul(weight, torch.sqrt(torch.sum((preds[:,i,:]-labels)**2, axis=1))).reshape(-1,1)
    return torch.mean(mae)