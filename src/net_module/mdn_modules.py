import os, sys
import math, timeit

import torch
from torch import nn
from torch import tensor as ts


class ClassicMixtureDensityModule(nn.Module):
    def __init__(self, dim_input, dim_output, num_components):
        super(ClassicMixtureDensityModule, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.M = num_components

        self.layer_mapping = nn.Linear(dim_input, (2*dim_output+1)*num_components)
        self.layer_alpha = nn.Softmax(dim=1) # If 1, go along each row

    def forward(self, x):
        self.p = self.layer_mapping(x)
        self.p.retain_grad()
        self.alpha = self.layer_alpha(self.p[:,:self.M])
        self.mu    = self.p[:, self.M:(self.dim_output+1)*self.M]
        self.sigma = torch.exp(self.p[:, (self.dim_output+1)*self.M:])
        self.mu    = self.mu.view(-1, self.M, self.dim_output)
        self.sigma = self.sigma.view(-1, self.M, self.dim_output)
        return self.alpha, self.mu, self.sigma

    # def forward(self, x):
    #     p = self.layer_mapping(x)
    #     alpha = self.layer_alpha(p[:,:self.M])
    #     mu    = p[:,self.M:(self.dim_output+1)*self.M]
    #     sigma = torch.exp(p[:, (self.dim_output+1)*self.M:])
    #     mu    = mu.view(-1, self.M, self.dim_output)
    #     sigma = sigma.view(-1, self.M, self.dim_output)
    #     return alpha, mu, sigma


class HeuristicMixtureDensityModule(nn.Module):
    def __init__(self, dim_input, dim_output, num_components):
        super(HeuristicMixtureDensityModule, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.M = num_components

        self.layer_mapping = nn.Linear(dim_input, (2*dim_output+1)*num_components)
        self.layer_alpha = nn.Softmax(dim=1) # If 1, go along each row
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        p = self.layer_mapping(x)
        pre_alpha = torch.sub(p[:,:self.M], torch.max(p[:,:self.M], axis=1).values.unsqueeze(1))

        alpha = self.layer_alpha(pre_alpha) # heuristics 1
        mu    = p[:,self.M:(self.dim_output+1)*self.M]
        sigma = self.sigmoid(p[:, (self.dim_output+1)*self.M:])  # heuristics 2
        mu    = mu.view(-1, self.M, self.dim_output)
        sigma = sigma.view(-1, self.M, self.dim_output)
        return alpha, mu, sigma


class SoftSigmaMixtureDensityModule(nn.Module):
    def __init__(self, dim_input, dim_output, num_components):
        super(SoftSigmaMixtureDensityModule, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.M = num_components

        self.layer_mapping = nn.Linear(dim_input, (2*dim_output+1)*num_components)
        self.layer_alpha = nn.Softmax(dim=1) # If 1, go along each row
        self.layer_sigma = ReExp_Layer()

    def forward(self, x):
        p = self.layer_mapping(x)
        alpha = self.layer_alpha(p[:,:self.M])
        mu    = p[:,self.M:(self.dim_output+1)*self.M]
        sigma = self.layer_sigma(p[:, (self.dim_output+1)*self.M:])
        mu    = mu.view(-1, self.M, self.dim_output)
        sigma = sigma.view(-1, self.M, self.dim_output)
        return alpha, mu, sigma


class SplitMixtureDensityModule(nn.Module):
    def __init__(self, dim_input, dim_output, num_components):
        super(SplitMixtureDensityModule, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.M = num_components

        self.activate = nn.ReLU()
        self.layer_alpha = nn.Sequential(
            nn.Linear(dim_input, num_components),
            nn.Softmax(dim=1) # If 1, go along each row
        )
        self.layer_mu    = nn.Sequential(
            nn.Linear(dim_input, dim_output*num_components)
        )
        self.layer_sigma = nn.Sequential(
            nn.Linear(dim_input, dim_output*num_components)
        )

    def forward(self, x):
        alpha = self.layer_alpha(self.activate(x))
        mu    = self.layer_mu(self.activate(x))
        sigma = torch.exp(self.layer_sigma(self.activate(x)))
        mu    = mu.view(-1, self.M, self.dim_output)
        sigma = sigma.view(-1, self.M, self.dim_output)
        return alpha, mu, sigma


class SamplingMixtureDensityModule(nn.Module):
    def __init__(self, dim_input, num_components): # number of inputs * dimension of an input
        super(SamplingMixtureDensityModule, self).__init__()
        self.dim_input  = dim_input
        self.M = num_components

        self.myMLP = nn.Sequential(nn.Linear(dim_input, num_components))
        self.sfx = nn.Softmax(dim=1) # for each batch

    def forward(self, x): # x as a feature vector, in nbatch * dimension of x
        '''
            gamma = r1,1 r1,2 ... r1,M
                    r2,1 r2,2 ... r2,M
                    ...
                    rN,1 rN,2 ... rN,M
            alpha = a1, a2, ... aM
            mu = u1,1 ...
                 u2,1 ...
                 ...
                 uM,1 ...
        '''
        gamma = self.sfx(self.myMLP(x)) # assigning weight [r1, r2, ..., rM]
        alpha = torch.sum(gamma, axis=0)/x.shape[0]
        mu    = torch.zeros(self.M, self.dim_input).float().cuda()
        sigma = torch.zeros(self.M, self.dim_input).float().cuda()
        for i in range(self.M):
            mu[i,:]    = torch.sum(gamma[:,i].unsqueeze(1)*x,              axis=0) / torch.sum(gamma[:,i])
            sigma[i,:] = torch.sum(gamma[:,i].unsqueeze(1)*(x-mu[i,:])**2, axis=0) / torch.sum(gamma[:,i]) # simplified
        return alpha.unsqueeze(0), mu.unsqueeze(0), sigma.unsqueeze(0)





class ReExp_Layer(nn.Module):
    '''
    Description:
        A rectified exponential layer.
        Only the negative part of the exponential retains.
        The positive part is linear: y=x+1.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        l = nn.ELU() # ELU: max(0,x)+min(0,α∗(exp(x)−1))
        return torch.add(l(x), 1) # assure no negative sigma produces!!! 