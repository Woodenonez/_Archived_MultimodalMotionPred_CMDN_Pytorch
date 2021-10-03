import os, sys
import math, timeit

import torch
from torch import nn
from torch import tensor as ts


class AdaptiveMixtureDensityModule(nn.Module):
    def __init__(self, dim_input, dim_output, num_components):
        super(AdaptiveMixtureDensityModule, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.M = num_components

        self.layer_mapping = nn.Linear(dim_input, 2*dim_output*num_components)
        self.sfx = nn.Softmax(dim=1) # If 1, go along each row
        self.sigmoid = nn.Sigmoid()

    # def forward(self, x):
        # self.p = self.layer_mapping(x)
        # self.p.retain_grad()
        # self.alpha = self.layer_alpha(self.p[:,:self.M])
        # self.mu    = self.p[:, self.M:(self.dim_output+1)*self.M]
        # self.sigma = torch.exp(self.p[:, (self.dim_output+1)*self.M:])
        # self.mu    = self.mu.view(-1, self.M, self.dim_output)
        # self.sigma = self.sigma.view(-1, self.M, self.dim_output)
        # return self.alpha, self.mu, self.sigma

    def forward(self, x, data):
        # data is the GT
        p = self.layer_mapping(x)
        mu    = p[:,:self.dim_output*self.M]
        sigma = self.sigmoid(p[:, self.dim_output*self.M:])

        prob = self.cal_GauProb(mu, sigma, data) # (BxG)
        adaptive_alpha = self.sfx(prob)

        mu    = mu.view(-1, self.M, self.dim_output)
        sigma = sigma.view(-1, self.M, self.dim_output)
        return adaptive_alpha, mu, sigma

    def cal_GauProb(self, mu, sigma, data):
        """
        Return the probability of "data" given MoG parameters "mu" and "sigma".
        
        Arguments:
            mu    (BxGxC) - The means of the Gaussians. 
            sigma (BxGxC) - The standard deviations of the Gaussians.
            data  (BxC)   - A batch of data points (coordinates of position).

        Return:
            probabilities (BxG): The probability of each point in the probability
                of the distribution in the corresponding mu/sigma index.
                (Assume the dimensions of the output are independent to each other.)
        """
        x = x.unsqueeze(1).expand_as(mu) # BxC -> Bx1xC -> BxGxC
        prob = torch.rsqrt(torch.tensor(2*math.pi)) * torch.exp(-((x - mu) / sigma)**2 / 2) / sigma
        return torch.prod(prob, dim=2) # overall probability for all output's dimensions in each component, BxG 



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