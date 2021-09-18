import os, sys
import math, timeit

import torch
from torch import tensor as ts
from torch import nn

class BackboneNetwork():
    def __init__(self, some_parameters):
        pass

class EWTA_MDF():
    def __init__(self, inputs, labels, num_hypos):
        self.inputs = inputs  # batch * input dimension
        self.labels = labels  # batch * output dimension, each output is a vector
        self.dim_output = labels.shape[1]
        self.M = num_hypos

    def disassembling(self, hypos): # hypos in (batch * (hypos * output dimension))
        hypo_splits = torch.split(hypos, [self.dim_output for _ in range(self.M)], 1)
        return hypo_splits

    def assembling(self, hypo_splits):
        hypos = torch.cat(hypo_splits, axis=1)
        return hypos

    def forward(self, data):
        # backbone = BackboneNetwork(some_parameters)
        # outputs = backbone(self.inputs)
        outputs = data # XXX
        out_hypo_splits = self.disassembling(outputs)
        print(out_hypo_splits)
        out_hypos = self.assembling(out_hypo_splits)
        
        ### Fitting
        pre_predicted = SomePreNetwork(out_hypos)



if __name__ == '__main__':

    inputs  = torch.zeros(1,10)
    outputs = torch.zeros(1,2)
    num_hypos = 10

    data = torch.rand(1, num_hypos*outputs.shape[1])
    print(data)

    this = EWTA_MDF(inputs, outputs, num_hypos)
    this.forward(data)