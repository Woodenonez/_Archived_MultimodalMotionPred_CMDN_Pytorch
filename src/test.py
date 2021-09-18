import os, sys

import numpy as np

import torch
from torch import nn
from torch import tensor as ts
import matplotlib.pyplot as plt

def s(x):
    return 1/(1+np.exp(-x))

def e(x):
    return np.exp(x)

def softmax(xs):
    return e(xs)/sum(e(xs))

z = np.linspace(-5,5,500)

s0 = e(z)
s1 = e(s(z))
s2 = s(e(z))
s3 = s(z)

# ds0 = e(z)
ds1 = e(s(z)) * s(z) * (1-s(z))
ds2 = s(e(z)) * (1-s(e(z))) * e(z)
ds3 = s(z) * (1-s(z))

# plt.plot(z,ds1,'b')
# plt.plot(z,ds2,'r')
# plt.plot(z,ds3,'g')
# # plt.plot(z,ds0,'k')
# plt.xlabel('z_(\sigma)')
# plt.ylabel('d\sigma')
# plt.show()

# plt.plot(z,s1,'b')
# plt.plot(z,s2,'r')
# plt.plot(z,s3,'g')
# # plt.plot(z,ds0,'k')
# plt.xlabel('z_(\sigma)')
# plt.ylabel('sigma')
# plt.show()

a = 0
b = a-1
c = b-1

r = softmax(np.array([s(a),s(b),s(c)]))
print(r)