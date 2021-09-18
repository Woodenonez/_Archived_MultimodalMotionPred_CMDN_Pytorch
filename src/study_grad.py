import os, sys
import math

import numpy as np
import matplotlib.pyplot as plt

def phi(x, mu, sigma): # Gaussian distribution
    return np.exp(-np.power(x-mu,2) / (2*np.power(sigma, 2))) / (sigma*np.sqrt(2*math.pi))

def gmm(alpha_vec, phi_vec):
    return np.dot(alpha_vec.transpose(), phi_vec)

def posterior(alpha_vec, phi_vec):
    return np.multiply(alpha_vec, phi_vec) / gmm(alpha_vec, phi_vec)

def grad_alpha(alpha_vec, phi_vec):
    return -posterior(alpha_vec, phi_vec)/alpha_vec

def grad_mu(mu_vec, sigma_vec, phi_vec, x):
    return posterior(alpha_vec, phi_vec) * (mu_vec-x) / sigma_vec**2

def grad_sigma(mu_vec, sigma_vec, phi_vec, x):
    return -posterior(alpha_vec, phi_vec) * (np.linalg.norm(mu_vec-x)**2/sigma_vec**3 - 1/sigma_vec)

x = np.array([10])

mu1 = np.array([9])
mu2 = np.array([8])
sig1 = np.array([1])
sig2 = np.array([1])
alp1 = 0.5
alp2 = 1-alp1
alpha_vec = np.array([[alp1],[alp2]])

phi_vec = phi(x, np.vstack((mu1,mu2)), np.vstack((sig1,sig2)))
print(f'phi1, phi2: {phi_vec.transpose()}')

mix = gmm(alpha_vec, phi_vec)
print(f'GMM: {mix}')

post = posterior(alpha_vec, phi_vec)
print(f'post1, post2: {post.transpose()}')

g_alpha = grad_alpha(alpha_vec, phi_vec)
print(f'grad_alp1, grad_alp2, ratio: {g_alpha.transpose()} {g_alpha[0]/g_alpha[1]}')

g_mu = grad_mu(np.vstack((mu1,mu2)), np.vstack((sig1,sig2)), phi_vec, x)
print(f'grad_mu1, grad_mu2: {g_mu.transpose()}')

g_sigma = grad_sigma(np.vstack((mu1,mu2)), np.vstack((sig1,sig2)), phi_vec, x)
print(f'grad_sigma1, grad_sigma2: {g_sigma.transpose()}')

# x1 = np.linspace(mu1-3*sig1, mu1+3*sig1, 100)
# y1 = phi(x1, mu1, sig1)
# plt.plot(x1, y1, 'b.')
# x2 = np.linspace(mu2-3*sig2, mu2+3*sig2, 100)
# y2 = phi(x2, mu2, sig2)
# plt.plot(x2, y2, 'r.')
# plt.plot(x, max(y1), 'kx')
# plt.show()