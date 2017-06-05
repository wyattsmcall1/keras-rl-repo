#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class project CS450
@author: deniso2@illinois.edu
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import gym
#import scipy.io as sio

from gym import wrappers
env = gym.make('Pendulum-v0')

data_x = [] # storage for all transitions
data_y = []
data_u = []

#initialize RBF kernel centers
N = 100 # n of kernels
centers = np.array([np.linspace(-1,1,N),
                    np.linspace(-1,1,N),
                    np.linspace(-8,8,N),
                    np.linspace(-2,2,N)])
W = np.random.rand(3,N)
var = np.diag([2./N,2./N,16./N,4./N])
J = []

def rbf_kernel(observation,action):
    #matrix form code
    #obs = np.repeat([np.append(observation,action)],N,axis=0)
    #print(obs)
    #delta = (obs.T - centers)
    #print(delta)
    #K = np.exp(-np.sum(delta*delta,axis=0)/(2*0.01))
    #print(K)
    #prediction = np.dot(W,K.T)
    K_all = np.zeros(N)
    obs = np.append(observation,action)
    for i in range(N):
        delta = obs - centers.T[i]
        K = np.exp(-np.dot(np.dot(delta,var),delta))
        K_all[i] = K
        
    prediction = np.dot(K_all, W.T)
    #print(prediction)
    return [prediction,K_all]

J_hist = []

for i_episode in range(1000):
    observation = env.reset()
    J_sum = 0
    for t in range(100):
        #env.render()
        #print("Past:",observation)
        prev_observation = observation
        J_prev = J
        
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        
        #data_x.append([prev_observation])
        #data_y.append([observation])
        #data_u.append([action])
        
        predicted_observation, K = rbf_kernel(prev_observation,action)
        #np.reshape(predicted_observation,0)
                               
        J = np.dot((observation - predicted_observation),(observation - predicted_observation))
        
        
        dJdw = 2 * np.outer((observation - predicted_observation),K)
        
        W = W - 0.000001*dJdw
        
        #print(J)
        J_sum += J
        
    print(i_episode,':',J_sum)
    if (J_sum > 10**9):
        break
    J_hist.append(J_sum)
    
    
plt.figure(0)
plt.plot(J_hist)
plt.title("Convergence of SGD optimization")
plt.xlabel("Episodes")
plt.ylabel("Loss function accumulated")
plt.show()

#sio.savemat('data.mat', {'input':data_x,'action':data_u,'output':data_y})
        
#with open('transition_data.pickle', 'wb') as fp:
#    pickle.dump(data_collection, fp)