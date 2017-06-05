#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 14:40:10 2017

@author: dos
"""

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
    #matrix code
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

for i_episode in range(10):
    
    observation = env.reset()
    for t in range(100):
        #env.render()
        #print("Past:",observation)
        prev_observation = observation
        J_prev = J
        
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        
        data_x.append([prev_observation])
        data_y.append([observation])
        data_u.append([action])
        
        print("data collection done")

    J_c = 10
    while (J_c > 1):
        J_c = 0
        for i in range(100):
            predicted_observation, K = rbf_kernel(data_x[i],data_u[i])
            J_c += np.dot((data_y[i] - predicted_observation),(data_y[i] - predicted_observation).T)
            dJdw = - 2 * np.outer((data_y[i] - predicted_observation),K)
            W = W + 0.00001*dJdw
        
        print(J_c)
        

#sio.savemat('data.mat', {'input':data_x,'action':data_u,'output':data_y})
        
#with open('transition_data.pickle', 'wb') as fp:
#    pickle.dump(data_collection, fp)