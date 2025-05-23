import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List
import math
import torch.nn.functional as F
import torch as T
import torch.nn as nn
import torch.optim as optim
import symulacja_sieci_sensorowej as sym
from utils import plot_learning_curve


          
          

# Initialize the network
network = sym.SensorNetwork(sym.NETWORK_AREA)
energy_over_time = []
data_over_time = []
n_games = 500


for _ in range(n_games):
# Sensor network simulation
    for t in range(sym.DURATION):
        network.step()
        network.sensors[0].agent.
        
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = network.sensors[0].agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            network.sensors[0].agent.store_transition(observation, action, reward,
                                  observation_, done)
            network.sensors[0].agent.learn()
            observation = observation_
        network.sensors[0].scores.append(score)
        network.sensors[0].eps_history.append(network.sensors[0].agent.epsilon)

        avg_score = np.mean(network.sensors[0].scores[-100:])

        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % network.sensors[0].agent.epsilon)
    
    x = [i+1 for i in range(n_games)]
    filename = 'wyuczenie _sieci.png'
    plot_learning_curve(x, network.sensors[0].scores, network.sensors[0].eps_history, filename)

# Convert recorded data to NumPy arrays
# energy_over_time = np.array(energy_over_time)
# data_over_time = np.array(data_over_time)

print(len(network.sensors[0].colect_data_by_network))

