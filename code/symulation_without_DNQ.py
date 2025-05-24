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

N_GAMES = 100


def symulation():
    """
    Function to train the sensor network.
    This function initializes the sensor network and runs a series of training episodes.
    Each episode consists of a series of time steps where the sensors interact with the environment.
    The training process involves choosing actions, receiving rewards, and updating the agent's knowledge.
    """
    # Initialize the network
    network = sym.SensorNetwork(sym.NETWORK_AREA)
    energy_over_time = []
    data_over_time = []
    scores ,eps_history = [], []
    n_games = N_GAMES
    best_score = 0
    
    path_for_log_energy = "/home/jan/Informatyka/Projekt_indywidualny/logs/stan_energi_po_symulacji_bez_niczego.txt"
    
    # create a file to store the results
    # This file will be used to log the energy state of the sensors after each game
    with open(path_for_log_energy, "w") as f:
        pass 

    for i in range(n_games):
    # Sensor network simulation
        t = 0 #time spent in symulation
            
        score = 0
        done = False
        observation = network.reset()
        while not done:
            action = None
            observation_, reward, done, info = network.step(action)
            score += reward
            t+=1
        scores.append(score)

                
        
        with open(path_for_log_energy, "a") as file:
            file.write(f"Game {i}:\n")
            file.write(f"Time used: {t}\n")
            file.write(f"Dane zebrane przez główną jednostkę: {len(network.sensors[0].colect_data_by_network)}\n")
            file.write(f"Energia pozostała w sensorach: {[s.energy for s in network.sensors[1:]]}\n\n\n")    
    
          
          

symulation()
   
   
   
   