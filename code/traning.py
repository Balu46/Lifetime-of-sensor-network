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
from Hyperparamiters import N_GAMES_FOR_TRANING




def traning():
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
    n_games = N_GAMES_FOR_TRANING
    best_score = 0
    
    path_for_log_energy = "/home/jan/Informatyka/Projekt_indywidualny/logs/stan_energi_po_symulacji.txt"
    
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
            action = network.sensors[0].agent.choose_action(observation)
            observation_, reward, done, info = network.step(action)
            score += reward
            network.sensors[0].agent.store_transition(observation, action, reward,
                            observation_, done)
            network.sensors[0].agent.learn()
            observation = observation_
            t+=1
        scores.append(score)
        eps_history.append(network.sensors[0].agent.epsilon)

        avg_score = np.mean(scores[-100:])
        if score > best_score:
            best_score = score
            network.sensors[0].agent.update_best_model()
                
        
        with open(path_for_log_energy, "a") as file:
            file.write(f"Game {i}:\n")
            file.write(f"Time used: {t}\n")
            file.write(f"Dane zebrane przez główną jednostkę: {len(network.sensors[0].colect_data_by_network)}\n")
            file.write(f"Energia pozostała w sensorach: {[s.energy for s in network.sensors[1:]]}\n\n\n")    


        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % network.sensors[0].agent.epsilon)
    
    network.sensors[0].agent.save_best_model('/home/jan/Informatyka/Projekt_indywidualny/code/models/best_model.pth')
    
    
      
          
          

traning()
   
   
   
   
    
# x = [i+1 for i in range(N_GAMES)]
# filename = 'wyuczenie _sieci.png'
# plot_learning_curve(x, scores, eps_history, filename)



# # Wyniki
# print(f"Dane zebrane przez główną jednostkę: {len(network.sensors[0].colect_data_by_network)}")
# print(f"Energia pozostała w sensorach: {[s.energy for s in network.sensors[1:]]}")

# # Wizualizacja
# plt.figure(figsize=(10, 8))
# plt.scatter(
#     [s.position[0][0] for s in network.sensors],
#     [s.position[0][1] for s in network.sensors],
#     c=['red' if s.id == 0 else 'blue' for s in network.sensors],
#     label='Główna jednostka (czerwony)'
# )
# plt.title("Pozycje sensorów w sieci")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.grid(True)
# plt.legend()
# plt.savefig("/home/jan/Informatyka/Projekt_indywidualny/temp4.png")
