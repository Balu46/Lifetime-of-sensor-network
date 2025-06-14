import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List
import math
import torch.nn.functional as F
import torch as T
import torch.nn as nn
import torch.optim as optim
import os
from Hyperparamiters import *
import json

class DeepQNetwork(nn.Module):
    """Implementacja sieci neuronowej DQN do uczenia przez wzmacnianie.
    
    Atrybuty:
        input_dims (int): Wymiar wejścia sieci
        fc1_dims (int): Rozmiar pierwszej warstwy ukrytej
        fc2_dims (int): Rozmiar drugiej warstwy ukrytej
        n_actions (int): Liczba możliwych akcji
        device (torch.device): Urządzenie do obliczeń (CPU/GPU)
    """
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

class Agent():
    """Agent wykorzystujący DQN do podejmowania decyzji w sieci sensorowej.
    
    Atrybuty:
        gamma (float): Współczynnik dyskontowania
        epsilon (float): Współczynnik eksploracji
        lr (float): Szybkość uczenia
        action_space (list): Przestrzeń możliwych akcji
        mem_size (int): Rozmiar pamięci doświadczeń
        Q_eval (DeepQNetwork): Główna sieć Q
        best_Q_eval (DeepQNetwork): Najlepsza sieć Q
    """
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=1000000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.path = '/home/jan/Informatyka/Projekt_indywidualny/code/models/best_model.pth'

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, 
                                 input_dims=input_dims, 
                                 fc1_dims=256, fc2_dims=256)
        
        self.best_Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, 
                                 input_dims=input_dims, 
                                 fc1_dims=256, fc2_dims=256)


        if os.path.exists(self.path) and LOAD_BEST_MODEL:
            self.Q_eval.load_state_dict(T.load(self.path, map_location=self.Q_eval.device))
            self.best_Q_eval.to(self.Q_eval.device)
            print(f"Model załadowany z pliku: {self.path}")
        else:
            print("Brak zapisanego modelu – uruchamianie od zera.")

        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        """Wybierz akcję na podstawie obserwacji (ε-zachłannie).
    
        Args:
            observation: Aktualny stan środowiska
        
        Returns:
            int: Wybrana akcja (indeks sensora do uśpienia)
        """
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                      else self.eps_min
                      
    def update_best_model(self):
        self.best_Q_eval.load_state_dict(self.Q_eval.state_dict())
        
    def save_best_model(self, filename):
        if SAVE_BEST_MODEL:
            T.save(self.best_Q_eval.state_dict(), filename)

class Data:
    def __init__(self):
        self.data = -1
        self.communique = ''
        self.device = -1 ,# from what device each data is
        self.destination = None #where data should go 
        self.route =  [] # route of data, list of sensors that data passed through
        
    def create_mined_data(self,device, destination) -> None:
        self.data = 1
        self.communique = 'mined data'
        self.device = device
        self.destination = destination
              



class Sensor:
    """Pojedynczy sensor w sieci.
    
    Atrybuty:
        id (int): Unikalny identyfikator sensora
        position (np.array): Pozycja (x,y) w przestrzeni 2D
        energy (float): Pozostała energia
        is_sleeping (bool): Czy sensor jest w trybie uśpienia
        routing_table (list): Lista sąsiednich sensorów
    """
    def __init__(self, id: int, main_unit_pos, area_size = NETWORK_AREA, battery_capasity = BATTERY_CAPACITY):    
        
        self.main_unit_position = main_unit_pos
        self.energy = battery_capasity
        self.area_size = area_size
        
        self.position = np.random.rand(1, 2) * self.area_size  # Random sensor positions
        self.coverage_radius = np.random.uniform(10, 20, 1)
        self.transfer_coverage_distance = self.coverage_radius * 3
        
        self.data_to_transfer  = []
        self.recived_data = []
        self.routing_table = []
        
        self.is_sleeping = False
        self.sleep_time = 0
        
        self.id = id
        
      
    def reset(self): 
        self.energy = BATTERY_CAPACITY
        self.data_to_transfer  = []
        self.recived_data = []
        self.is_sleeping = False
        self.sleep_time = 0
        
    def recive_data(self, data: Data) -> int:
        if data.destination == self.id:
            self.recived_data.append(data)
        else:
            self.send_data(data)
        return 1
        
    def colect_data(self) -> bool: # if colected data return True, else False
        if not self.is_active():
            return False
        if self.is_sleeping:
            self.sleep_time -= 1
            if self.sleep_time <= 0:
                self.is_sleeping = False
            return False
        self.energy -= MINING_COST
        mined = Data()
        mined.create_mined_data(id, 0) # 0 is always id of main unit
        mined.route.append(self.id)  # Add current sensor to the route
        if self.send_data(mined):
            return True
        return False
        
    def go_to_sleep(self,time: int) -> bool:
        self.is_sleeping = True
        self.sleep_time = time
        return True
        
    def send_data(self, data: Data) -> bool:
        """Przesyła dane do sąsiada najbliższego głównej jednostki (algorytm GPSR).
    
        Args:
            data (Data): Dane do przesłania
        
        Returns:
            bool: True jeśli przesłanie się powiodło
        """
        if not self.routing_table or not self.is_active():
            return False

        # Wybierz sąsiada najbliższego do głównej jednostki (GPSR)
        best_neighbor = min(
            self.routing_table,
            key=lambda neighbor: np.linalg.norm(neighbor.position - self.main_unit_position),
        )
        
        if best_neighbor.is_active:
            self.energy -= TRANSMISSION_COST
            data.route.append(best_neighbor.id)  # Add current sensor to the route
            best_neighbor.recive_data(data)
            return True
        return False
    
    def is_active(self) -> bool:
        if self.energy <= TRANSMISSION_COST:
            return False
        return True
        
class main_unit(Sensor):
    def __init__(self, id: int, area_size = NETWORK_AREA, battery_capasity = BATTERY_CAPACITY):
        super(main_unit,self).__init__(id,area_size,battery_capasity)
       
        self.communicates = []
        self.colect_data_by_network = []
        self.transfer_coverage_distance = (float)(np.random.uniform(20, 40, 1)) 
        self.agent = Agent(gamma=0.99, epsilon=1, batch_size=64, n_actions=NUM_OF_SENSORS_IN_NETWORK + 1, eps_end=0.01, input_dims=3*NUM_NODES + 2,   lr = 0.003)
        
        
    def recive_data(self, data: Data) -> bool:
        if data.destination == self.id:
            self.colect_data_by_network.append(data)
            # print(f"Main unit received data from sensor {data.device}. Route: {data.route}")
        else:
            self.send_data(data)   
            
    def reset(self): 
        self.energy = BATTERY_CAPACITY
        self.data_to_transfer  = []
        self.recived_data = []
        self.is_sleeping = False
        self.sleep_time = 0
        self.communicates = []
        self.colect_data_by_network = []


def distance_between_2_sensors(sensor1: Sensor, sensor2: Sensor):
    # return math.sqrt((sensor2.position[0] - sensor1.position[0])**2 + (sensor2.position[1] - sensor1.position[1])**2)
    return np.linalg.norm((sensor2.position) - (sensor1.position))
   
        
class SensorNetwork:
    """Główna klasa zarządzająca całą siecią sensorów.
    
    Atrybuty:
        sensors (list): Lista wszystkich sensorów
        area_size (tuple): Rozmiar obszaru sieci
    """

    def __init__(self, area_size, number_of_main_units = NUM_OF_MAIN_UNITS, num_sensors = NUM_OF_SENSORS_IN_NETWORK):
        self.sensors = []
        # self.main_unit = main_unit(0)
        self.sensors.append(main_unit(0))
        self.num_sensors = num_sensors
        self.area_size = area_size
        
        for i in range(1, num_sensors + 1):
            self.sensors.append(Sensor(i,self.sensors[0].position))


        for sensor in self.sensors:
            for sensor_2 in self.sensors:
                if sensor != sensor_2:
                    if distance_between_2_sensors(sensor1=sensor, sensor2=sensor_2) <= (sensor.transfer_coverage_distance):
                        sensor.routing_table.append(sensor_2)   
                        
      
    
    def step(self, action=None):
        """Wykonuje krok symulacji.
    
        Args:
            action: Akcja podjęta przez agenta
        
        Returns:
            tuple: (observation, reward, done, info)
        """
        reward = 0
        done = False

        # 1. Akcja agenta (jeśli jest przekazana)
        if action is not None:
            # Przykład: agent wybiera, który sensor ma zbierać dane
            if not action >= self.num_sensors:
                selected_sensor = self.sensors[action+1]  # +1 because main unit is at index 0
                selected_sensor.go_to_sleep(15)  # set sleep time


       
        for sensor in self.sensors[1:]:
            if sensor.is_active():
                if sensor.colect_data():
                    reward -= 1  # - Penalty for sensors using energy


        # Reward for main unit collecting data
        reward += len(self.sensors[0].colect_data_by_network) * 2

        # - Penalty for sensors with low energy
        for sensor in self.sensors:
            if sensor.energy <= TRANSMISSION_COST or sensor.energy <= MINING_COST:
                reward -= 10

        if not self.is_active():
            done = True

        observation = self.get_observation()

        return observation, reward, done, {}

    def get_observation(self):
        """Zwraca stan środowiska jako wektor numeryczny."""
        observation = []
        for sensor in self.sensors:
            observation.append(sensor.energy / BATTERY_CAPACITY)  # Normalization
        observation.append(len(self.sensors[0].colect_data_by_network))
        for sensor in self.sensors:
            observation.append(sensor.position[0][0] / self.area_size[0])
            observation.append(sensor.position[0][1] / self.area_size[1])
        active_sensors = sum(1 for sensor in self.sensors if sensor.is_active)
        observation.append(active_sensors / self.num_sensors)   
        return np.array(observation, dtype=np.float32)
            
            
    def is_active(self):
        """Checks if at least one sensor still has enough energy to operate."""
        for sensor in self.sensors[1:]: 
            if sensor.is_active():
                return True
        return False
    
    def reset(self):
        for sensor in self.sensors:
            sensor.reset()      
        observation = self.get_observation()
        return observation        
          
          
          
          
def save_history_to_file(history, filename="simulation_history.json"):
    with open(filename, 'w') as f:
        json.dump({
            "history": history
        }, f, indent=2)

def load_history_from_file(filename="simulation_history.json"):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data["history"]          
          
          
          
def symulation() :
    """
    Function to train the sensor network.
    This function initializes the sensor network and runs a series of training episodes.
    Each episode consists of a series of time steps where the sensors interact with the environment.
    The training process involves choosing actions, receiving rewards, and updating the agent's knowledge.
    """
    # Initialize the network
    network = SensorNetwork(NETWORK_AREA)
    scores ,eps_history = [], []
    
    best_score = 0
    
    history = []    
    
    
    path_for_log_energy = "/home/jan/Informatyka/Projekt_indywidualny/logs/stan_energi_po_symulacji.txt"
    
    # create a file to store the results
    # This file will be used to log the energy state of the sensors after each game
    with open(path_for_log_energy, "w") as f:
        pass 

    
    # Sensor network simulation
    t = 0 #time spent in symulation
            
    step = 0
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
        history.append({
            "step": step,
            "main_unit": {
                "id": network.sensors[0].id,
                "energy": network.sensors[0].energy,
                "position": network.sensors[0].position.tolist(),
                "collected_data": len(network.sensors[0].colect_data_by_network),
                "transfer_distance": float(network.sensors[0].transfer_coverage_distance),
                "is_sleeping": network.sensors[0].is_sleeping,
                "coverage_radius": float(network.sensors[0].coverage_radius),
                "data" : [
                {
                    "from_device": data.route[0],
                    "destination" : data.destination,
                    "route" : data.route 
                }
                for data in network.sensors[0].colect_data_by_network
                
                ],
            },

            "sensors": [
                {
                    "id": sensor.id,
                    "energy": sensor.energy,
                    "position": sensor.position.tolist(),
                    "is_sleeping": sensor.is_sleeping,
                    "sleep_time": sensor.sleep_time,
                    "coverage_radius": float(sensor.coverage_radius),
                    "transfer_distance": float(sensor.transfer_coverage_distance)
                }
                for sensor in network.sensors[1:]   
            ],
            "reward": reward,
            "data_collected": len(network.sensors[0].colect_data_by_network)
        })
        step += 1
    
    
    scores.append(score)
    eps_history.append(network.sensors[0].agent.epsilon)

    avg_score = np.mean(scores[-100:])
    if score > best_score:
        best_score = score
        network.sensors[0].agent.update_best_model()
        
    with open(path_for_log_energy, "a") as file:
            file.write(f"Symulacja:\n")
            file.write(f"Time used: {t}\n")
            file.write(f"Dane zebrane przez główną jednostkę: {len(network.sensors[0].colect_data_by_network)}\n")
            file.write(f"Energia pozostała w sensorach: {[s.energy for s in network.sensors[1:]]}\n\n\n")        

    return history
          
     
     
     
def symulation_witout_optymalizacion():
    """
    Function to train the sensor network.
    This function initializes the sensor network and runs a series of training episodes.
    Each episode consists of a series of time steps where the sensors interact with the environment.
    The training process involves choosing actions, receiving rewards, and updating the agent's knowledge.
    """
    # Initialize the network
    network = SensorNetwork(NETWORK_AREA)
    scores ,eps_history = [], []
    
    best_score = 0
    
    history = []    
    
    
    path_for_log_energy = "/home/jan/Informatyka/Projekt_indywidualny/logs/stan_energi_po_symulacji_bez_niczego.txt"
    
    # create a file to store the results
    # This file will be used to log the energy state of the sensors after each game
    with open(path_for_log_energy, "w") as f:
        pass 

    
    # Sensor network simulation
    t = 0 #time spent in symulation
            
    step = 0
    score = 0
    done = False
    observation = network.reset()
    while not done:
        
        observation_, reward, done, info = network.step(None)
        score += reward
        t+=1
        history.append({
            "step": step + 1,
            "main_unit": {
                "id": network.sensors[0].id,
                "energy": network.sensors[0].energy,
                "position": network.sensors[0].position.tolist(),
                "collected_data": len(network.sensors[0].colect_data_by_network),
                "transfer_distance": float(network.sensors[0].transfer_coverage_distance),
                "is_sleeping": network.sensors[0].is_sleeping,
                "coverage_radius": float(network.sensors[0].coverage_radius),
                "data" : [
                {
                    "from_device": data.route[0],
                    "destination" : data.destination,
                    "route" : data.route 
                }
                for data in network.sensors[0].colect_data_by_network
                
                ],
            },

            "sensors": [
                {
                    "id": sensor.id,
                    "energy": sensor.energy,
                    "position": sensor.position.tolist(),
                    "is_sleeping": sensor.is_sleeping,
                    "sleep_time": sensor.sleep_time,
                    "coverage_radius": float(sensor.coverage_radius),
                    "transfer_distance": float(sensor.transfer_coverage_distance)
                }
                for sensor in network.sensors[1:]   
            ],
            "reward": reward,
            "data_collected": len(network.sensors[0].colect_data_by_network)
        })
        step += 1
    
    
    scores.append(score)
    eps_history.append(network.sensors[0].agent.epsilon)

    avg_score = np.mean(scores[-100:])
    if score > best_score:
        best_score = score
        network.sensors[0].agent.update_best_model()
        
    with open(path_for_log_energy, "a") as file:
            file.write(f"Symulacja:\n")
            file.write(f"Time used: {t}\n")
            file.write(f"Dane zebrane przez główną jednostkę: {len(network.sensors[0].colect_data_by_network)}\n")
            file.write(f"Energia pozostała w sensorach: {[s.energy for s in network.sensors[1:]]}\n\n\n")        

    return history
               



     
print("Starting simulation...")