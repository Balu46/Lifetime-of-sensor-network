import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List
import math
import torch.nn.functional as F
import torch as T
import torch.nn as nn
import torch.optim as optim


# Parameters of the sensor network
NUM_OF_MAIN_UNITS = 1
NUM_OF_SENSORS_IN_NETWORK = 10 # Number of sensor nodes

BATTERY_CAPACITY = 100  # Maximum energy capacity (in units)

TRANSMISSION_COST = 5  # Energy cost per data transmission
MINING_COST = 1

DURATION = 10  # Number of simulation steps

NETWORK_AREA = (100, 100)  # Network area size (units)

NUM_OF_DATA_IN_BACH = 10

NUM_NODES = NUM_OF_MAIN_UNITS + NUM_OF_SENSORS_IN_NETWORK


class DeepQNetwork(nn.Module):
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

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, 
                                 input_dims=input_dims, 
                                 fc1_dims=256, fc2_dims=256)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
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


class Data:
    def __init__(self):
        self.data = -1
        self.communique = ''
        self.device = -1 ,# from what device each data is
        self.destination = None #where data should go 
        
    def create_mined_data(self,device, destination) -> None:
        self.data = 1
        self.communique = 'mined data'
        self.device = device
        self.destination = destination
              



class Sensor:
    def __init__(self, id: int, main_unit_pos, area_size = NETWORK_AREA, battery_capasity = BATTERY_CAPACITY):    
        
        self.main_unit_position = main_unit_pos
        self.energy = battery_capasity
        self.area_size = area_size
        
        self.position = np.random.rand(1, 2) * self.area_size  # Random sensor positions
        self.coverage_radius = np.random.uniform(10, 20, 1)
        self.transfer_coverage_distance = self.coverage_radius * 4
        
        self.data_to_transfer  = []
        self.recived_data = []
        self.routing_table = []
        
        self.is_active = True
        self.sleep_time = 0
        
        self.id = id
        
      
    def reset(self): 
        self.energy = BATTERY_CAPACITY
        self.data_to_transfer  = []
        self.recived_data = []
        self.routing_table = []
        self.is_active = True
        self.sleep_time = 0
        
    def recive_data(self, data: Data) -> int:
        if data.destination == self.id:
            self.recived_data.append(data)
        else:
            self.send_data(data)
        return 1
        
    def colect_data(self) -> bool:
        # self.energy -= MINING_COST
        mined = Data()
        mined.create_mined_data(id, 0) # 0 is always id of main unit
        if self.send_data(mined):
            return True
        return False
        
    def go_to_sleep(self,time: int) -> bool:
        self.is_active = False
        self.sleep_time = time
        return True
        
    def send_data(self, data: Data) -> bool:
        if not self.routing_table:
            return False

        # Wybierz sąsiada najbliższego do głównej jednostki (GPSR)
        best_neighbor = min(
            self.routing_table,
            key=lambda neighbor: np.linalg.norm(neighbor.position - self.main_unit_position),
        )
        
        if best_neighbor.is_active:
            best_neighbor.recive_data(data)
            # self.energy -= TRANSMISSION_COST
            return True
        return False
        
class main_unit(Sensor):
    def __init__(self, id: int, area_size = NETWORK_AREA, battery_capasity = BATTERY_CAPACITY):
        super(main_unit,self).__init__(id,area_size,battery_capasity)
       
        self.communicates = []
        self.colect_data_by_network = []
        self.transfer_coverage_distance = (float)(np.random.uniform(20, 40, 1)) 
        self.agent = Agent(gamma=0.99, epsilon=1, batch_size=64, n_actions=NUM_OF_SENSORS_IN_NETWORK, eps_end=0.01, input_dims=8,   lr = 0.003  )
        
        
    def recive_data(self, data: Data) -> bool:
        if data.destination == self.id:
            self.colect_data_by_network.append(data)
        else:
            self.send_data(data)   
            
   


def distance_between_2_sensors(sensor1: Sensor, sensor2: Sensor):
    # return math.sqrt((sensor2.position[0] - sensor1.position[0])**2 + (sensor2.position[1] - sensor1.position[1])**2)
    return np.linalg.norm((sensor2.position) - (sensor1.position))
   
        
class SensorNetwork:
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
                        print(f'{sensor.id} : {sensor_2.id}')
      
        
    
    
    def step(self):
        """Each sensor randomly decides whether to transmit data, consuming energy and transfering it to main unit."""
        for i in range(1,self.num_sensors+1):
            self.sensors[i].colect_data()
            
            
    # def is_active(self):
    #     """Checks if at least one sensor still has enough energy to operate."""
    #     return np.any(self.energy_levels > TRANSMISSION_COST)


          
          

# Initialize the network
network = SensorNetwork(NETWORK_AREA)
energy_over_time = []
data_over_time = []

# Sensor network simulation
for t in range(DURATION):
    # if not network.is_active():
    #     break  # Stop simulation when all sensors are depleted
    network.step()
    # energy_over_time.append(network.energy_levels.copy())
    # data_over_time.append(network.data_collected.copy())

# Convert recorded data to NumPy arrays
# energy_over_time = np.array(energy_over_time)
# data_over_time = np.array(data_over_time)

print(len(network.sensors[0].colect_data_by_network))

