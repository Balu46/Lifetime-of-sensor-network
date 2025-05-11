import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List


# Parameters of the sensor network
NUM_OF_SENSORS_IN_NETWORK = 10 # Number of sensor nodes
BATTERY_CAPACITY = 100  # Maximum energy capacity (in units)
TRANSMISSION_COST = 5  # Energy cost per data transmission
MINING_COST = 1
TIME_STEPS = 50  # Number of simulation steps
NETWORK_AREA = (100, 100)  # Network area size (units)
NUM_OF_DATA_IN_BACH = 10
NUM_OF_MAIN_UNITS = 1


class Data:
    def __init__(self):
        self.data = -1
        self.communique = ''
        self.device = -1# from what device each data is
        self.destination = '' #where data should go 
    def create_mined_data(self,device, destination) -> None:
        self.data = 1
        self.communique = 'mined data'
        self.device = device
        self.destination = destination
              



class Sensor:
    def __init__(self, id: int, area_size = NETWORK_AREA, battery_capasity = BATTERY_CAPACITY):
        self.energy = battery_capasity
        self.pos = (random.randint(0, area_size[0]), random.randint(0, area_size[1]))
        self.data_to_transfer  = []
        self.recived_data = []
        self.routing_table = []
        self.is_active = True
        self.sleep_time = 0
        self.id = -1
        
        
    def recive_data(self, data: Data) -> int:
        if data.destination == f"{self.id}":
            self.recived_data.append(data)
        else:
            self.send_data(data)
        
    def colect_data(self) -> Data:
        # self.energy -= MINING_COST
        mined = Data()
        mined.create_mined_data(id,'main')
        if self.send_data(mined):
            return 1
        
    def go_to_sleep(self,time: int) -> bool:
        self.is_active = False
        self.sleep_time = time
        return True
        
    def send_data(self, data: Data) -> bool:
        flag = False
        for sensor in self.near_sensors:
            if sensor.is_active:
                sensor.recived_data.append(data)
                # self.energy -= TRANSMISSION_COST  
                flag = True
        return flag        
        
class main_unit(Sensor):
    def __init__(self, id: int, area_size = NETWORK_AREA, battery_capasity = BATTERY_CAPACITY):
        super(main_unit,self).__init__(id,area_size,battery_capasity)
       
        self.communicates = []
        self.colect_data_by_network = []
       
        
class SensorNetwork:
    def __init__(self, area_size, number_of_main_units, num_nodes):
        self.sensors = []
        self.main_unit = main_unit(0)
        
        for i in range(1, NUM_OF_SENSORS_IN_NETWORK + 1):
            self.sensors.append(Sensor(i))

            
        self.num_nodes = num_nodes
        self.area_size = area_size
        
        self.coverage_radius_main_unit_transfer = np.random.uniform(20, 40, number_of_main_units)
        self.reset()
        self.units_routing_table = self.routing_table()
        
    def reset(self):
        """Initializes sensor nodes with full battery, random positions, and coverage radius."""
        self.energy_levels = np.full(self.num_nodes, BATTERY_CAPACITY)
        self.data_collected = np.zeros(self.num_nodes)
        self.positions = np.random.rand(self.num_nodes, 2) * self.area_size  # Random sensor positions
        self.coverage_radius = np.random.uniform(10, 20, self.num_nodes)
        self.coverage_radius_transfer =  self.coverage_radius * 2 # Sensor coverage radiusreset for transmiting data
        return self.energy_levels.copy()
    
    def step(self):
        """Each sensor randomly decides whether to transmit data, consuming energy and transfering it to main unit."""
        for i in range(self.num_nodes):
            if self.energy_levels[i] > TRANSMISSION_COST and random.random() > 0.3 :  # 70% chance to transmit
                for ii in range(self.num_nodes):
                    if ii == i:
                        continue
                    if (self.positions[i][0] + self.coverage_radius_transfer[i] >= self.positions[ii][0] and
                        self.positions[i][1] + self.coverage_radius_transfer[i] >= self.positions[ii][1]):
                        
                        self.energy_levels[i] -= TRANSMISSION_COST
                        self.data_collected[i] += self.coverage_radius[i] / 10  # More data collected with a larger coverage radius
                        break
            
    def is_active(self):
        """Checks if at least one sensor still has enough energy to operate."""
        return np.any(self.energy_levels > TRANSMISSION_COST)

          
          
          
                
        
# Sensor network simulation class
# class SensorNetwork:
#     def __init__(self, num_nodes, area_size, number_of_main_units):
#         self.num_nodes = num_nodes
#         self.area_size = area_size
#         self.main_unit_pos = (np.random.rand(number_of_main_units ,2)* self.area_size).squeeze()
#         self.coverage_radius_main_unit_transfer = np.random.uniform(20, 40, number_of_main_units)
#         self.reset()
#         self.units_routing_table = self.routing_table()
        
#     def reset(self):
#         """Initializes sensor nodes with full battery, random positions, and coverage radius."""
#         self.energy_levels = np.full(self.num_nodes, BATTERY_CAPACITY)
#         self.data_collected = np.zeros(self.num_nodes)
#         self.positions = np.random.rand(self.num_nodes, 2) * self.area_size  # Random sensor positions
#         self.coverage_radius = np.random.uniform(10, 20, self.num_nodes)
#         self.coverage_radius_transfer =  self.coverage_radius * 2 # Sensor coverage radiusreset for transmiting data
#         return self.energy_levels.copy()
    
#     def step(self):
#         """Each sensor randomly decides whether to transmit data, consuming energy and transfering it to main unit."""
#         for i in range(self.num_nodes):
#             if self.energy_levels[i] > TRANSMISSION_COST and random.random() > 0.3 :  # 70% chance to transmit
#                 for ii in range(self.num_nodes):
#                     if ii == i:
#                         continue
#                     if (self.positions[i][0] + self.coverage_radius_transfer[i] >= self.positions[ii][0] and
#                         self.positions[i][1] + self.coverage_radius_transfer[i] >= self.positions[ii][1]):
                        
#                         self.energy_levels[i] -= TRANSMISSION_COST
#                         self.data_collected[i] += self.coverage_radius[i] / 10  # More data collected with a larger coverage radius
#                         break
            
#     def is_active(self):
#         """Checks if at least one sensor still has enough energy to operate."""
#         return np.any(self.energy_levels > TRANSMISSION_COST)

#     def routing_table(self) -> np.array:
#         routing_table = []
#         for i in range(self.num_nodes):
#             help_table = []
#             for ii in range(self.num_nodes):
#                     if (self.positions[i][0] + self.coverage_radius_transfer[i] >= self.positions[ii][0] and
#                         self.positions[i][1] + self.coverage_radius_transfer[i] >= self.positions[ii][1]):
#                             help_table.append(ii)
#                     else:
#                         help_table.append(-1)
#             routing_table.append(help_table)
#         print(routing_table)    
#         return np.array(routing_table)
                        
            
# Initialize the network
network = SensorNetwork(NUM_OF_SENSORS_IN_NETWORK, NETWORK_AREA, NUM_OF_MAIN_UNITS)
energy_over_time = []
data_over_time = []

# Sensor network simulation
for t in range(TIME_STEPS):
    if not network.is_active():
        break  # Stop simulation when all sensors are depleted
    network.step()
    energy_over_time.append(network.energy_levels.copy())
    data_over_time.append(network.data_collected.copy())

# Convert recorded data to NumPy arrays
energy_over_time = np.array(energy_over_time)
data_over_time = np.array(data_over_time)










# Visualization of energy levels over time
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i in range(NUM_NODES):
    plt.plot(energy_over_time[:, i], label=f"Sensor {i}")
plt.xlabel("Time")
plt.ylabel("Energy Level")
plt.title("Energy Level Changes in Sensors")
plt.legend()

# Visualization of collected data over time
energy_over_time = np.array(energy_over_time)
data_over_time = np.array(data_over_time)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i in range(NUM_NODES):
    plt.plot(energy_over_time[:, i], label=f"Sensor {i}")
plt.xlabel("Czas")
plt.ylabel("Poziom energii")
plt.title("Zmiana poziomu energii w sensorach")
plt.legend()

plt.subplot(1, 2, 2)
for i in range(NUM_NODES):
    plt.plot(data_over_time[:, i], label=f"Sensor {i}")
plt.xlabel("Czas")
plt.ylabel("Liczba przesłanych pakietów")
plt.title("Zebrane dane przez sensory")
plt.legend()

plt.tight_layout()
plt.savefig("/home/jan/Informatyka/Projekt_indywidualny/temp2.png")

# Visualization of sensor positions and their coverage areas
plt.figure(figsize=(6, 6))
plt.scatter(network.positions[:, 0], network.positions[:, 1], c='blue', label='Sensors')
for i in range(NUM_NODES):
    circle = plt.Circle((network.positions[i, 0], network.positions[i, 1]), network.coverage_radius_transfer[i], color='r', alpha=0.3)
    circle2 = plt.Circle((network.positions[i, 0], network.positions[i, 1]), network.coverage_radius[i], color='g', alpha=0.3)
    plt.gca().add_patch(circle)
    plt.gca().add_patch(circle2)

plt.scatter(network.main_unit_pos[0], network.main_unit_pos[ 1], c='green', label='Main Unit')

circle = plt.Circle((network.main_unit_pos[0], network.main_unit_pos[1]), network.coverage_radius_main_unit_transfer, color='r', alpha=0.3)
plt.gca().add_patch(circle)

plt.xlim(0, NETWORK_AREA[0])
plt.ylim(0, NETWORK_AREA[1])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Położenie sensorów i ich obszary zasięgu")
plt.legend()
plt.savefig("/home/jan/Informatyka/Projekt_indywidualny/temp.png")
