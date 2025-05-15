import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List
import math

# Parameters of the sensor network
NUM_OF_MAIN_UNITS = 1
NUM_OF_SENSORS_IN_NETWORK = 10 # Number of sensor nodes

BATTERY_CAPACITY = 100  # Maximum energy capacity (in units)

TRANSMISSION_COST = 5  # Energy cost per data transmission
MINING_COST = 1

DURATION = 10  # Number of simulation steps

NETWORK_AREA = (100, 100)  # Network area size (units)

NUM_OF_DATA_IN_BACH = 10




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
        
        
    def recive_data(self, data: Data) -> int:
        if data.destination == self.id:
            self.colect_data_by_network.append(data)
        else:
            self.send_data(data)   
            
   


def distance_between_2_sensors(sensor1: Sensor, sensor2: Sensor) -> float:
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
            
            
    def is_active(self):
        """Checks if at least one sensor still has enough energy to operate."""
        return np.any(self.energy_levels > TRANSMISSION_COST)

          

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



# # Visualization of energy levels over time
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# for i in range(NUM_NODES):
#     plt.plot(energy_over_time[:, i], label=f"Sensor {i}")
# plt.xlabel("Time")
# plt.ylabel("Energy Level")
# plt.title("Energy Level Changes in Sensors")
# plt.legend()

# # Visualization of collected data over time
# energy_over_time = np.array(energy_over_time)
# data_over_time = np.array(data_over_time)

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# for i in range(NUM_NODES):
#     plt.plot(energy_over_time[:, i], label=f"Sensor {i}")
# plt.xlabel("Czas")
# plt.ylabel("Poziom energii")
# plt.title("Zmiana poziomu energii w sensorach")
# plt.legend()

# plt.subplot(1, 2, 2)
# for i in range(NUM_NODES):
#     plt.plot(data_over_time[:, i], label=f"Sensor {i}")
# plt.xlabel("Czas")
# plt.ylabel("Liczba przesłanych pakietów")
# plt.title("Zebrane dane przez sensory")
# plt.legend()

# plt.tight_layout()
# plt.savefig("/home/jan/Informatyka/Projekt_indywidualny/temp2.png")

# # Visualization of sensor positions and their coverage areas
# plt.figure(figsize=(6, 6))
# plt.scatter(network.positions[:, 0], network.positions[:, 1], c='blue', label='Sensors')
# for i in range(NUM_NODES):
#     circle = plt.Circle((network.positions[i, 0], network.positions[i, 1]), network.coverage_radius_transfer[i], color='r', alpha=0.3)
#     circle2 = plt.Circle((network.positions[i, 0], network.positions[i, 1]), network.coverage_radius[i], color='g', alpha=0.3)
#     plt.gca().add_patch(circle)
#     plt.gca().add_patch(circle2)

# plt.scatter(network.main_unit_pos[0], network.main_unit_pos[ 1], c='green', label='Main Unit')

# circle = plt.Circle((network.main_unit_pos[0], network.main_unit_pos[1]), network.coverage_radius_main_unit_transfer, color='r', alpha=0.3)
# plt.gca().add_patch(circle)

# plt.xlim(0, NETWORK_AREA[0])
# plt.ylim(0, NETWORK_AREA[1])
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("Położenie sensorów i ich obszary zasięgu")
# plt.legend()
# plt.savefig("/home/jan/Informatyka/Projekt_indywidualny/temp.png")
