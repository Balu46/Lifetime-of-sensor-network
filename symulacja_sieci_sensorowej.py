import numpy as np
import random
import matplotlib.pyplot as plt


# Parameters of the sensor network
NUM_NODES = 10  # Number of sensor nodes
BATTERY_CAPACITY = 100  # Maximum energy capacity (in units)
TRANSMISSION_COST = 5  # Energy cost per data transmission
TIME_STEPS = 50  # Number of simulation steps
NETWORK_AREA = (100, 100)  # Network area size (units)
NUM_OF_DATA_IN_BACH = 10
NUM_OF_MAIN_UNITS = 1


        

# Sensor network simulation class
class SensorNetwork:
    def __init__(self, num_nodes, area_size, number_of_main_units):
        self.num_nodes = num_nodes
        self.area_size = area_size
        self.main_unit_pos = (np.random.rand(number_of_main_units ,2)* self.area_size).squeeze()
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

    def routing_table(self) -> np.array:
        routing_table = []
        for i in range(self.num_nodes):
            help_table = []
            for ii in range(self.num_nodes):
                    if (self.positions[i][0] + self.coverage_radius_transfer[i] >= self.positions[ii][0] and
                        self.positions[i][1] + self.coverage_radius_transfer[i] >= self.positions[ii][1]):
                            help_table.append(ii)
                    else:
                        help_table.append(-1)
            routing_table.append(help_table)
        print(routing_table)    
        return np.array(routing_table)
                        
            
# Initialize the network
network = SensorNetwork(NUM_NODES, NETWORK_AREA, NUM_OF_MAIN_UNITS)
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
