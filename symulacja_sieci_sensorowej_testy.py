import numpy as np
import random
import matplotlib.pyplot as plt

# Parameters of the sensor network
NUM_NODES = 10  # Number of sensor nodes
BATTERY_CAPACITY = 100  # Maximum energy capacity (in units)
TRANSMISSION_COST = 5  # Energy cost per data transmission
TIME_STEPS = 50  # Number of simulation steps
NETWORK_AREA = (100, 100)  # Network area size (units)

# Sensor network simulation class
class SensorNetwork:
    def __init__(self, num_nodes, area_size):
        self.num_nodes = num_nodes
        self.area_size = area_size
        self.reset()
    
    def reset(self):
        """Initializes sensor nodes with full battery, random positions, and coverage radius."""
        self.energy_levels = np.full(self.num_nodes, BATTERY_CAPACITY)
        self.data_collected = np.zeros(self.num_nodes)
        self.positions = np.random.rand(self.num_nodes, 2) * self.area_size  # Random sensor positions
        self.coverage_radius = np.random.uniform(10, 20, self.num_nodes)  # Sensor coverage radiusreset
        return self.energy_levels.copy()
    
    def step(self):
        """Each sensor randomly decides whether to transmit data, consuming energy."""
        for i in range(self.num_nodes):
            if self.energy_levels[i] > TRANSMISSION_COST and random.random() > 0.3:  # 70% chance to transmit
                self.energy_levels[i] -= TRANSMISSION_COST
                self.data_collected[i] += self.coverage_radius[i] / 10  # More data collected with a larger coverage radius
    
    def is_active(self):
        """Checks if at least one sensor still has enough energy to operate."""
        return np.any(self.energy_levels > TRANSMISSION_COST)

# Initialize the network
network = SensorNetwork(NUM_NODES, NETWORK_AREA)
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
plt.savefig("temp2.png")

# Visualization of sensor positions and their coverage areas
plt.figure(figsize=(6, 6))
plt.scatter(network.positions[:, 0], network.positions[:, 1], c='blue', label='Sensory')
for i in range(NUM_NODES):
    circle = plt.Circle((network.positions[i, 0], network.positions[i, 1]), network.coverage_radius[i], color='r', alpha=0.3)
    plt.gca().add_patch(circle)
plt.xlim(0, NETWORK_AREA[0])
plt.ylim(0, NETWORK_AREA[1])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Położenie sensorów i ich obszary zasięgu")
plt.legend()
plt.savefig("temp.png")
