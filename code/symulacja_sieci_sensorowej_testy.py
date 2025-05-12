import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
NUM_NODES = 10
BATTERY_CAPACITY = 100
TRANSMISSION_COST = 5
TIME_STEPS = 100
EPISODES = 50
NETWORK_AREA = (100, 100)
STATE_SIZE = 2  # [battery_level, coverage_radius]
ACTION_SIZE = 2  # [sleep, transmit]
GAMMA = 0.95
ALPHA = 0.001
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MEMORY_SIZE = 1000
BATCH_SIZE = 32

# Neural Network for Q-value approximation
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.out = nn.Linear(24, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# Sensor agent with its own DQN
class SensorAgent:
    def __init__(self):
        self.model = DQN(STATE_SIZE, ACTION_SIZE).to(device)
        self.target_model = DQN(STATE_SIZE, ACTION_SIZE).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=ALPHA)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1])  # Explore
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()  # Exploit

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state in batch:
            state_tensor = torch.FloatTensor(state).to(device)
            next_state_tensor = torch.FloatTensor(next_state).to(device)
            target = reward + GAMMA * torch.max(self.target_model(next_state_tensor)).item()
            current_q = self.model(state_tensor)[action]
            loss = (current_q - target) ** 2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# Initialize agents and environment
agents = [SensorAgent() for _ in range(NUM_NODES)]
positions = np.random.rand(NUM_NODES, 2) * NETWORK_AREA
coverage_radius = np.random.uniform(10, 20, NUM_NODES)

# Simulation
for episode in range(EPISODES):
    energy_levels = np.full(NUM_NODES, BATTERY_CAPACITY)
    data_collected = np.zeros(NUM_NODES)

    for t in range(TIME_STEPS):
        if np.all(energy_levels <=   TRANSMISSION_COST):
            break

        for i in range(NUM_NODES):
            if energy_levels[i] <= TRANSMISSION_COST:
                continue  # Sensor sleeps due to low battery

            state = [energy_levels[i] / BATTERY_CAPACITY, coverage_radius[i] / 20.0]
            action = agents[i].act(state)

            if action == 1 and energy_levels[i] > TRANSMISSION_COST:
                energy_levels[i] -= TRANSMISSION_COST
                reward = coverage_radius[i] / 10
                data_collected[i] += reward
            else:
                reward = 0

            next_state = [energy_levels[i] / BATTERY_CAPACITY, coverage_radius[i] / 20.0]
            agents[i].remember(state, action, reward, next_state)
            agents[i].replay()

    for agent in agents:
        agent.update_target_model()
        if agent.epsilon > EPSILON_END:
            agent.epsilon *= EPSILON_DECAY

# Visualize sensor positions and coverage
plt.figure(figsize=(6, 6))
plt.scatter(positions[:, 0], positions[:, 1], c='blue', label='Sensors')
for i in range(NUM_NODES):
    circle = plt.Circle((positions[i, 0], positions[i, 1]), coverage_radius[i], color='r', alpha=0.3)
    plt.gca().add_patch(circle)
plt.xlim(0, NETWORK_AREA[0])
plt.ylim(0, NETWORK_AREA[1])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Sensor Positions and Coverage (After Training)")
plt.legend()
plt.grid(True)
plt.savefig("/home/jan/Informatyka/Projekt_indywidualny/temp4.png")
