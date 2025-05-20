import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List

# Parametry sieci
NUM_OF_MAIN_UNITS = 1
NUM_OF_SENSORS_IN_NETWORK = 10
BATTERY_CAPACITY = 100
TRANSMISSION_COST = 5
MINING_COST = 1
DURATION = 100  # Liczba kroków symulacji
NETWORK_AREA = (100, 100)  # Rozmiar obszaru sieci

class Data:
    def __init__(self):
        self.data = -1
        self.communique = ''
        self.device = -1  # ID urządzenia źródłowego
        self.destination = None  # ID docelowego urządzenia
        
    def create_mined_data(self, device, destination):
        self.data = 1
        self.communique = 'mined data'
        self.device = device
        self.destination = destination

class Sensor:
    def __init__(self, id: int, area_size=NETWORK_AREA, battery_capacity=BATTERY_CAPACITY):
        self.id = id
        self.energy = battery_capacity
        self.area_size = area_size
        self.position = np.random.rand(2) * area_size  # Losowa pozycja [x, y]
        self.coverage_radius = np.random.uniform(10, 20)
        self.routing_table = []  # Lista dostępnych sąsiadów
        self.is_active = True
        self.sleep_time = 0
        self.data_to_transfer = []
        self.recived_data = []
        
    def reset(self):
        self.energy = BATTERY_CAPACITY
        self.is_active = True
        self.sleep_time = 0
        self.data_to_transfer = []
        self.recived_data = []
        
    def recive_data(self, data: Data) -> bool:
        if data.destination == self.id:
            self.recived_data.append(data)
            return True
        else:
            return self.send_data(data)
        
    def colect_data(self) -> bool:
        mined_data = Data()
        mined_data.create_mined_data(self.id, 0)  # Główna jednostka ma ID=0
        return self.send_data(mined_data)
        
    def send_data(self, data: Data) -> bool:
        if not self.routing_table or self.energy < TRANSMISSION_COST:
            return False
        
        # GPSR: Wybierz sąsiada najbliżej głównej jednostki
        main_unit_pos = self.routing_table[0].position  # Zakładamy, że główna jednostka jest pierwsza
        best_neighbor = min(
            self.routing_table,
            key=lambda neighbor: np.linalg.norm(neighbor.position - main_unit_pos)
        )
        
        if best_neighbor.is_active:
            # self.energy -= TRANSMISSION_COST
            return best_neighbor.recive_data(data)
        return False

class MainUnit(Sensor):
    def __init__(self, id: int, area_size=NETWORK_AREA, battery_capacity=BATTERY_CAPACITY):
        super().__init__(id, area_size, battery_capacity)
        self.colect_data_by_network = []
        
    def recive_data(self, data: Data) -> bool:
        if data.destination == self.id:
            self.colect_data_by_network.append(data)
            return True
        return False

class SensorNetwork:
    def __init__(self, area_size, num_main_units=1, num_sensors=10):
        self.sensors = []
        self.sensors.append(MainUnit(0))  # Główna jednostka (ID=0)
        
        # Inicjalizacja sensorów
        for i in range(1, num_sensors + 1):
            self.sensors.append(Sensor(i))
        
        # Budowa tablic routingu (każdy sensor zna swoich sąsiadów)
        for sensor in self.sensors:
            for other in self.sensors:
                if sensor != other:
                    distance = np.linalg.norm(sensor.position - other.position)
                    if distance <= sensor.coverage_radius:
                        sensor.routing_table.append(other)
    
    def step(self):
        """Symulacja kroku transmisji danych."""
        for sensor in self.sensors[1:]:  # Pomijamy główną jednostkę
            if sensor.is_active:
                sensor.colect_data()
    
    def simulate(self, time: int):
        """Uruchomienie symulacji."""
        for _ in range(time):
            self.step()

# Inicjalizacja i symulacja
network = SensorNetwork(NETWORK_AREA)
network.simulate(DURATION)

# Wyniki
print(f"Dane zebrane przez główną jednostkę: {len(network.sensors[0].colect_data_by_network)}")
print(f"Energia pozostała w sensorach: {[s.energy for s in network.sensors[1:]]}")

# Wizualizacja
plt.figure(figsize=(10, 8))
plt.scatter(
    [s.position[0] for s in network.sensors],
    [s.position[1] for s in network.sensors],
    c=['red' if s.id == 0 else 'blue' for s in network.sensors],
    label='Główna jednostka (czerwony)'
)
plt.title("Pozycje sensorów w sieci")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
plt.savefig("/home/jan/Informatyka/Projekt_indywidualny/temp4.png")


