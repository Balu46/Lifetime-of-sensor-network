import matplotlib.pyplot as plt
import json
from symulacja_sieci_sensorowej import BATTERY_CAPACITY,NUM_OF_SENSORS_IN_NETWORK, symulation, symulation_witout_optymalizacion
import os
import numpy as np

# Przykład 1: Nagrody w czasie
def plot_rewards(history, with_otymalization=True):
    rewards = [step["reward"] for step in history]
    plt.plot(rewards)
    plt.xlabel("Krok symulacji")
    plt.ylabel("Nagroda")
    plt.title("Nagrody w czasie")
    if with_otymalization:
        plt.savefig('Projekt_indywidualny/logs/rewards_over_time_with_optymalization.png')
    else:
        plt.savefig('Projekt_indywidualny/logs/rewards_over_time_without_optymalization.png')
    

# Przykład 2: Energia sensorów w czasie
def plot_energy(history, with_otymalization=True):
    for sensor_id in range(1, NUM_OF_SENSORS_IN_NETWORK + 1):
        energies = [step["sensors"][sensor_id - 1]["energy"] for step in history]
        plt.plot(energies, label=f"Sensor {sensor_id}")
    plt.xlabel("Krok symulacji")
    plt.ylabel("Energia")
    plt.title("Energia sensorów w czasie")
    plt.legend()
    if with_otymalization:
        plt.savefig('Projekt_indywidualny/logs/energy_over_time_with_optymalization.png')
    else:
        plt.savefig('Projekt_indywidualny/logs/energy_over_time_without_optymalization.png')


    
def plot_comparison(history_opt, history_no_opt):
    data_opt = [step["data_collected"] for step in history_opt]
    data_no_opt = [step["data_collected"] for step in history_no_opt]
    plt.plot(data_opt, label="Z optymalizacją")
    plt.plot(data_no_opt, label="Bez optymalizacji")
    plt.xlabel("Krok symulacji")
    plt.ylabel("Zebrane dane")
    plt.legend()
    plt.savefig('Projekt_indywidualny/logs/comparison_over_time.png')
    
def compare_average_energy(history_optimized, history_baseline):
    """
    Compares the average sensor energy between DQN-optimized simulation and baseline simulation (without optimization).

    Args:
        history_optimized (list): Simulation history with DQN optimization (output of symulation())
        history_baseline (list): Baseline simulation history without optimization (output of symulation_witout_optymalizacion())
    """
    # Oblicz średnią energię dla każdego kroku w obu symulacjach
    avg_energy_optimized = []
    for step in history_optimized:
        energies = [sensor["energy"] for sensor in step["sensors"]]
        avg_energy_optimized.append(np.mean(energies))
        
    avg_energy_baseline = []
    for step in history_baseline:
        energies = [sensor["energy"] for sensor in step["sensors"]]
        avg_energy_baseline.append(np.mean(energies))
    
    # Przygotuj wykres
    plt.figure(figsize=(12, 6))
    
    # Długość najkrótszej historii (aby wykresy były porównywalne)
    min_length = min(len(avg_energy_optimized), len(avg_energy_baseline))
    
    plt.plot(avg_energy_optimized[:min_length], 
             label=f"Z optymalizacją (DQN)\nKońcowa średnia: {avg_energy_optimized[-1]:.2f}",
             color='green', linewidth=2)
    plt.plot(avg_energy_baseline[:min_length], 
             label=f"Bez optymalizacji\nKońcowa średnia: {avg_energy_baseline[-1]:.2f}",
             color='red', linewidth=2, linestyle='--')
    
    plt.title("Porównanie średniej energii sensorów między metodami")
    plt.xlabel("Krok symulacji")
    plt.ylabel("Średnia energia sensorów")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    diff = avg_energy_optimized[-1] - avg_energy_baseline[-1]
    plt.annotate(f"Różnica: {diff:.2f} ({diff/BATTERY_CAPACITY*100:.1f}% pojemności baterii)",
                 xy=(0.5, 0.1), xycoords='axes fraction',
                 ha='center', bbox=dict(boxstyle="round", fc="white", ec="gray"))
    
    plt.tight_layout()
    plt.savefig('Projekt_indywidualny/logs/comparison_of_average_energy_over_time.png')



filename_without_DNQ = 'Projekt_indywidualny/logs/simulation_data_without_DNQ.json'
if not os.path.exists(filename_without_DNQ):
    history_without_optimization  = symulation_witout_optymalizacion()
    
    with open(filename_without_DNQ, "w") as f:
        json.dump({"simulation_data": history_without_optimization}, f)
else:
    with open(filename_without_DNQ, 'r') as f:
        history_without_optimization = json.load(f)
    history_without_optimization = history_without_optimization["simulation_data"]



    
filename = 'Projekt_indywidualny/logs/simulation_data.json'
if not os.path.exists(filename):
    data = symulation()
else:
    with open(filename, 'r') as f:
        data = json.load(f)
    data = data["simulation_data"]




# You need to call one function at a time to generate the plots or else they will overwrite each other.


# plot_rewards(data)
# plot_energy(data)
# plot_comparison(data, history_without_optimization)  
# compare_average_energy(data, history_without_optimization)
# plot_energy(history_without_optimization, with_otymalization=False)
# plot_rewards(history_without_optimization, with_otymalization=False) # you can also call this function to plot rewards without optimization but is is not necessary beneficial to do so