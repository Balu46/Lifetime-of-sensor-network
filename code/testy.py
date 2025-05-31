import tkinter as tk
from tkinter import ttk
import numpy as np
import time
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class SensorNetworkGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Sensor Network Simulation")
        
        # Kolory dla różnych stanów sensorów
        self.COLORS = {
            'active': 'blue',
            'inactive': 'gray',
            'sleeping': 'purple',
            'main_unit': 'red'
        }
        
        # Inicjalizacja zmiennych symulacji
        self.is_running = False
        self.current_step = 0
        self.simulation_speed = 500
        
        self.setup_ui()
        
        # Inicjalizacja sieci i wykonanie pierwszego kroku
        self.network = SensorNetwork(NETWORK_AREA)
        self.perform_initial_step()  # Nowa metoda wykonująca pierwszy krok
        
    def setup_ui(self):
        # Konfiguracja interfejsu użytkownika
        self.control_frame = ttk.Frame(self.master, padding="10")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.visualization_frame = ttk.Frame(self.master, padding="10")
        self.visualization_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        self.setup_control_panel()
        self.setup_visualization()
        self.setup_energy_plot()

    def perform_initial_step(self):
        """Wykonuje pierwszy krok symulacji i aktualizuje wizualizację"""
        observation, reward, done, _ = self.network.step()
        self.current_step += 1
        self.update_visualization()
        self.score_label.config(text=f"Current score: {reward}")

    def setup_control_panel(self):
        # Przyciski kontrolne
        self.start_button = ttk.Button(self.control_frame, text="Start", command=self.start_simulation)
        self.start_button.pack(fill=tk.X, pady=5)
        
        self.pause_button = ttk.Button(self.control_frame, text="Pause", command=self.pause_simulation, state=tk.DISABLED)
        self.pause_button.pack(fill=tk.X, pady=5)
        
        self.step_button = ttk.Button(self.control_frame, text="Step", command=self.step_simulation)
        self.step_button.pack(fill=tk.X, pady=5)
        
        self.reset_button = ttk.Button(self.control_frame, text="Reset", command=self.reset_simulation)
        self.reset_button.pack(fill=tk.X, pady=5)
        
        # Przycisk pełnej symulacji
        self.full_sim_button = ttk.Button(self.control_frame, text="Run Full Simulation", 
                                        command=self.run_full_simulation)
        self.full_sim_button.pack(fill=tk.X, pady=5)
        
        # Kontrola prędkości
        self.speed_label = ttk.Label(self.control_frame, text="Simulation Speed:")
        self.speed_label.pack(fill=tk.X, pady=(10,0))
        
        self.speed_slider = ttk.Scale(self.control_frame, from_=50, to=1000, 
                                    command=self.update_speed, orient=tk.HORIZONTAL)
        self.speed_slider.set(self.simulation_speed)
        self.speed_slider.pack(fill=tk.X)
        
        # Panel statystyk
        self.stats_frame = ttk.LabelFrame(self.control_frame, text="Statistics", padding="10")
        self.stats_frame.pack(fill=tk.X, pady=10)
        
        self.time_label = ttk.Label(self.stats_frame, text=f"Time step: {self.current_step}/{DURATION}")
        self.time_label.pack(anchor=tk.W)
        
        self.data_label = ttk.Label(self.stats_frame, text="Data collected: 0")
        self.data_label.pack(anchor=tk.W)
        
        self.active_label = ttk.Label(self.stats_frame, text="Active sensors: 10")
        self.active_label.pack(anchor=tk.W)
        
        self.score_label = ttk.Label(self.stats_frame, text="Current score: 0")
        self.score_label.pack(anchor=tk.W)
        
    def setup_visualization(self):
        # Canvas do wizualizacji sieci
        self.canvas = tk.Canvas(self.visualization_frame, bg='white', 
                               width=NETWORK_AREA[0]*5, height=NETWORK_AREA[1]*5)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        # Legenda
        legend_frame = ttk.Frame(self.visualization_frame)
        legend_frame.pack(fill=tk.X)
        
        ttk.Label(legend_frame, text="Legend:").pack(side=tk.LEFT)
        
        # Dodanie wszystkich stanów do legendy
        colors = ['main_unit', 'active', 'sleeping', 'inactive']
        for color_type in colors:
            tk.Canvas(legend_frame, width=20, height=20, 
                      bg=self.COLORS[color_type]).pack(side=tk.LEFT, padx=5)
            ttk.Label(legend_frame, text=color_type.capitalize()).pack(side=tk.LEFT)

    def setup_energy_plot(self):
        # Wykres energii
        self.energy_fig, self.energy_ax = plt.subplots(figsize=(4, 2), dpi=80)
        self.energy_ax.set_title("Sensor Energy Levels")
        self.energy_ax.set_ylim(0, BATTERY_CAPACITY)
        self.energy_ax.set_xlim(0, NUM_OF_SENSORS_IN_NETWORK+1)
        
        self.energy_canvas = FigureCanvasTkAgg(self.energy_fig, master=self.energy_plot_frame)
        self.energy_canvas.draw()
        self.energy_canvas.get_tk_widget().pack(fill=tk.X)

    def update_visualization(self):
        self.canvas.delete("all")
        
        # Rysuj główną jednostkę
        self.draw_main_unit()
        
        # Rysuj wszystkie sensory
        for sensor in self.network.sensors[1:]:
            self.draw_sensor(sensor)
        
        # Aktualizuj statystyki
        self.update_stats()

    def draw_main_unit(self):
        main = self.network.sensors[0]
        x = int(main.position[0][0]*5)
        y = int(main.position[0][1]*5)
        
        self.canvas.create_oval(x-10, y-10, x+10, y+10, 
                              fill=self.COLORS['main_unit'], outline='black')
        self.canvas.create_text(x, y-15, text="Main Unit", font=('Arial', 8))
        
        cov_radius = int(float(main.transfer_coverage_distance)*5)
        self.canvas.create_oval(x-cov_radius, y-cov_radius, 
                               x+cov_radius, y+cov_radius, 
                               outline='red', dash=(2,2))

    def draw_sensor(self, sensor):
        x = int(sensor.position[0][0]*5)
        y = int(sensor.position[0][1]*5)
        
        # Wybierz kolor w zależności od stanu
        if sensor.is_sleeping:
            color = self.COLORS['sleeping']
        elif sensor.is_active():
            color = self.COLORS['active']
        else:
            color = self.COLORS['inactive']
        
        # Rysuj sensor
        self.canvas.create_oval(x-8, y-8, x+8, y+8, 
                              fill=color, outline='black')
        self.canvas.create_text(x, y-12, text=f"S{sensor.id}", font=('Arial', 7))
        
        # Rysuj obszar pokrycia tylko dla aktywnych sensorów
        if sensor.is_active() and not sensor.is_sleeping:
            cov_radius = int(float(sensor.coverage_radius)*5)
            self.canvas.create_oval(x-cov_radius, y-cov_radius, 
                                   x+cov_radius, y+cov_radius, 
                                   outline='blue')
        
        # Rysuj połączenia
        self.draw_connections(sensor, x, y)

    def draw_connections(self, sensor, x, y):
        for neighbor in sensor.routing_table:
            if neighbor.id > sensor.id:
                x2 = int(neighbor.position[0][0]*5)
                y2 = int(neighbor.position[0][1]*5)
                
                # Linia przerywana dla uśpionych sensorów
                dash_pattern = (3,2) if sensor.is_sleeping or neighbor.is_sleeping else (1,1)
                self.canvas.create_line(x, y, x2, y2, 
                                      fill='light gray', dash=dash_pattern)

    def update_stats(self):
        self.time_label.config(text=f"Time step: {self.current_step}/{DURATION}")
        data_collected = len(self.network.sensors[0].colect_data_by_network)
        self.data_label.config(text=f"Data collected: {data_collected}")
        
        active_count = 0
        sleeping_count = 0
        for sensor in self.network.sensors[1:]:
            if sensor.is_sleeping:
                sleeping_count += 1
            elif sensor.is_active():
                active_count += 1
                
        self.active_label.config(text=f"Active: {active_count}, Sleeping: {sleeping_count}, Inactive: {NUM_OF_SENSORS_IN_NETWORK-active_count-sleeping_count}")
        
        self.update_energy_plot()

    def update_energy_plot(self):
        self.energy_ax.clear()
        energies = [sensor.energy for sensor in self.network.sensors[1:]]
        sensor_ids = [f"S{i}" for i in range(1, len(energies)+1)]
        
        colors = []
        for sensor in self.network.sensors[1:]:
            if sensor.is_sleeping:
                colors.append('purple')
            elif sensor.energy > TRANSMISSION_COST:
                colors.append('blue')
            else:
                colors.append('red')
                
        self.energy_ax.bar(sensor_ids, energies, color=colors)
        self.energy_ax.set_ylim(0, BATTERY_CAPACITY)
        self.energy_ax.set_title("Sensor Energy Levels")
        self.energy_ax.set_ylabel("Energy")
        self.energy_canvas.draw()

    # Pozostałe metody bez zmian
    def start_simulation(self): 
        if not self.is_running:
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            self.step_button.config(state=tk.DISABLED)
            self.run_simulation()
            
    def pause_simulation(self):
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.step_button.config(state=tk.NORMAL)
        
    def reset_simulation(self):
        self.pause_simulation()
        self.current_step = 0
        self.network = SensorNetwork(NETWORK_AREA)
        self.perform_initial_step()  # Reset również wykonuje pierwszy krok
        
    def step_simulation(self):
        if not self.is_running:
            observation, reward, done, _ = self.network.step()
            self.current_step += 1
            self.score_label.config(text=f"Current score: {reward}")
            self.update_visualization()
            
            if done:
                self.pause_simulation()
                print("Simulation completed!")
                
    def run_simulation(self):
        if self.is_running and self.current_step < DURATION:
            self.step_simulation()
            self.master.after(self.simulation_speed, self.run_simulation)
        else:
            self.pause_simulation()
            
    def update_speed(self, value):
        self.simulation_speed = int(float(value))
        
    def run_full_simulation(self):
        self.pause_simulation()
        
        def run_in_thread():
            self.start_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.DISABLED)
            self.step_button.config(state=tk.DISABLED)
            self.reset_button.config(state=tk.DISABLED)
            
            # Uruchom pełną symulację
            main_units = symulation()
            
            # Aktualizuj stan końcowy
            self.network = SensorNetwork(NETWORK_AREA)
            if main_units:
                self.network.sensors[0] = main_units[-1]
            self.current_step = DURATION
            self.update_visualization()
            
            self.start_button.config(state=tk.NORMAL)
            self.reset_button.config(state=tk.NORMAL)
            self.step_button.config(state=tk.NORMAL)
            
        threading.Thread(target=run_in_thread, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = SensorNetworkGUI(root)
    root.mainloop()