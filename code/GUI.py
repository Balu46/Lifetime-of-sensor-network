import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
from symulacja_sieci_sensorowej import SensorNetwork, Sensor, symulation
from typing import List
import numpy as np
import math
import time
import tkinter as tk
from tkinter import ttk
import numpy as np
import time
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from Hyperparamiters import *
import json

import json
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class SensorNetworkVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Sensor Network Simulation Visualizer")
        
        # Simulation data
        self.simulation_data = []
        self.current_step = 0
        self.is_playing = False
        self.playback_speed = 500  # ms
        
        # Colors for visualization
        self.COLORS = {
            'main_unit': 'red',
            'active': 'blue',
            'sleeping': 'purple',
            'inactive': 'gray'
        }
        
        # Setup UI
        self.setup_ui()
        
        # Load simulation data automatically
        self.load_simulation_data()
        
    def setup_ui(self):
        # Main frames
        self.control_frame = ttk.Frame(self.master, padding=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.visualization_frame = ttk.Frame(self.master, padding=10)
        self.visualization_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        # Visualization canvas
        self.canvas = tk.Canvas(self.visualization_frame, bg='white', 
                               width=NETWORK_AREA[0]*5, height=NETWORK_AREA[1]*5)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        # Control buttons
        self.setup_controls()
        
        # Stats display
        self.setup_stats()
        
        # Energy plot
        self.setup_energy_plot()
        
        # Legend
        self.setup_legend()
    
    def setup_controls(self):
        control_group = ttk.LabelFrame(self.control_frame, text="Playback Controls", padding=10)
        control_group.pack(fill=tk.X, pady=5)
        
        # Button grid
        button_frame = ttk.Frame(control_group)
        button_frame.pack()
        
        self.play_button = ttk.Button(button_frame, text="▶", width=3, 
                                    command=self.start_playback)
        self.play_button.grid(row=0, column=1, padx=2)
        
        self.pause_button = ttk.Button(button_frame, text="⏸", width=3, 
                                     command=self.pause_playback, state=tk.DISABLED)
        self.pause_button.grid(row=0, column=2, padx=2)
        
        self.step_back_button = ttk.Button(button_frame, text="⏮", width=3,
                                         command=self.step_backward)
        self.step_back_button.grid(row=0, column=0, padx=2)
        
        self.step_forward_button = ttk.Button(button_frame, text="⏭", width=3,
                                            command=self.step_forward)
        self.step_forward_button.grid(row=0, column=3, padx=2)
        
        # Speed control
        speed_frame = ttk.Frame(control_group)
        speed_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        self.speed_slider = ttk.Scale(speed_frame, from_=100, to=2000, 
                                     command=self.update_speed, orient=tk.HORIZONTAL)
        self.speed_slider.set(self.playback_speed)
        self.speed_slider.pack(fill=tk.X)
        
        # Load button
        ttk.Button(self.control_frame, text="Reload Simulation", 
                  command=self.load_simulation_data).pack(fill=tk.X, pady=5)
    
    def setup_stats(self):
        stats_group = ttk.LabelFrame(self.control_frame, text="Statistics", padding=10)
        stats_group.pack(fill=tk.X, pady=5)
        
        self.step_label = ttk.Label(stats_group, text="Step: 0/0")
        self.step_label.pack(anchor=tk.W)
        
        self.data_label = ttk.Label(stats_group, text="Data collected: 0")
        self.data_label.pack(anchor=tk.W)
        
        self.energy_label = ttk.Label(stats_group, text="Avg energy: 0%")
        self.energy_label.pack(anchor=tk.W)
        
        self.status_label = ttk.Label(stats_group, text="Status: Ready")
        self.status_label.pack(anchor=tk.W)
    
    def setup_energy_plot(self):
        self.energy_fig, self.energy_ax = plt.subplots(figsize=(4, 2), dpi=80)
        self.energy_ax.set_title("Sensor Energy Levels")
        self.energy_ax.set_ylim(0, BATTERY_CAPACITY)
        
        self.energy_canvas = FigureCanvasTkAgg(self.energy_fig, master=self.control_frame)
        self.energy_canvas.draw()
        self.energy_canvas.get_tk_widget().pack(fill=tk.X, pady=10)
    
    def setup_legend(self):
        legend_frame = ttk.Frame(self.visualization_frame)
        legend_frame.pack(fill=tk.X)
        
        ttk.Label(legend_frame, text="Legend:").pack(side=tk.LEFT)
        
        for state, color in self.COLORS.items():
            tk.Canvas(legend_frame, width=20, height=20, bg=color).pack(side=tk.LEFT, padx=2)
            ttk.Label(legend_frame, text=state).pack(side=tk.LEFT)
    
    def load_simulation_data(self, filename="simulation_data.json"):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.simulation_data = data["simulation_data"]
            self.current_step = 0
            self.update_visualization()
            self.update_stats()
            
            self.status_label.config(text=f"Status: Loaded {len(self.simulation_data)} steps")
            messagebox.showinfo("Success", f"Successfully loaded simulation with {len(self.simulation_data)} steps")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load simulation: {str(e)}")
            self.status_label.config(text="Status: Load failed")
    
    def update_visualization(self):
        if not self.simulation_data or self.current_step >= len(self.simulation_data):
            return
            
        self.canvas.delete("all")
        step_data = self.simulation_data[self.current_step]
        
        # Draw main unit
        main = step_data["main_unit"]
        x, y = int(main["position"][0][0]*5), int(main["position"][0][1]*5)
        self.canvas.create_oval(x-10, y-10, x+10, y+10, 
                              fill=self.COLORS['main_unit'], outline='black')
        self.canvas.create_text(x, y-15, text="Main Unit", font=('Arial', 8))
        
        # Draw coverage area
        cov_radius = int(main["transfer_distance"]*5)
        self.canvas.create_oval(x-cov_radius, y-cov_radius, 
                               x+cov_radius, y+cov_radius, 
                               outline='red', dash=(2,2))
        
        # Draw sensors
        for sensor in step_data["sensors"]:
            x, y = int(sensor["position"][0][0]*5), int(sensor["position"][0][1]*5)
            
            # Determine color based on state
            if sensor["is_sleeping"]:
                color = self.COLORS['sleeping']
            elif sensor["energy"] > TRANSMISSION_COST:
                color = self.COLORS['active']
            else:
                color = self.COLORS['inactive']
            
            # Draw sensor
            self.canvas.create_oval(x-8, y-8, x+8, y+8, fill=color, outline='black')
            self.canvas.create_text(x, y-12, text=f"S{sensor['id']}", font=('Arial', 7))
            
            # Draw coverage area if active
            if color == self.COLORS['active']:
                cov_radius = int(sensor["coverage_radius"]*5)
                self.canvas.create_oval(x-cov_radius, y-cov_radius, 
                                       x+cov_radius, y+cov_radius, 
                                       outline='blue')
            
            # Draw connections
            for neighbor in step_data["sensors"]:
                if neighbor["id"] > sensor["id"]:
                    x2, y2 = int(neighbor["position"][0][0]*5), int(neighbor["position"][0][1]*5)
                    dash = (3,2) if sensor["is_sleeping"] or neighbor["is_sleeping"] else (1,1)
                    self.canvas.create_line(x, y, x2, y2, fill='light gray', dash=dash)
        
        # Update energy plot
        self.update_energy_plot(step_data)
    
    def update_energy_plot(self, step_data):
        self.energy_ax.clear()
        
        energies = [s["energy"] for s in step_data["sensors"]]
        sensor_ids = [f"S{s['id']}" for s in step_data["sensors"]]
        
        colors = []
        for sensor in step_data["sensors"]:
            if sensor["is_sleeping"]:
                colors.append(self.COLORS['sleeping'])
            elif sensor["energy"] > TRANSMISSION_COST:
                colors.append(self.COLORS['active'])
            else:
                colors.append(self.COLORS['inactive'])
        
        self.energy_ax.bar(sensor_ids, energies, color=colors)
        self.energy_ax.set_ylim(0, BATTERY_CAPACITY)
        self.energy_ax.set_title(f"Energy Levels - Step {self.current_step}")
        self.energy_canvas.draw()
    
    def update_stats(self):
        if not self.simulation_data:
            return
            
        step_data = self.simulation_data[self.current_step]
        
        # Update labels
        self.step_label.config(text=f"Step: {self.current_step+1}/{len(self.simulation_data)}")
        self.data_label.config(text=f"Data collected: {step_data['data_collected']}")
        
        # Calculate average energy
        avg_energy = np.mean([s["energy"] for s in step_data["sensors"]])
        self.energy_label.config(text=f"Avg energy: {avg_energy/BATTERY_CAPACITY:.1%}")
    
    def start_playback(self):
        if not self.simulation_data:
            return
            
        self.is_playing = True
        self.play_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.playback()
    
    def pause_playback(self):
        self.is_playing = False
        self.play_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
    
    def playback(self):
        if self.is_playing and self.current_step < len(self.simulation_data)-1:
            self.current_step += 1
            self.update_visualization()
            self.update_stats()
            self.master.after(self.playback_speed, self.playback)
        else:
            self.pause_playback()
    
    def step_forward(self):
        if self.current_step < len(self.simulation_data)-1:
            self.current_step += 1
            self.update_visualization()
            self.update_stats()
    
    def step_backward(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.update_visualization()
            self.update_stats()
    
    def update_speed(self, value):
        self.playback_speed = int(float(value))

symulation_data = symulation()
with open("simulation_data.json", "w") as f:
    json.dump({"simulation_data": symulation_data}, f)
# Initialize the main application window

root = tk.Tk()
app = SensorNetworkVisualizer(root)
root.mainloop()