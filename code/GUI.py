import tkinter as tk
from tkinter import ttk, messagebox
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

class SensorNetworkVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Sensor Network Simulation Visualizer")
        
        # Simulation data
        self.simulation_data = []
        self.current_step = 0
        self.is_playing = False
        self.playback_speed = 500  # ms
        self.show_packet_routes = True  # Toggle for packet visualization
        self.packet_animation_step = 0
        
        # Colors for visualization
        self.COLORS = {
            'main_unit': 'red',
            'active': 'blue',
            'sleeping': 'purple',
            'inactive': 'gray',
            'packet': 'green',
            'packet_route': 'lime'
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
        
        # Additional controls
        control_bottom = ttk.Frame(control_group)
        control_bottom.pack(fill=tk.X, pady=5)
        
        # Speed control
        ttk.Label(control_bottom, text="Speed:").pack(side=tk.LEFT)
        self.speed_slider = ttk.Scale(control_bottom, from_=100, to=2000, 
                                     command=self.update_speed, orient=tk.HORIZONTAL)
        self.speed_slider.set(self.playback_speed)
        self.speed_slider.pack(fill=tk.X, padx=5)
        
        # Packet visualization toggle
        self.packet_toggle = ttk.Checkbutton(control_bottom, text="Show Packets",
                                           command=self.toggle_packet_vis)
        self.packet_toggle.pack(pady=5)
        self.packet_toggle.state(['selected'])  # Enabled by default
        
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
        
        self.packet_label = ttk.Label(stats_group, text="Active packets: 0")
        self.packet_label.pack(anchor=tk.W)
        
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
    
    def load_simulation_data(self, filename="/home/jan/Informatyka/Projekt_indywidualny/logs/simulation_data.json"):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.simulation_data = data["simulation_data"]
            self.current_step = 0
            self.packet_animation_step = 0
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
        self.draw_main_unit(step_data["main_unit"])
        
        # Draw all sensors
        for sensor in step_data["sensors"]:
            self.draw_sensor(sensor)
        
        # Draw packet routes if enabled
        if self.show_packet_routes:
            self.draw_packet_routes(step_data["main_unit"])
        
        # Update energy plot
        self.update_energy_plot(step_data)
    
    def draw_main_unit(self, main_unit):
        x, y = int(main_unit["position"][0][0]*5), int(main_unit["position"][0][1]*5)
        
        # Draw main unit
        self.canvas.create_oval(x-10, y-10, x+10, y+10, 
                              fill=self.COLORS['main_unit'], outline='black')
        self.canvas.create_text(x, y-15, text="Main Unit", font=('Arial', 8))
        
        # Draw coverage area
        cov_radius = int(main_unit["transfer_distance"]*5)
        self.canvas.create_oval(x-cov_radius, y-cov_radius, 
                               x+cov_radius, y+cov_radius, 
                               outline='red', dash=(2,2))
    
    def draw_sensor(self, sensor):
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
    
    def draw_packet_routes(self, main_unit):
        if "data" not in main_unit or not main_unit["data"]:
            print(f"No packet data at step {self.current_step}")  # Debug
            return
        
        # print(f"Drawing {len(main_unit['data'])} packets at step {self.current_step}")  # Debug
        
        # Get all sensor positions including main unit (id=0)
        sensor_positions = {0: main_unit["position"][0]}
        for sensor in self.simulation_data[self.current_step]["sensors"]:
            sensor_positions[sensor["id"]] = sensor["position"][0]
        
        # print(f"Sensor positions: {sensor_positions.keys()}")  # Debug
        
        active_packets = 0
        for packet in main_unit["data"]:
            if not isinstance(packet, dict) or "route" not in packet:
                # print(f"Invalid packet format: {packet}")  # Debug
                continue
                
            route = packet["route"]
            if len(route) < 2:
                # print(f"Packet route too short: {route}")  # Debug
                continue
            
            # print(f"Drawing packet route: {route}")  # Debug
            active_packets += 1
            
            # Draw complete route
            for i in range(len(route)-1):
                from_id = route[i]
                to_id = route[i+1]
                
                if from_id in sensor_positions and to_id in sensor_positions:
                    x1, y1 = sensor_positions[from_id][0]*5, sensor_positions[from_id][1]*5
                    x2, y2 = sensor_positions[to_id][0]*5, sensor_positions[to_id][1]*5
                    self.canvas.create_line(x1, y1, x2, y2, 
                                        fill=self.COLORS['packet_route'], 
                                        width=2, dash=(3,1))
                else:
                    print(f"Missing nodes in route: {from_id}->{to_id}")  # Debug
            
            # Draw animated packet
            if self.is_playing or self.packet_animation_step > 0:
                valid_route = [node_id for node_id in route if node_id in sensor_positions]
                if len(valid_route) >= 2:
                    route_positions = [sensor_positions[node_id] for node_id in valid_route]
                    total_length = sum(np.linalg.norm(np.array(route_positions[i+1])-np.array(route_positions[i])) 
                                for i in range(len(route_positions)-1))
                    
                    if total_length > 0:
                        current_length = (self.packet_animation_step % 20) / 20 * total_length
                        segment_start = route_positions[0]
                        remaining_length = current_length
                        
                        for i in range(len(route_positions)-1):
                            segment_end = route_positions[i+1]
                            segment_vector = np.array(segment_end) - np.array(segment_start)
                            segment_len = np.linalg.norm(segment_vector)
                            
                            if remaining_length <= segment_len:
                                direction = segment_vector / segment_len
                                packet_pos = np.array(segment_start) + direction * remaining_length
                                x, y = packet_pos[0]*5, packet_pos[1]*5
                                self.canvas.create_oval(x-5, y-5, x+5, y+5, 
                                                    fill=self.COLORS['packet'], 
                                                    outline='black')
                                break
                            
                            remaining_length -= segment_len
                            segment_start = segment_end
        
        self.packet_label.config(text=f"Active packets: {active_packets}")
        
        if active_packets == 0:
            print("Warning: No valid packets drawn! Check route data.")  # Debug   
         
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
        if self.is_playing:
            if self.current_step < len(self.simulation_data)-1:
                self.current_step += 1
                self.update_visualization()
                self.update_stats()
                self.master.after(self.playback_speed, self.playback)
            else:
                self.current_step = 0  # Loop to beginning
                self.update_visualization()
                self.update_stats()
                self.master.after(self.playback_speed, self.playback)
    
    def step_forward(self):
        if self.current_step < len(self.simulation_data)-1:
            self.current_step += 1
            self.packet_animation_step = 0
            self.update_visualization()
            self.update_stats()
    
    def step_backward(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.packet_animation_step = 0
            self.update_visualization()
            self.update_stats()
    
    def update_speed(self, value):
        self.playback_speed = int(float(value))
    
    def toggle_packet_vis(self):
        self.show_packet_routes = not self.show_packet_routes
        self.update_visualization()


symulation_data = symulation()
with open("/home/jan/Informatyka/Projekt_indywidualny/logs/simulation_data.json", "w") as f:
    json.dump({"simulation_data": symulation_data}, f)
# Initialize the main application window

root = tk.Tk()
app = SensorNetworkVisualizer(root)
root.mainloop()