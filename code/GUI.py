import tkinter as tk
from tkinter import ttk
import numpy as np
import math
from PIL import Image, ImageTk, ImageDraw
import time
from symulacja_sieci_sensorowej import SensorNetwork, Sensor, NETWORK_AREA, BATTERY_CAPACITY, DURATION
from typing import List
# from traning import traning

class SensorNetworkGUI(tk.Tk):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.title("Sensor Network Simulation")
        self.geometry("1000x800")
        self.current_step = 0
        self.is_running = False
        self.speed = 1.0
        self.setup_ui()
        
    def setup_ui(self):
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(self.control_frame, text="Start", command=self.start_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.pause_button = ttk.Button(self.control_frame, text="Pause", command=self.pause_simulation)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = ttk.Button(self.control_frame, text="Reset", command=self.reset_simulation)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        self.speed_label = ttk.Label(self.control_frame, text="Speed:")
        self.speed_label.pack(side=tk.LEFT, padx=5)
        
        self.speed_slider = ttk.Scale(self.control_frame, from_=0.1, to=5.0, value=1.0, 
                                    command=self.update_speed)
        self.speed_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.info_frame = ttk.LabelFrame(self.control_frame, text="Simulation Info")
        self.info_frame.pack(side=tk.RIGHT, padx=5)
        
        self.step_label = ttk.Label(self.info_frame, text="Step: 0/0")
        self.step_label.pack(side=tk.LEFT, padx=5)
        
        self.data_label = ttk.Label(self.info_frame, text="Data collected: 0")
        self.data_label.pack(side=tk.LEFT, padx=5)
        
        self.energy_label = ttk.Label(self.info_frame, text="Avg energy: 100%")
        self.energy_label.pack(side=tk.LEFT, padx=5)
        
        self.init_visualization()
        
    def init_visualization(self):
        self.img_width = 800
        self.img_height = 600
        self.scale_x = self.img_width / NETWORK_AREA[0]
        self.scale_y = self.img_height / NETWORK_AREA[1]
        
        # Utwórz obraz z przezroczystością
        self.base_image = Image.new("RGBA", (self.img_width, self.img_height), (255, 255, 255, 255))
        self.draw = ImageDraw.Draw(self.base_image)
        
        # Narysuj główną jednostkę
        main_unit = self.network.sensors[0]
        x, y = self.scale_position(main_unit.position[0])
        self.draw_main_unit(x, y, main_unit.transfer_coverage_distance * self.scale_x)
        
        # Narysuj sensory
        for sensor in self.network.sensors[1:]:
            x, y = self.scale_position(sensor.position[0])
            self.draw_sensor(x, y, sensor.coverage_radius * self.scale_x, 
                           sensor.transfer_coverage_distance * self.scale_x, 
                           sensor.energy / BATTERY_CAPACITY)
        
        self.update_canvas()
        
    def scale_position(self, pos):
        return pos[0] * self.scale_x, pos[1] * self.scale_y
        
    def draw_main_unit(self, x, y, radius):
        # Utwórz osobny obraz dla przezroczystych elementów
        overlay = Image.new("RGBA", (self.img_width, self.img_height), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Zakres transferu (czerwony z przezroczystością)
        overlay_draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                           fill=(255, 0, 0, 30), outline=(255, 0, 0, 150))
        
        # Połącz warstwy
        self.base_image = Image.alpha_composite(self.base_image, overlay)
        self.draw = ImageDraw.Draw(self.base_image)
        
        # Ikona głównej jednostki (bez przezroczystości)
        size = 20
        self.draw.rectangle([x-size/2, y-size/2, x+size/2, y+size/2], 
                          fill="red", outline="black")
        
    def draw_sensor(self, x, y, coverage_radius, transfer_radius, energy_level):
        # Utwórz osobny obraz dla przezroczystych elementów
        overlay = Image.new("RGBA", (self.img_width, self.img_height), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Zakres transferu (zielony z przezroczystością)
        overlay_draw.ellipse([x-transfer_radius, y-transfer_radius, 
                            x+transfer_radius, y+transfer_radius], 
                           fill=(0, 255, 0, 25), outline=(0, 255, 0, 100))
        
        # Zakres pokrycia (niebieski z przezroczystością)
        overlay_draw.ellipse([x-coverage_radius, y-coverage_radius, 
                            x+coverage_radius, y+coverage_radius], 
                           fill=(0, 0, 255, 30), outline=(0, 0, 255, 150))
        
        # Połącz warstwy
        self.base_image = Image.alpha_composite(self.base_image, overlay)
        self.draw = ImageDraw.Draw(self.base_image)
        
        # Ikona sensora (kolor zależny od energii)
        r, g = 255, int(255 * energy_level)
        color = f'#{r:02x}{g:02x}00'
        size = 15
        self.draw.ellipse([x-size/2, y-size/2, x+size/2, y+size/2], 
                         fill=color, outline="black")
        
    def update_canvas(self):
        # Konwertuj RGBA do RGB (Tkinter nie obsługuje przezroczystości bezpośrednio)
        rgb_image = self.base_image.convert("RGB")
        self.tk_image = ImageTk.PhotoImage(rgb_image)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)
        
    def update_info(self):
        active_sensors = sum(1 for s in self.network.sensors if s.is_active)
        energy_levels = [s.energy for s in self.network.sensors]
        
        self.step_label.config(text=f"Step: {self.current_step}/{DURATION}")
        self.data_label.config(text=f"Data collected: {len(self.network.sensors[0].colect_data_by_network)}")
        self.energy_label.config(text=f"Avg energy: {np.mean(energy_levels)/BATTERY_CAPACITY*100:.1f}%")
        
    def start_simulation(self):
        if not self.is_running:
            self.is_running = True
            self.run_simulation()
            
    def pause_simulation(self):
        self.is_running = False
        
    def reset_simulation(self):
        self.is_running = False
        self.current_step = 0
        self.network = SensorNetwork(NETWORK_AREA)
        self.init_visualization()
        self.update_info()
        
    def update_speed(self, value):
        self.speed = float(value)
        
    def run_simulation(self):
        if self.current_step >= DURATION or not self.is_running:
            self.is_running = False
            return
            
        
        observation, reward, done, _ = self.network.step()
        self.current_step += 1
        
        self.init_visualization()
        
        for data_packet in self.network.sensors[0].colect_data_by_network:
            path = self.trace_data_path(data_packet)
            self.draw_data_flow(path)
        
        self.update_info()
        
        delay = int(1000 / self.speed)
        self.after(delay, self.run_simulation)
        
    def draw_data_flow(self, path):
        for arrow in self.data_arrows:
            self.canvas.delete(arrow)
        self.data_arrows = []
        
        for i in range(len(path)-1):
            x1, y1 = self.scale_position(path[i].position[0])
            x2, y2 = self.scale_position(path[i+1].position[0])
            
            arrow = self.canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, 
                                          fill="red", width=2, arrowshape=(8,10,5))
            self.data_arrows.append(arrow)
        
    def trace_data_path(self, data_packet):
        path = []
        current_sensor_id = data_packet.device
        path.append(self.network.sensors[current_sensor_id])
        
        while current_sensor_id != 0:
            if not self.network.sensors[current_sensor_id].routing_table:
                break
            next_hop = min(
                self.network.sensors[current_sensor_id].routing_table,
                key=lambda neighbor: np.linalg.norm(neighbor.position - self.network.sensors[0].position))
            path.append(next_hop)
            current_sensor_id = next_hop.id
        
        return path


network = SensorNetwork(NETWORK_AREA)
app = SensorNetworkGUI(network)
app.mainloop()