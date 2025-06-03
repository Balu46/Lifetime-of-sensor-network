
# 👤 User Documentation

## 1. How to Use the Application

### Running the Application

```bash
python GUI.py
```

---

## 2. Key Features

### Graphical User Interface (GUI)


**Main GUI elements:**

1. **Visualization Panel** – displays the current state of the network  
2. **Playback Controls** – buttons to control the simulation (start/pause/step)  
3. **Statistics** – shows current network metrics  
4. **Energy Chart** – displays energy levels of individual sensors

### Simulation

- Automatic **energy optimization** using the DQN algorithm  
- **Data routing visualization**  
- **Sensor status monitoring** (active / sleeping / inactive)

---

## 3. Known Issues

| Issue                                  | Workaround                                                   |
|----------------------------------------|--------------------------------------------------------------|
| No data on first launch                | Make sure `simulation_data.json` file exists in the directory |
| Model loading error                    | Check the path in the `Agent.__init__()` constructor         |
| Simulation stops too early             | Increase `BATTERY_CAPACITY` in `Hyperparameters.py`          |

---

## 4. Unusual Behaviors

1. **Sudden sensor shutdown** – when energy drops below the threshold, a sensor shuts down without warning  
2. **Long packet routes** – packets sometimes take detours to reach the main unit  
3. **Delayed GUI updates** – with many sensors, the GUI may respond more slowly

---

## 💡 Additional Notes

### 1. Optimization

Key parameters can be adjusted in `Hyperparameters.py`:

- `BATTERY_CAPACITY` – sensor lifespan  
- `TRANSMISSION_COST` – data transmission cost  
- `NETWORK_AREA` – size of the sensor network area

### 2. Logs

Detailed simulation logs are saved in:

- `/logs/stan_energi_po_symulacji.txt`  
- `/logs/simulation_data.json`

### 3. GUI Options

The GUI allows for:

- Controlling **simulation speed**  
- Enabling/disabling **packet route visualization**  
- Browsing simulation history via a **next and past state buttons**
