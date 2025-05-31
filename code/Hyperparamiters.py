# Defoult Parameters of the sensor network, if you change them, you need to retrain model from scratch
NUM_OF_MAIN_UNITS = 1
NUM_OF_SENSORS_IN_NETWORK = 10 # Number of sensor nodes

BATTERY_CAPACITY = 100  # Maximum energy capacity (in units)

TRANSMISSION_COST = 5  # Energy cost per data transmission
MINING_COST = 1 # Energy cost per data mining operation

NETWORK_AREA = (100, 100)  # Network area size (units)


NUM_NODES = NUM_OF_MAIN_UNITS + NUM_OF_SENSORS_IN_NETWORK # do not change it

N_GAMES = 1 # Number of symulacion at once

N_GAMES_FOR_TRANING = 100 # Number of symulacion for traning

LOAD_BEST_MODEL = True # Load best model from file if you change hyperparameters, you need to change it to False and retrain model from scratch

SAVE_BEST_MODEL = True # Save best model to file

