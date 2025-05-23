import gymnasium as gym

from utils import plot_learning_curve
import numpy as np


env = gym.make('LunarLander-v3')

observation = env.reset()

print(observation)

``