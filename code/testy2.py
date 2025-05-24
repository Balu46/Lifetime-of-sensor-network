import gymnasium as gym

# Tworzenie środowiska z renderowaniem
env = gym.make("LunarLander-v3", render_mode="human")  # "human" pokazuje okno animacji

# Resetowanie środowiska
observation, info = env.reset()

for _ in range(1000):  # 1000 kroków symulacji
    action = env.action_space.sample()  # Losowa akcja (do testowania)
    observation, reward, terminated, truncated, info = env.step(action)
    
    print(reward) # Wyświetlenie informacji o stanie
    
    if terminated or truncated:
        observation, info = env.reset()  # Restart jeśli epizod się zakończył

env.close()  # Zamknięcie środowiska





# if __name__ == '__main__':
#     env = gym.make('LunarLander-v2')
#     agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
#                   eps_end=0.01, input_dims=[8], lr=0.003)
#     scores, eps_history = [], []
#     n_games = 500

#     for i in range(n_games):
#         score = 0
#         done = False
#         observation = env.reset()
#         while not done:
#             action = agent.choose_action(observation)
#             observation_, reward, done, info = env.step(action)
#             score += reward
#             agent.store_transition(observation, action, reward,
#                                   observation_, done)
#             agent.learn()
#             observation = observation_
#         scores.append(score)
#         eps_history.append(agent.epsilon)

#         avg_score = np.mean(scores[-100:])

#         print('episode ', i, 'score %.2f' % score,
#               'average score %.2f' % avg_score,
#               'epsilon %.2f' % agent.epsilon)
    
#     x = [i+1 for i in range(n_games)]
#     filename = 'lunar_lander_2020.png'
#     plot_learning_curve(x, scores, eps_history, filename)

# Wyniki
# print(f"Dane zebrane przez główną jednostkę: {len(network.sensors[0].colect_data_by_network)}")
# print(f"Energia pozostała w sensorach: {[s.energy for s in network.sensors[1:]]}")

# # Wizualizacja
# plt.figure(figsize=(10, 8))
# plt.scatter(
#     [s.position[0] for s in network.sensors],
#     [s.position[1] for s in network.sensors],
#     c=['red' if s.id == 0 else 'blue' for s in network.sensors],
#     label='Główna jednostka (czerwony)'
# )
# plt.title("Pozycje sensorów w sieci")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.grid(True)
# plt.legend()
# plt.savefig("/home/jan/Informatyka/Projekt_indywidualny/temp4.png"


# )