# Javier Prado - 21486
# Gabriel García - 21352
# Bryan España - 21550
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import matplotlib.pyplot as plt
import pickle

from gymnasium.envs.toy_text.frozen_lake import generate_random_map
def run(episodes, is_training =True,render= False):
    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=4), is_slippery=True, render_mode='human' if render else None)
    
    if (is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # Inicializar un arreglo de 16 x 4
    else:
        f = open('frozen_lake_model.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    #Qlearning formula depende de 2 hyperparametros
    learning_rate_a = 0.9 # Tasa de aprendizaje (alpha)
    discount_factor_g = 0.9 # Factor de descuento (gamma)

    rewards_per_episode = np.zeros(episodes)  # Para rastrear la recompensa total en cada episodio

    epsilon = 1         # 1 = 100% de acciones aleatorias
    epsilon_decay_rate = 0.0001        # Tasa de decaimiento de epsilon
    rng = np.random.default_rng()   # Generador de números aleatorios

    for i in range(episodes):
        state = env.reset()[0] # estado: 0 to 15 , 0 = top-left, 15 = bottom-right
        terminated = False     # True when it falls in a hole or reaches the goal
        truncated = False       # True when actions > 200

        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state,:])

            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )
            
            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake_rewards.png')

    if is_training:
        #uso de pickle para guardar el modelo
        f = open('frozen_lake_model.pkl', 'wb')
        pickle.dump(q, f)
        f.close()

if __name__=="__main__":
    run(1000, is_training=True)