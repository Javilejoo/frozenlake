import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):

    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # Inicializar un arreglo de 16 x 4
    else:
        with open('frozen_lake4x4.pkl', 'rb') as f:
            q = pickle.load(f)

    learning_rate_a = 0.9 # Tasa de aprendizaje (alpha)
    discount_factor_g = 0.9 # Factor de descuento (gamma)
    epsilon = 1         # 1 = 100% de acciones aleatorias
    epsilon_decay_rate = 0.0001        # Tasa de decaimiento de epsilon
    rng = np.random.default_rng()   # Generador de números aleatorios

    rewards_per_episode = np.zeros(episodes)
    
    print("Entrenamiento en progreso...")
    for i in range(episodes):
        state = env.reset()[0]  # Obtener el estado inicial del entorno
        terminated = False      # True cuando cae en un agujero o alcanza la meta
        truncated = False       # True cuando acciones > 200
        
        total_reward = 0  # Para rastrear la recompensa total en cada episodio

        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # Acciones: 0=izquierda, 1=abajo, 2=derecha, 3=arriba
            else:
                action = np.argmax(q[state,:])

            new_state, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward  # Acumular la recompensa total
            
            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state

            if render:
                env.render()
                
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001

        rewards_per_episode[i] = total_reward  # Guardar la recompensa total del episodio
        
        # Imprimir algo por cada episodio
        print(f"Episodio {i+1}: Recompensa total: {total_reward}")
        
        # Imprimir el progreso del entrenamiento
        if (i+1) % (episodes // 10) == 0:
            percentage = ((i+1) / episodes) * 100
            print(f"   Progreso: {percentage:.2f}%")

    print("Entrenamiento completado.")
    env.close()

    # Graficar la recompensa total acumulada por episodio
    plt.plot(rewards_per_episode)
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa total')
    plt.title('Recompensa total acumulada por episodio')
    plt.savefig('frozen_lake4x4_rewards.png')

    if is_training:
        with open("frozen_lake4x4.pkl","wb") as f:
            pickle.dump(q, f)

if __name__ == '__main__':
    run(1000, is_training=True, render=True)
