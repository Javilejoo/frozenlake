import gymnasium as gym

def run():
    env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=True, render_mode='human')

    state = env.reset()  # Use reset() to get the initial state
    terminated = False  # True when falling in a hole or reaching the goal
    truncated = False  # True when action > 200

    while not terminated and not truncated:
        action = env.action_space.sample()
        step_result = env.step(action)  # Store the result in a variable
        print("Step result:", step_result)  # Print the result to see its structure
        # Adjust your code accordingly based on the structure of the step result

    env.close()

if __name__ == "__main__":
    run()
