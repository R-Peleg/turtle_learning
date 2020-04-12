import gym
import numpy as np
import time
from turtle_env import TurtleEnv
from turtle_agent import TurtleAgent
import matplotlib.pyplot as plt

def main():
    env = TurtleEnv()
    agent = TurtleAgent(env.observation_space, env.action_space)
    train_data = []
    DISPLAY_GAMES = 1000
    game_performance = []
    try:
        for game_index in range(10001):
            if game_index == 0 or game_index > 1000 and game_index % DISPLAY_GAMES == 0:
                env.enable_draw()
            else:
                env.disable_draw()
            env.reset()
            total_reward = 0
            action = env.action_space.sample()
            previous_observation = None
            start_time = time.time()
            for step_index in range(75):
                env.render()
                observation, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
                if previous_observation:
                    agent.feedback(previous_observation, action, observation, reward)
                action = agent.act(observation)
                previous_observation = observation
            if game_index % 100 == 0:
                print(f'Round {game_index}, total reward: {total_reward} (took {time.time() - start_time}s)')
            if game_index % 250 == 0:
                env.save_to_file(f'turtle_{game_index}.ps')
            game_performance.append(total_reward / step_index)
    except KeyboardInterrupt:
        print('Interrupted, showing graph')
    c = np.cumsum(np.insert(game_performance, 0, 0))
    performace_moving_avg = (c[100:] - c[:-100]) / float(100)
    plt.plot(performace_moving_avg)
    plt.show()


if __name__ == '__main__':
    main()
