import os
import gc
import torch
import pygame
import pickle
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt


class QTable:
    def __init__(self, observation_space, action_space, bin_size=100):
        # Initialize bins
        self.bins = [
            np.linspace(-4.8, 4.8, bin_size),  # For Cart Position
            np.linspace(-25, 25, bin_size),      # For Cart Velocity
            np.linspace(-0.418, 0.418, bin_size), # For Pole Angle
            np.linspace(-25, 25, bin_size)       # For Pole Angular Velocity
        ]
        self.q_table = np.zeros([bin_size] * observation_space.shape[0] + [action_space.n])

    def discretize_state(self, state):
        # Discretize continuous states
        state_discrete = []
        for i in range(len(state)):
            state_discrete.append(np.digitize(state[i], self.bins[i]) - 1)
        return tuple(state_discrete)
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

class EpsilonGreedyPolicy:
    def __init__(self, epsilon_max, epsilon_decay, epsilon_min, q_table):
        self.epsilon = epsilon_max
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = q_table

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.q_table.shape[-1])
        else:
            return np.argmax(self.q_table[state])

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class SARSA:
    def __init__(self, env, q_table, policy, Learning_rate=0.25, Discount_factor=0.995, max_episode=8000, max_step=500, render=False):
        self.env = env
        self.q_table = q_table
        self.policy = policy
        self.Learning_rate = Learning_rate
        self.Discount_factor = Discount_factor
        self.max_episode = max_episode
        self.max_step = max_step
        self.render = render
        self.best_average_reward = -float('inf')

    def train(self):
        reward_history, epsilon_history, loss_history = [], [], []
        for episode in range(self.max_episode+1):
            state, _ = self.env.reset(seed=seed)  
            state = self.q_table.discretize_state(state)
            action = self.policy.select_action(state)
            episode_reward, total_loss = 0, 0

            for step in range(self.max_step):
                if self.render:
                    self.env.render()

                next_state_raw, reward, done,_, _ = self.env.step(action)
                next_state = self.q_table.discretize_state(next_state_raw)
                next_action = self.policy.select_action(next_state)

                td_target = reward + self.Discount_factor * self.q_table.q_table[next_state + (next_action,)] # yi
                td_error = td_target - self.q_table.q_table[state + (action,)] # yi - Q(s,a)
                self.q_table.q_table[state + (action,)] += self.Learning_rate * td_error #Q(s,a) <- Q(s,a) + alpha(yi - Q(s,a))

                total_loss += td_error ** 2 # loss = 1/n * sigma(yi - Q(s,a))**2
                state, action = next_state, next_action
                episode_reward += reward

                if done:
                    break

            reward_history.append(episode_reward)
            epsilon_history.append(self.policy.epsilon)
            loss_history.append(total_loss)
            self.policy.update_epsilon()

            if episode % 4000 == 0 and episode > 0:
                self.q_table.save('d:\\term8\Deep Reinforcement Learning\HWs\HW1\SARSA\SARSA' + '_' + f'{episode}' + '.pth')
                print('\n~~~~~~Interval Save: Model saved.\n')
                self.plot_progress(reward_history, epsilon_history, loss_history, episode)
            result = (f"Episode: {episode}, "
                      f"Raw Reward: {episode_reward:.2f}, ")
            print(result)

        self.env.close()
        return reward_history, epsilon_history, loss_history
    
    def test(self, model_path, num_episodes=10):
        # Load the saved Q-table
        with open(model_path, 'rb') as f:
            self.q_table.q_table = pickle.load(f)
        
        total_rewards = []
        for episode in range(num_episodes):
            state, _ = self.env.reset(seed=seed)
            state = self.q_table.discretize_state(state)
            episode_reward = 0

            for _ in range(self.max_step):
                action = np.argmax(self.q_table.q_table[state])  # Always choose the best action
                next_state_raw, reward, done, _, _ = self.env.step(action)
                next_state = self.q_table.discretize_state(next_state_raw)

                state = next_state
                episode_reward += reward

                if done:
                    break

            total_rewards.append(episode_reward)
            print(f"Test Episode: {episode + 1}, Reward: {episode_reward}")
        
        average_reward = sum(total_rewards) / num_episodes
        print(f"Average Reward over {num_episodes} episodes: {average_reward}")
        return total_rewards

    def plot_progress(self, reward_history, epsilon_history, loss_history, episode):
        plt.figure()
        plt.title("Epsilon in epsilon-greedy policy")
        plt.plot(epsilon_history, label='epsilon', color='#EB88E2')
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.legend()

        # Only save as file if last episode
        if episode == self.max_episode:
            plt.savefig('d:\\term8\Deep Reinforcement Learning\HWs\HW1\SARSA/Epsilon_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()

        # Calculate the Simple Moving Average (SMA) with a window size of 50
        sma = np.convolve(reward_history, np.ones(50)/50, mode='valid')
        plt.figure()
        plt.title("Reward for each episode")
        plt.plot(reward_history, label='Raw Reward', color='#FF48B0', alpha=1)
        plt.plot(sma, label='Average', color='#72DDF7')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()

        # Only save as file if last episode
        if episode == self.max_episode:
            plt.savefig('d:\\term8\Deep Reinforcement Learning\HWs\HW1\SARSA\\reward_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()

        loss = np.convolve(loss_history, np.ones(50)/50, mode='valid') # loss = 1/n * sigma(yi - Q(s,a))**2
        plt.figure()
        plt.title("Loss")
        plt.plot(loss_history, label='Loss', color='#832161', alpha=1)
        plt.plot(loss, label='Average', color='#72DDF7')
        plt.xlabel("Episode")
        plt.ylabel("Loss")

        # Only save as file if last episode
        if episode == self.max_episode:
            plt.savefig('d:\\term8\Deep Reinforcement Learning\HWs\HW1\SARSA\Loss_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()


    def save_q_table(self, episode):
        with open(f"best_q_table_episode_{episode}.pkl", "wb") as f:
            pickle.dump(self.q_table.q_table, f)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gc.collect()
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Used for debugging; CUDA related errors shown immediately.

# Seed everything for reproducible results
seed = 2024
np.random.seed(seed)
np.random.default_rng(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

render = True
env = gym.make('CartPole-v1',max_episode_steps=500,render_mode = "human" if render else None)
observation_space = env.observation_space
action_space = env.action_space
print(action_space)
action_space.seed(seed)
print(action_space)

q_table = QTable(observation_space, action_space)
policy = EpsilonGreedyPolicy(1.0, 0.999, 0.01, q_table.q_table)
sarsa_agent = SARSA(env, q_table, policy, render=True)  # Set render=True to visualize training

# reward_history, epsilon_history, loss_history = sarsa_agent.train()

model_path = 'd:\\term8\\Deep Reinforcement Learning\\HWs\\HW1\\SARSA\\SARSA_4000.pth'
test_rewards = sarsa_agent.test(model_path, num_episodes=4)