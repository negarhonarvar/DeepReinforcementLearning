import os
import gc
import torch
import numpy as np
import random
import torch.nn as nn
import gymnasium as gym
import random
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

global target_Q
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gc.collect()
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Used for debugging; CUDA related errors shown immediately.

# Seed everything for reproducible results
seed = 42
np.random.seed(seed)
np.random.default_rng(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Replay_Memory:

    def __init__(self, memory_capacity=10000):
        self.buffer = deque(maxlen=memory_capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.stack(states), actions, rewards, np.stack(next_states), dones

    def __len__(self):
        return len(self.buffer)


class D3QN(torch.nn.Module):

    def __init__(self, state_size=8, action_size=4, hidden_size=64):
        super(D3QN, self).__init__()
        self.layer1 = torch.nn.Linear(state_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.advantage = torch.nn.Linear(hidden_size, action_size)
        self.value = torch.nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        advantage = self.advantage(x)
        value = self.value(x)
        return value + (advantage - advantage.max(dim=1, keepdim=True)[0])


class D3QN_Agent:
    def __init__(self, state_size=8, action_size=4, hidden_size=64,
                 learning_rate=1e-3, discount_factor=0.99, memory_capacity=10000, batch_size=64):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.action_size = action_size
        self.main_network = D3QN(state_size, action_size, hidden_size).to(self.device)
        self.target_network = D3QN(state_size, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=learning_rate)
        self.memory = Replay_Memory(memory_capacity)

    def learn(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            self.update_model()

    def select_action(self, state, eps=0.):
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.main_network.eval()
            with torch.no_grad():
                action_values = self.main_network(state)

            self.main_network.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def update_model(self):

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(np.array(dones).astype(np.uint8)).float().to(self.device)

        q_values = self.main_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        next_action_values = self.main_network(next_states).max(1)[1].unsqueeze(-1)

        next_q_values = self.target_network(next_states).gather(1, next_action_values).detach().squeeze(-1)
        target_Q = next_q_values

        expected_q_values = rewards + self.discount_factor * next_q_values * (1 - dones)

        loss = torch.nn.MSELoss()(q_values, expected_q_values)

        self.optimizer.zero_grad()

        loss.backward()

        # Step the optimizer
        self.optimizer.step()

    def hard_update(self):
        self.target_network.load_state_dict(self.main_network.state_dict())


def train(agent, env, max_episodes=2000, epsilon_max=1.0, epsilon_min=0.01, epsilon_decay=0.995, target_update=10):

    target_reward_history = []
    scores = []
    final_score = deque(maxlen=100)
    epsilon_max = epsilon_max

    # Loop over episodes
    for current_episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        score = 0

        while True:

            action = agent.select_action(state, epsilon_max)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        final_score.append(score)
        scores.append(score)

        epsilon_max = max(epsilon_min, epsilon_decay * epsilon_max)

        print(f"\rEpisode {current_episode}\tAverage Score: {np.mean(final_score):.2f}", end="")
        if current_episode % target_update == 0:
            agent.hard_update()
        if current_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(current_episode, np.mean(final_score)))

        if current_episode % 100 == 0 and np.mean(final_score) >= 200:
            save_path = 'd:\\term8\Deep Reinforcement Learning\HWs\HW2\D3QN'
            agent.save(save_path + '_' + f'{current_episode}' + '.pth')
            break

    plt.figure()
    plt.title("Final Scores")
    plt.plot(final_score, label='score', color='#FF48B0', alpha=1)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig('d:\\term8\Deep Reinforcement Learning\HWs\HW2\DQN/reward_plot.png', format='png', dpi=600,
                bbox_inches='tight')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    plt.clf()
    plt.close()

    return scores


def test(env, agent):

    score = 0
    state, _ = env.reset(seed=42)

    while True:
        action = agent.select_action(state, 0)
        state, reward, terminated, truncated, _ = env.learn(action)
        done = terminated or truncated

        score += reward
        if done:
            break

    return score



train_mode = True
render = not train_mode
env = gym.make('LunarLander-v2', render_mode="human" if render else None)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = D3QN_Agent(state_size, action_size)
if train_mode:
    scores = train(agent, env)
else:
    score = test(env, agent)
    print("Score obtained:", score)





