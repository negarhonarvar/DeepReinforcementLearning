import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            # nn.Linear(64, 64),
            # nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        mean = self.actor(state)
        std_dev = torch.exp(self.log_std)
        value = self.critic(state)
        return mean, std_dev, value


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.dones[:]


class PPO:
    def __init__(self, hyperparams):
        self.config = hyperparams
        self.gamma = hyperparams['gamma']
        self.K_epochs = hyperparams['K_epochs']
        self.policy = ActorCritic(hyperparams['state_dim'], hyperparams['action_dim']).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=hyperparams['lr'])
        self.policy_old = ActorCritic(hyperparams['state_dim'], hyperparams['action_dim']).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        self.target_kl = hyperparams['target_kl']
        self.batch_size = 32
        self.entropy_coef = 0.01
        self.mse_coef = 0.5
        self.beta = hyperparams['beta']
        self.eps_clip = 0.2

    def select_action(self, memory, state):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        mean, std_dev, _ = self.policy_old(state)
        dist = Normal(mean, std_dev)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1)
        memory.states.append(state)
        memory.actions.append(action)
        memory.log_probs.append(log_prob)

        return action.squeeze(0).detach().cpu().numpy()

    def Learn(self, memory):
            rewards = torch.tensor(memory.rewards, device=device, dtype=torch.float)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            old_states = torch.stack(memory.states).detach()
            old_actions = torch.stack(memory.actions).detach()
            old_log_probs = torch.stack(memory.log_probs).detach()

            for _ in range(self.K_epochs):
                for i in range(0, len(old_states), self.batch_size):
                    indices = slice(i, min(i + self.batch_size, len(old_states)))
                    sampled_states = old_states[indices]
                    sampled_actions = old_actions[indices]
                    sampled_log_probs = old_log_probs[indices]
                    sampled_rewards = rewards[indices]
                    mean, std_dev, state_values = self.policy(sampled_states)
                    dist = Normal(mean, std_dev)
                    new_log_probs = dist.log_prob(sampled_actions).sum(dim=2)
                    kl_divergence = torch.distributions.kl_divergence(Normal(mean, std_dev),
                        Normal(mean.detach(), std_dev.detach()) ).mean()
                    advantages = sampled_rewards - state_values.squeeze()
                    ratios = torch.exp(new_log_probs - sampled_log_probs).unsqueeze(-1).squeeze(2)
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
                    loss = -torch.min(surr1, surr2).mean() + self.mse_coef * self.MseLoss(state_values.squeeze(),
                                                    sampled_rewards) - self.entropy_coef * dist.entropy().mean()
                    loss += self.beta * kl_divergence
                    if kl_divergence.item() > 1.5 * self.target_kl:
                        self.beta *= 2
                    elif kl_divergence.item() < self.target_kl / 1.5:
                        self.beta /= 2
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            self.policy_old.load_state_dict(self.policy.state_dict())
            return loss.detach().numpy()

    def save_model(self, filename):
        torch.save(self.policy_old.state_dict(), filename)


def train(hyperparams, render=False):
    env = gym.make(hyperparams['env_name'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ppo = PPO(hyperparams)
    memory = Memory()
    score_history = []
    loss_history = []
    for i_episode in range(0, 500):
        state = env.reset()[0]
        state = np.array(state).reshape(1, -1)
        state = torch.FloatTensor(state).to(device)
        score = 0
        step_size = 0
        while True:
            action = ppo.select_action(memory, state)
            action = action.squeeze()
            state, reward, done, _, _ = env.step(action)
            state = np.array(state).reshape(1, -1)
            state = torch.FloatTensor(state).to(device)
            memory.rewards.append(reward)
            memory.dones.append(done)
            score += reward
            step_size += 1
            # print("step size is :" , step_size ," the score is :", score)
            if done or step_size >= 1000:
                print(score)
                break
        score_history.append(score)
        loss_history.append(ppo.Learn(memory))
        memory.clear_memory()

        # Print the current episode and score
        result = (f"Episode: {i_episode}, "
                  f"Steps: {step_size}, "
                  f"Reward: {score:.2f}, ")
        print(result)
        if i_episode % 20 == 0:
            avg_score = np.mean(score_history[-20:])
            print(f'Episode: {i_episode}, Average Score: {avg_score:.2f}')

        if i_episode % 500 == 0:
            ppo.save_model(f'checkpoint_{i_episode}.pth')
    sma = np.convolve(loss_history, np.ones(50) / 50, mode='valid')
    plt.figure()
    plt.title("Loss")
    plt.plot(loss_history, label='Loss', color='#F6CE3B', alpha=1)
    plt.plot(sma, label='SMA 50', color='#385DAA')
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('./kl=0.1Loss_plot.png', format='png', dpi=600, bbox_inches='tight')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    plt.clf()
    plt.close()

    sma = np.convolve(score_history,np.ones(50)/50, mode='valid')
    plt.figure()
    plt.title("Rewards")
    plt.plot(score_history, label='Raw Reward', color='#F6CE3B', alpha=1)
    plt.plot(sma, label='SMA 50', color='#385DAA')
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.legend()
    plt.savefig('./kl=0.1reward_plot.png', format='png', dpi=600, bbox_inches='tight')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    plt.clf()
    plt.close()


if __name__ == '__main__':
    hyperparams = {
        'env_name': 'Swimmer-v4',
        'state_dim': 8,
        'action_dim': 2,
        'lr': 0.0003,
        'gamma': 0.99,
        'K_epochs': 4,
        'max_episodes': 1000,
        'max_steps': 1000,
        'target_kl': 0.1,
        'beta': 0.99

    }
    train(hyperparams, render=False)
