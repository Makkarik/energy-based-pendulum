import csv
import os

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from .reward import EnergyReward

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std

    def get_action(self, state, deterministic=False):
        mean, std = self.forward(state)
        if deterministic:
            return mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            return action

    def log_prob(self, state, action):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)

    def entropy(self, state):
        _, std = self.forward(state)
        return torch.log(std * torch.sqrt(torch.tensor(2 * np.pi * np.e))).sum(dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze()


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        tau=0.95,
        clip_param=0.2,
        ppo_epochs=10,
        mini_batch_size=64,
        entropy_coef=0.01,
        value_coef=0.5,
    ):
        self.gamma = gamma
        self.tau = tau
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value = ValueNetwork(state_dim, hidden_dim)

        self.optimizer = torch.optim.Adam([
            {"params": self.policy.parameters(), "lr": lr},
            {"params": self.value.parameters(), "lr": lr},
        ])

    def compute_advantages(self, rewards, values, masks):
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * masks[t] - values[t]
            gae = delta + self.gamma * self.tau * masks[t] * gae

            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def update(self, states, actions, rewards, masks, energy_rewards=None):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        masks = torch.FloatTensor(masks)

        # Compute values and advantages
        with torch.no_grad():
            values = self.value(states)
            advantages, returns = self.compute_advantages(rewards, values, masks)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get old log probabilities
        with torch.no_grad():
            old_log_probs = self.policy.log_prob(states, actions)

        # Create dataset and dataloader for mini-batch updates
        dataset = TensorDataset(states, actions, old_log_probs, advantages, returns)
        dataloader = DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=True)

        # Track metrics
        policy_loss_epoch = 0
        value_loss_epoch = 0
        entropy_epoch = 0
        kl_epoch = 0

        # PPO update loop
        for _ in range(self.ppo_epochs):
            for (
                batch_states,
                batch_actions,
                batch_old_log_probs,
                batch_advantages,
                batch_returns,
            ) in dataloader:
                # Policy loss
                log_probs = self.policy.log_prob(batch_states, batch_actions)
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Calculate KL divergence (for logging only)
                kl_div = (batch_old_log_probs - log_probs).mean().item()

                # Clipped policy objective
                surrogate1 = ratio * batch_advantages
                surrogate2 = (
                    torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
                    * batch_advantages
                )
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                # Value loss
                value_pred = self.value(batch_states)
                value_loss = F.mse_loss(value_pred, batch_returns)

                # Entropy bonus
                entropy = self.policy.entropy(batch_states).mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                # Update networks
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Track metrics
                policy_loss_epoch += policy_loss.item()
                value_loss_epoch += value_loss.item()
                entropy_epoch += entropy.item()
                kl_epoch += kl_div

        # Average metrics over all batches
        num_batches = len(dataloader) * self.ppo_epochs
        return {
            "policy_loss": policy_loss_epoch / num_batches,
            "value_loss": value_loss_epoch / num_batches,
            "entropy": entropy_epoch / num_batches,
            "kl": kl_epoch / num_batches,
        }


def collect_trajectories(env, agent, energy_reward_calculator, num_steps=2048):
    states, actions, rewards, dones = [], [], [], []
    energy_rewards = []

    state, _ = env.reset(seed=SEED)
    episode_reward = 0
    episode_energy_reward = 0
    episode_length = 0
    episode_rewards = []
    episode_energy_rewards = []
    episode_lengths = []

    for _ in range(num_steps):
        with torch.no_grad():
            action = agent.policy.get_action(torch.FloatTensor(state)).detach().numpy()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Calculate energy-based reward using the provided class
        energy_reward = energy_reward_calculator(state)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        energy_rewards.append(energy_reward)
        dones.append(float(not done))  # Store as mask (0 if done, 1 otherwise)

        episode_reward += reward
        episode_energy_reward += energy_reward
        episode_length += 1

        if done:
            state, _ = env.reset()
            episode_rewards.append(episode_reward)
            episode_energy_rewards.append(episode_energy_reward)
            episode_lengths.append(episode_length)
            episode_reward = 0
            episode_energy_reward = 0
            episode_length = 0
        else:
            state = next_state

    if episode_length > 0:  # Add incomplete episode
        episode_rewards.append(episode_reward)
        episode_energy_rewards.append(episode_energy_reward)
        episode_lengths.append(episode_length)

    return {
        "states": np.array(states),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "energy_rewards": np.array(energy_rewards),
        "masks": np.array(dones),
        "episode_rewards": episode_rewards,
        "episode_energy_rewards": episode_energy_rewards,
        "episode_lengths": episode_lengths,
    }


def train_ppo(
    env_name,
    num_epochs=500,
    steps_per_epoch=4096,
    gamma=0.99,
    reward_type="rewards",
):
    # Create environment
    env = gym.make(env_name)

    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create agent
    agent = PPO(state_dim, action_dim, gamma=gamma)

    # Create energy reward calculator
    energy_reward_calculator = EnergyReward()
    best_reward = -np.inf

    # Create CSV logger
    os.makedirs("./results", exist_ok=True)
    csv_file = open(f"./results/ppo-train-{reward_type}.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch",
        "length",
        "reward",
        "energy",
        "policy_loss",
        "value_loss",
        "kl",
        "entropy",
    ])

    # Training loop
    for epoch in trange(num_epochs, desc="PPO training"):
        # Collect trajectories
        trajectories = collect_trajectories(
            env, agent, energy_reward_calculator, steps_per_epoch
        )

        # Update agent
        update_info = agent.update(
            trajectories["states"],
            trajectories["actions"],
            trajectories[reward_type],
            trajectories["masks"],
        )

        # Write to CSV
        for i in range(len(trajectories["episode_lengths"])):
            csv_writer.writerow([
                epoch,
                trajectories["episode_lengths"][i],
                trajectories["episode_rewards"][i],
                trajectories["episode_energy_rewards"][i],
                update_info["policy_loss"],
                update_info["value_loss"],
                update_info["kl"],
                update_info["entropy"],
            ])
        csv_file.flush()

        # Save the best model
        if np.mean(trajectories["episode_rewards"]) > best_reward:
            best_reward = np.mean(trajectories["episode_rewards"])
            torch.save(
                {
                    "policy": agent.policy.state_dict(),
                    "value": agent.value.state_dict(),
                },
                "./results/ppo-best.pt",
            )

    csv_file.close()
    env.close()


def evaluate(
    env_name, agent, num_episodes: int = 10, record_video: bool = True, reward_type: str = "reward"
):
    # Create environment
    if record_video:
        env = gym.make(env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            "results",
            episode_trigger=lambda x: x < 1,
            disable_logger=True,
            name_prefix="ppo-energy",
        )
    else:
        env = gym.make(env_name)

    # Create energy reward calculator
    energy_reward_calculator = EnergyReward()

    episode_rewards = []
    episode_energy_rewards = []
    episode_lengths = []

    for _ in trange(num_episodes, desc="PPO evaluation"):
        state, _ = env.reset(seed=np.random.randint(10000))
        done = False
        episode_reward = 0
        episode_energy_reward = 0
        episode_length = 0

        while not done:
            # Use deterministic action for evaluation
            with torch.no_grad():
                action = (
                    agent.policy.get_action(
                        torch.FloatTensor(state), deterministic=True
                    )
                    .detach()
                    .numpy()
                )

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Calculate energy-based reward
            energy_reward = energy_reward_calculator(state)

            episode_reward += reward
            episode_energy_reward += energy_reward
            episode_length += 1

            state = next_state

        episode_rewards.append(episode_reward)
        episode_energy_rewards.append(episode_energy_reward)
        episode_lengths.append(episode_length)

    data = pd.DataFrame(columns=["length", "reward", "energy"])
    data["length"] = episode_lengths
    data["reward"] = episode_rewards
    data["energy"] = episode_energy_rewards
    data.to_csv(f"./results/ppo-eval-{reward_type}.csv", index=False)

    # Calculate average metrics
    avg_reward = np.mean(episode_rewards)
    avg_energy_reward = np.mean(episode_energy_rewards)
    avg_length = np.mean(episode_lengths)

    print(f"Evaluation Results over {num_episodes} episodes:")
    print(f"Average Episode Length: {avg_length:.2f}")
    print(f"Average Episode Reward: {avg_reward:.2f}")
    print(f"Average Episode Energy Reward: {avg_energy_reward:.2f}")

    env.close()
    return avg_reward, avg_energy_reward, avg_length


def load_model(path, env_name):
    env = gym.make(env_name)

    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPO(state_dim, action_dim)
    checkpoint = torch.load(path)
    agent.policy.load_state_dict(checkpoint["policy"])
    agent.value.load_state_dict(checkpoint["value"])
    return agent
