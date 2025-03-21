import csv
import os

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
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

    def get_kl(self, state, other):
        mean1, std1 = self.forward(state)
        mean2, std2 = other.forward(state)

        dist1 = Normal(mean1, std1)
        dist2 = Normal(mean2, std2)

        kl = torch.distributions.kl.kl_divergence(dist1, dist2).sum(dim=-1).mean()
        return kl


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


class TRPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=64,
        gamma=0.99,
        tau=0.95,
        delta=0.01,
        damping=0.1,
        cg_iters=10,
        backtrack_iters=10,
        backtrack_coeff=0.8,
    ):
        self.gamma = gamma
        self.tau = tau
        self.delta = delta
        self.damping = damping
        self.cg_iters = cg_iters
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value = ValueNetwork(state_dim, hidden_dim)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=5e-4)

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

    def cg(self, Ax, b, iters=10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        r_dot_r = torch.dot(r, r)

        for _ in range(iters):
            Ap = Ax(p)
            alpha = r_dot_r / (torch.dot(p, Ap) + self.damping)

            x += alpha * p
            r -= alpha * Ap

            r_dot_r_new = torch.dot(r, r)
            beta = r_dot_r_new / r_dot_r
            r_dot_r = r_dot_r_new

            if r_dot_r < 1e-10:
                break

            p = r + beta * p

        return x

    def hessian_vector_product(self, states, old_policy, vector):
        kl = self.policy.get_kl(states, old_policy)

        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * vector).sum()
        grads = torch.autograd.grad(kl_v, self.policy.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])

        return flat_grad_grad_kl

    def surrogate_loss(self, states, actions, advantages, old_log_probs):
        log_probs = self.policy.log_prob(states, actions)
        ratio = torch.exp(log_probs - old_log_probs)
        return (ratio * advantages).mean()

    def update(self, states, actions, rewards, masks, energy_rewards=None):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        masks = torch.FloatTensor(masks)

        # Update value function
        values = self.value(states)
        advantages, returns = self.compute_advantages(rewards, values, masks)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Value loss
        value_loss = -F.mse_loss(self.value(states), returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Policy gradient
        old_policy = PolicyNetwork(states.shape[1], actions.shape[1])
        old_policy.load_state_dict(self.policy.state_dict())

        old_log_probs = old_policy.log_prob(states, actions).detach()

        # Compute policy gradient
        loss = self.surrogate_loss(states, actions, advantages, old_log_probs)
        grads = torch.autograd.grad(loss, self.policy.parameters())
        flat_grad = torch.cat([grad.view(-1) for grad in grads])

        # Compute search direction with conjugate gradient
        Ax = lambda x: self.hessian_vector_product(states, old_policy, x)  # noqa: E731
        step_dir = self.cg(Ax, flat_grad, self.cg_iters)

        # Compute step size with line search
        shs = 0.5 * (step_dir * Ax(step_dir)).sum(0, keepdim=True)
        lm = torch.sqrt(shs / self.delta)
        full_step = step_dir / lm

        # Get current parameters
        params = torch.cat([param.view(-1) for param in self.policy.parameters()])

        # Line search
        expected_improve = (flat_grad * full_step).sum(0, keepdim=True).item()

        success = False
        for i in range(self.backtrack_iters):
            step_size = self.backtrack_coeff**i
            new_params = params + step_size * full_step

            # Update policy parameters
            idx = 0
            for param in self.policy.parameters():
                param_size = param.numel()
                param.data.copy_(new_params[idx : idx + param_size].view(param.size()))
                idx += param_size

            # Compute new loss
            new_loss = self.surrogate_loss(
                states, actions, advantages, old_log_probs
            ).item()

            # Check improvement
            improve = new_loss - loss.item()
            if improve > 0.1 * step_size * expected_improve:
                success = True
                break

        if not success:
            # Restore old parameters
            idx = 0
            for param in self.policy.parameters():
                param_size = param.numel()
                param.data.copy_(params[idx : idx + param_size].view(param.size()))
                idx += param_size

        return {
            "policy_loss": -loss.item(),
            "value_loss": value_loss.item(),
            "kl": self.policy.get_kl(states, old_policy).item(),
            "entropy": -old_log_probs.mean().item(),
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


def train_trpo(
    env_name, num_epochs=500, steps_per_epoch=4096, gamma=0.99, reward_type='reward'
):
    # Create environment
    env = gym.make(env_name)

    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create agent
    agent = TRPO(state_dim, action_dim, gamma=gamma)

    # Create energy reward calculator
    energy_reward_calculator = EnergyReward()
    best_reward = -np.inf

    # Create CSV logger
    os.makedirs("./results", exist_ok=True)
    csv_file = open(f"./results/trpo-train-{reward_type}.csv", "w", newline="")
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
    for epoch in trange(num_epochs, desc="TRPO training"):
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

        # Save model periodically
        if np.mean(trajectories["rewards"]) > best_reward:
            best_reward = np.mean(trajectories["rewards"])
            torch.save(
                {
                    "policy": agent.policy.state_dict(),
                    "value": agent.value.state_dict(),
                },
                "./models/trpo-best.pt",
            )

    csv_file.close()
    env.close()


def evaluate(env_name, agent, num_episodes=10, record_video=True, reward_type: str = "reward"):
    # Create environment
    if record_video:
        env = gym.make(env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            "results",
            episode_trigger=lambda x: x < 1,
            disable_logger=True,
            name_prefix="trpo-energy",
        )
    else:
        env = gym.make(env_name)

    # Create energy reward calculator
    energy_reward_calculator = EnergyReward()

    episode_rewards = []
    episode_energy_rewards = []
    episode_lengths = []

    for _ in trange(num_episodes, desc="TRPO evaluation"):
        state, _ = env.reset(seed=np.random.randint(10000))
        done = False
        episode_reward = 0
        episode_energy_reward = 0
        episode_length = 0

        while not done:
            # Use deterministic action for evaluation
            action = (
                agent.policy.get_action(torch.FloatTensor(state), deterministic=True)
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
    data.to_csv(f"./results/trpo-eval-{reward_type}.csv", index=False)

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

    agent = TRPO(state_dim, action_dim)
    checkpoint = torch.load(path)
    agent.policy.load_state_dict(checkpoint["policy"])
    agent.value.load_state_dict(checkpoint["value"])
    return agent
