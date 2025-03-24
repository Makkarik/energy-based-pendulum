"""Tools for training and evaluating PPO and TRPO agents on gymnasium environments."""

import csv
import os
from typing import Callable

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.nn import Module

from .ppo import PPO
from .reward import EnergyReward
from .trpo import TRPO


def collect_trajectories(
    env: gym.Env,
    agent: PPO | TRPO,
    energy_reward_func: Callable,
    num_steps: int = 2048,
    seed: int | None = None,
):
    """
    Collects trajectories by interacting with the environment using the provided agent.

    Parameters
    ----------
    env : gym.Env
        The environment to interact with.
    agent : PPO or TRPO
        The agent used to select actions.
    energy_reward_func : Callable
        A function to calculate the energy-based reward.
    num_steps : int, optional
        The number of steps to collect, by default 2048.
    seed : int or None, optional
        Seed for the environment's random number generator, by default None.

    Returns
    -------
    dict
        A dictionary containing the collected trajectories with the following keys:
        - "states" (np.ndarray): Array of states.
        - "actions" (np.ndarray): Array of actions.
        - "rewards" (np.ndarray): Array of rewards.
        - "energies" (np.ndarray): Array of energy-based rewards.
        - "masks" (np.ndarray): Array of masks (0 if done, 1 otherwise).
        - "episode_rewards" (list): List of total rewards per episode.
        - "episode_energies" (list): List of total energy-based rewards per episode.
        - "episode_lengths" (list): List of episode lengths.

    """
    states, actions, rewards, dones = [], [], [], []
    energies = []

    state, _ = env.reset(seed=seed)
    episode_reward = 0
    episode_energy = 0
    episode_length = 0
    episode_rewards = []
    episode_energies = []
    episode_lengths = []

    for _ in range(num_steps):
        action = agent.policy.get_action(torch.FloatTensor(state)).detach().numpy()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Calculate energy-based reward using the provided class
        energy = energy_reward_func(state)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        energies.append(energy)
        dones.append(float(not done))  # Store as mask (0 if done, 1 otherwise)

        episode_reward += reward
        episode_energy += energy
        episode_length += 1

        if done:
            state, _ = env.reset()
            episode_rewards.append(episode_reward)
            episode_energies.append(episode_energy)
            episode_lengths.append(episode_length)
            episode_reward = 0
            episode_energy = 0
            episode_length = 0
        else:
            state = next_state

    if episode_length > 0:  # Add incomplete episode
        episode_rewards.append(episode_reward)
        episode_energies.append(episode_energy)
        episode_lengths.append(episode_length)

    return {
        "states": np.array(states),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "energies": np.array(energies),
        "masks": np.array(dones),
        "episode_rewards": episode_rewards,
        "episode_energies": episode_energies,
        "episode_lengths": episode_lengths,
    }


def train(
    env_name: str,
    agent: PPO | TRPO,
    num_epochs: int = 500,
    steps_per_epoch: int = 4096,
    gamma: float = 0.99,
    reward_type: str = "reward",
    seed: int | None = None,
):
    """
    Train a reinforcement learning agent using PPO or TRPO algorithm.

    Parameters
    ----------
    env_name : str
        The name of the environment to train on.
    agent : PPO or TRPO
        The agent to be trained.
    num_epochs : int, optional
        The number of training epochs (default is 500).
    steps_per_epoch : int, optional
        The number of steps per epoch (default is 4096).
    gamma : float, optional
        The discount factor (default is 0.99).
    reward_type : str, optional
        The type of reward to use for training (default is "reward").
    seed : int or None, optional
        The random seed for reproducibility (default is None).

    Returns
    -------
    agent
        The trained agent.

    """
    # Create environment
    env = gym.make(env_name)

    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create agent
    agent = agent(state_dim, action_dim, gamma=gamma)
    name = agent.__class__.__name__.lower()

    # Create energy reward calculator
    energy_reward_func = EnergyReward()
    best_reward = -np.inf

    # Create CSV logger
    os.makedirs("./results", exist_ok=True)
    csv_file = open(f"./results/{name}-train-{reward_type}.csv", "w", newline="")
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
    for epoch in tqdm.trange(num_epochs, desc=f"{name.upper()} training"):
        # Collect trajectories
        trajectories = collect_trajectories(
            env, agent, energy_reward_func, steps_per_epoch, seed
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
                trajectories["episode_energies"][i],
                update_info["policy_loss"],
                update_info["value_loss"],
                update_info["kl"],
                update_info["entropy"],
            ])
        csv_file.flush()

        # Save model periodically
        if np.mean(trajectories[reward_type]) > best_reward:
            best_reward = np.mean(trajectories[reward_type])
            torch.save(
                {
                    "policy": agent.policy.state_dict(),
                    "value": agent.value.state_dict(),
                },
                f"./results/{name}-{reward_type}-best.pt",
            )

    csv_file.close()
    env.close()
    return agent


def evaluate(
    env_name: str,
    agent: PPO | TRPO,
    num_episodes: int = 10,
    record_video: bool = True,
    reward_type: str = "reward",
):
    """
    Evaluate a reinforcement learning agent in a specified environment.

    Parameters
    ----------
    env_name : str
        The name of the environment to evaluate the agent in.
    agent : PPO | TRPO
        The agent to be evaluated, which should be an instance of PPO or TRPO.
    num_episodes : int, optional
        The number of episodes to run for evaluation (default is 10).
    record_video : bool, optional
        Whether to record video of the evaluation episodes (default is True).
    reward_type : str, optional
        The type of reward to log (default is "reward").

    Returns
    -------
    avg_reward : float
        The average reward obtained over the evaluation episodes.
    avg_energy_reward : float
        The average energy-based reward obtained over the evaluation episodes.
    avg_length : float
        The average length of the episodes.

    Notes
    -----
    This function creates the environment, runs the specified number of episodes,
    logs the results to a CSV file, and prints the average metrics. If `record_video`
    is True, it records the first episode as a video.

    """
    name = agent.__class__.__name__.lower()

    # Create environment
    if record_video:
        env = gym.make(env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            "results",
            episode_trigger=lambda x: x < 1,
            disable_logger=True,
            name_prefix=f"{name}-{reward_type}",
        )
    else:
        env = gym.make(env_name)

    # Create energy reward calculator
    energy_reward_calculator = EnergyReward()

    episode_rewards = []
    episode_energy_rewards = []
    episode_lengths = []

    # Create CSV logger
    os.makedirs("./results", exist_ok=True)
    csv_file = open(f"./results/{name}-eval-{reward_type}.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "length",
        "reward",
        "energy",
    ])

    for _ in tqdm.trange(num_episodes, desc=f"{name.upper()} evaluation"):
        state, _ = env.reset(seed=np.random.randint(10000))
        done = False
        episode_reward = 0
        episode_energy = 0
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
            episode_energy += energy_reward
            episode_length += 1

            state = next_state

        csv_writer.writerow([episode_length, episode_reward, episode_energy])

        episode_rewards.append(episode_reward)
        episode_energy_rewards.append(episode_energy)
        episode_lengths.append(episode_length)

    data = pd.DataFrame(columns=["length", "reward", "energy"])
    data["length"] = episode_lengths
    data["reward"] = episode_rewards
    data["energy"] = episode_energy_rewards
    data.to_csv(f"./results/{name}-eval-{reward_type}.csv", index=False)

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


def load_model(path: str, agent: Module) -> Module:
    """
    Load a trained model from a checkpoint file.

    Parameters
    ----------
    path : str
        The file path to the checkpoint file.
    agent : Module
        The agent module to load the model into.

    Returns
    -------
    Module
        The agent module with the loaded model.

    """
    checkpoint = torch.load(path)
    agent.policy.load_state_dict(checkpoint["policy"])
    agent.value.load_state_dict(checkpoint["value"])
    return agent
