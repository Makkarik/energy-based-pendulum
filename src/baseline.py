import os

import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm import trange

from reward import EnergyReward
from utils import mp4_to_gif

EPISODES = 1000
EPISODE_LENGTH = 1000
FOLDER = "./results"
SEED = 42


def main():
    # Create folder for results
    os.makedirs(FOLDER, exist_ok=True)
    # Define the environment
    env = gym.make("InvertedDoublePendulum-v5", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env=env,
        episode_trigger=lambda x: x == 0,
        video_length=EPISODE_LENGTH,
        video_folder=FOLDER,
        name_prefix="random",
    )
    env = gym.wrappers.RecordEpisodeStatistics(env=env, buffer_length=EPISODES)
    # Prepare seeds for the runs
    seeds = np.random.SeedSequence(entropy=SEED).generate_state(n_words=EPISODES)
    # Create the agent
    agent = np.random.default_rng(seed=SEED)
    # Prepare custom reward
    reward_func = EnergyReward()
    energy_rewards = []
    # Iterate over episodes
    for episode in trange(EPISODES, desc="Running the Random agent"):
        # Reset the environment
        obs, _ = env.reset(seed=int(seeds[episode]))
        end = False
        custom_reward = 0
        while not end:
            action = agent.uniform(
                low=env.action_space.low[0],
                high=env.action_space.high[0],
                size=(1,),
            )
            obs, _, terminated, truncated, _ = env.step(action)
            custom_reward += reward_func(obs)
            end = terminated or truncated
        # Add custom reward to the records
        energy_rewards.append(custom_reward)
    # Pack everything to the CSV file
    logs = pd.DataFrame(columns=["length", "reward", "energy"])
    logs["reward"] = env.return_queue
    logs["length"] = env.length_queue
    logs["energy"] = energy_rewards

    logs.to_csv(os.path.join(FOLDER, "random.csv"), index=False)
    print(f"Experiment has been completed. All files saved to {FOLDER}")


if __name__ == "__main__":
    main()
    mp4_to_gif(FOLDER)
