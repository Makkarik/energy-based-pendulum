import os
import csv
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
from torch import nn
from torch import multiprocessing

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

import warnings
warnings.filterwarnings("ignore")

def run_pipeline():
    # https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    num_cells = 256  # number of cells in each layer i.e. output dim.
    lr = 3e-4
    max_grad_norm = 1.0

    frames_per_batch = 1000
    # For a complete training, bring the number of frames up to 1M
    total_frames = 1_000

    sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = 10  # optimization steps per batch of data collected
    clip_epsilon = (
        0.2  # clip value for PPO loss: see the equation in the intro for more context.
    )
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4

    base_env = GymEnv("InvertedDoublePendulum-v4", device=device)

    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )

    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

    print("normalization constant shape:", env.transform[0].loc.shape)

    print("observation_spec:", env.observation_spec)
    print("reward_spec:", env.reward_spec)
    print("input_spec:", env.input_spec)
    print("action_spec (as defined by input_spec):", env.action_spec)

    check_env_specs(env)

    rollout = env.rollout(3)
    print("rollout of three steps:", rollout)
    print("Shape of the rollout TensorDict:", rollout.batch_size)

    actor_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
        NormalParamExtractor(),
    )

    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
        },
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )

    value_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(1, device=device),
    )

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    print("Running policy:", policy_module(env.reset()))
    print("Running value:", value_module(env.reset()))

    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        # these keys match by default but we set this for completeness
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )

    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""

    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for i, tensordict_data in enumerate(collector):
        # we now have a batch of data to work with. Let's learn something from it.
        for _ in range(num_epochs):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        logs["reward (sum)"].append(tensordict_data["next", "reward"].sum().item())
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        if i % 10 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our ``env`` horizon).
            # The ``rollout`` method of the ``env`` can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(1000, policy_module)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout
        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()

    save_animation(policy_module, env, device)
    
    return logs


def plot_results(logs):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(logs["reward"])
    plt.title("training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(logs["step_count"])
    plt.title("Max step count (training)")
    plt.subplot(2, 2, 3)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.subplot(2, 2, 4)
    plt.plot(logs["eval step_count"])
    plt.title("Max step count (test)")
    plt.show()


def log_episodes_to_csv(logs, filename):
    train_rewards = logs["reward (sum)"]
    train_durations = logs["step_count"]
    eval_rewards = logs["eval reward (sum)"]
    eval_durations = logs["eval step_count"]

    max_episodes = max(
        len(train_rewards),
        len(train_durations),
        len(eval_rewards),
        len(eval_durations)
    )

    with open(filename, 'w', newline='') as csvfile:
        fieldnames = [
            'episode',
            'train_reward',
            'train_duration',
            'eval_reward',
            'eval_duration'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for episode in range(max_episodes):
            train_reward = train_rewards[episode] if episode < len(train_rewards) else ''
            train_duration = train_durations[episode] if episode < len(train_durations) else ''

            eval_reward = eval_rewards[episode] if episode < len(eval_rewards) else ''
            eval_duration = eval_durations[episode] if episode < len(eval_durations) else ''

            writer.writerow({
                'episode': episode,
                'train_reward': train_reward,
                'train_duration': train_duration,
                'eval_reward': eval_reward,
                'eval_duration': eval_duration
            })

def save_animation(policy_module, training_env, device, num_episodes=1, filename_prefix="pendulum"):
    """
    Save animations of trained policy during both training and evaluation.
    
    Args:
        policy_module: The trained policy module
        training_env: The training environment (to get normalization stats)
        device: The device to run the environment on
        num_episodes: Number of episodes to record for each animation
        filename_prefix: Prefix for the output filenames
    
    Returns:
        dict: Paths to the created animation files
    """
    
    # Create directory for animations if it doesn't exist
    os.makedirs("animations", exist_ok=True)
    
    # Function to render and collect frames
    def collect_frames(env, policy, max_steps, exploration_type):
        with set_exploration_type(exploration_type), torch.no_grad():
            frames = []
            tensordict = env.reset()
            
            # Run the episode
            done = False
            step = 0
            
            while not done and step < max_steps:
                # Get the frame before taking action
                frame = env.base_env.render()
                frames.append(frame)
                
                # Take action
                action = policy(tensordict)
                tensordict = env.step(action)
                
                done = tensordict.get(("next", "done"), False).item()
                step += 1
            
            return frames
    
    # Create a new environment with render_mode for animations
    base_env = GymEnv("InvertedDoublePendulum-v4", device=device, render_mode="rgb_array")
    
    env = TransformedEnv(
        base_env,
        Compose(
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    
    # Copy normalization stats from training environment
    env.transform[0].loc = training_env.transform[0].loc.clone()
    env.transform[0].scale = training_env.transform[0].scale.clone()
    
    print("\nCreating animations...")
    
    # Save evaluation runs (deterministic policy)
    eval_frames_collection = []
    for episode in range(num_episodes):
        frames = collect_frames(
            env, 
            policy_module, 
            max_steps=1000, 
            exploration_type=ExplorationType.DETERMINISTIC
        )
        eval_frames_collection.append(frames)
        
        # Save each episode separately (uncomment if needed)
        # eval_filename = f"animations/{filename_prefix}_evaluation_episode_{episode+1}.gif"
        # imageio.mimsave(eval_filename, frames, fps=30)
        # print(f"Saved evaluation animation to {eval_filename}")
    
    # Create a combined animation with all episodes
    combined_eval_filename = f"animations/{filename_prefix}_evaluation_combined.gif"
    
    # Flatten the list of frames
    all_eval_frames = [frame for episode_frames in eval_frames_collection for frame in episode_frames]
    
    # Save combined animations
    imageio.mimsave(combined_eval_filename, all_eval_frames, fps=30)
    
    print(f"Saved combined evaluation animation to {combined_eval_filename}")


if __name__ == '__main__':
    logs = run_pipeline()
    plot_results(logs)
    log_episodes_to_csv(logs, "training_log.csv")