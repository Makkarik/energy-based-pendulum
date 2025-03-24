"""PPO implementation in PyTorch."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset


class PolicyNetwork(nn.Module):
    """
    Policy network for PPO.

    Parameters
    ----------
    input_dim : int
        Dimension of the input state.
    output_dim : int
        Dimension of the output action.
    hidden_dim : int, optional
        Dimension of the hidden layers, by default 64.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input state tensor.

        Returns
        -------
        mean : torch.Tensor
            Mean of the action distribution.
        std : torch.Tensor
            Standard deviation of the action distribution.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std

    def get_action(self, state, deterministic=False):
        """
        Get action from the policy network.

        Parameters
        ----------
        state : torch.Tensor
            Input state tensor.
        deterministic : bool, optional
            Whether to return a deterministic action, by default False.

        Returns
        -------
        action : torch.Tensor
            Sampled action tensor.
        """
        mean, std = self.forward(state)
        if deterministic:
            return mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            return action

    def log_prob(self, state, action):
        """
        Compute log probability of an action.

        Parameters
        ----------
        state : torch.Tensor
            Input state tensor.
        action : torch.Tensor
            Action tensor.

        Returns
        -------
        log_prob : torch.Tensor
            Log probability of the action.
        """
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)

    def entropy(self, state):
        """
        Compute entropy of the action distribution.
        
        Parameters
        ----------
        state : torch.Tensor
            Input state tensor.
        
        Returns
        -------
        entropy : torch.Tensor
            Entropy of the action distribution.
        """
        _, std = self.forward(state)
        return torch.log(std * torch.sqrt(torch.tensor(2 * np.pi * np.e))).sum(dim=-1)


class ValueNetwork(nn.Module):
    """
    Value network for PPO.

    Parameters
    ----------
    input_dim : int
        Dimension of the input state.
    hidden_dim : int, optional
        Dimension of the hidden layers, by default 64.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input state tensor.

        Returns
        -------
        value : torch.Tensor
            State value.
        """
    
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze()


class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm.
    Parameters
    ----------
    state_dim : int
        Dimension of the state space.
    action_dim : int
        Dimension of the action space.
    hidden_dim : int, optional
        Number of hidden units in the neural networks (default is 64).
    lr : float, optional
        Learning rate for the optimizer (default is 3e-4).
    gamma : float, optional
        Discount factor for rewards (default is 0.99).
    tau : float, optional
        GAE (Generalized Advantage Estimation) parameter (default is 0.95).
    clip_param : float, optional
        Clipping parameter for PPO (default is 0.2).
    ppo_epochs : int, optional
        Number of epochs for PPO updates (default is 10).
    mini_batch_size : int, optional
        Mini-batch size for PPO updates (default is 64).
    entropy_coef : float, optional
        Coefficient for entropy bonus (default is 0.01).
    value_coef : float, optional
        Coefficient for value loss (default is 0.5).
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.95,
        clip_param: float = 0.2,
        ppo_epochs: int = 10,
        mini_batch_size: int = 64,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
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

    def _compute_advantages(self, rewards, values, masks):
        """
        Compute advantages and returns using Generalized Advantage Estimation (GAE).

        Parameters
        ----------
        rewards : torch.Tensor
            Tensor of rewards.
        values : torch.Tensor
            Tensor of value estimates.
        masks : torch.Tensor
            Tensor of masks indicating episode boundaries.
        Returns
        -------
        advantages : torch.Tensor
            Tensor of computed advantages.
        returns : torch.Tensor
            Tensor of computed returns.
        """
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

    def update(self, states, actions, rewards, masks):
        """
        Perform a PPO update.

        Parameters
        ----------
        states : array-like
            Array of states.
        actions : array-like
            Array of actions.
        rewards : array-like
            Array of rewards.
        masks : array-like
            Array of masks indicating episode boundaries.
        Returns
        -------
        dict
            Dictionary containing average policy loss, value loss, entropy, and KL 
            divergence (for logging only) over the update.
        """
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        masks = torch.FloatTensor(masks)

        # Compute values and advantages
        with torch.no_grad():
            values = self.value(states)
            advantages, returns = self._compute_advantages(rewards, values, masks)
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
