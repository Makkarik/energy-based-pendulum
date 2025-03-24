"""TRPO implementation in PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class PolicyNetwork(nn.Module):
    """
    Policy network for TRPO.

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

    def get_kl(self, state, other):
        """
        Compute KL divergence between two policies.

        Parameters
        ----------
        state : torch.Tensor
            Input state tensor.
        other : PolicyNetwork
            Another policy network to compare with.

        Returns
        -------
        kl : torch.Tensor
            KL divergence.
        """
        mean1, std1 = self.forward(state)
        mean2, std2 = other.forward(state)

        dist1 = Normal(mean1, std1)
        dist2 = Normal(mean2, std2)

        kl = torch.distributions.kl.kl_divergence(dist1, dist2).sum(dim=-1).mean()
        return kl


class ValueNetwork(nn.Module):
    """
    Value network for TRPO.

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


class TRPO:
    """
    Trust Region Policy Optimization (TRPO) algorithm.

    Parameters
    ----------
    state_dim : int
        Dimension of the state space.
    action_dim : int
        Dimension of the action space.
    hidden_dim : int, optional
        Dimension of the hidden layers, by default 64.
    gamma : float, optional
        Discount factor, by default 0.99.
    tau : float, optional
        GAE (Generalized Advantage Estimation) parameter, by default 0.95.
    delta : float, optional
        KL divergence constraint, by default 0.01.
    damping : float, optional
        Damping factor for conjugate gradient, by default 0.1.
    value_updates : int, optional
        Number of value function updates per iteration, by default 5.
    cg_iters : int, optional
        Number of conjugate gradient iterations, by default 10.
    backtrack_iters : int, optional
        Number of backtracking line search iterations, by default 10.
    backtrack_coeff : float, optional
        Backtracking coefficient, by default 0.8.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        gamma: float = 0.99,
        tau: float = 0.95,
        delta: float = 0.01,
        damping: float = 0.1,
        value_updates: int = 5,
        cg_iters: int = 10,
        backtrack_iters: int = 10,
        backtrack_coeff: float = 0.8,
    ):
        self.gamma = gamma
        self.tau = tau
        self.delta = delta
        self.damping = damping
        self.value_updates = value_updates
        self.cg_iters = cg_iters
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value = ValueNetwork(state_dim, hidden_dim)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=5e-4)

    def _compute_advantages(self, rewards, values, masks):
        """
        Compute advantages using GAE.

        Parameters
        ----------
        rewards : torch.Tensor
            Rewards tensor.
        values : torch.Tensor
            Values tensor.
        masks : torch.Tensor
            Masks tensor.

        Returns
        -------
        advantages : torch.Tensor
            Computed advantages.
        returns : torch.Tensor
            Computed returns.
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

    def _cojugate_gradients(self, Ax, b):
        """
        Solve Ax = b using conjugate gradient method.

        Parameters
        ----------
        Ax : function
            Function to compute the Hessian-vector product.
        b : torch.Tensor
            Right-hand side vector.

        Returns
        -------
        x : torch.Tensor
            Solution vector.
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        r_dot_r = torch.dot(r, r)

        for _ in range(self.cg_iters):
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

    def _hessian_vector_product(self, states, old_policy, vector):
        """
        Compute Hessian-vector product.

        Parameters
        ----------
        states : torch.Tensor
            States tensor.
        old_policy : PolicyNetwork
            Old policy network.
        vector : torch.Tensor
            Vector to multiply with the Hessian.

        Returns
        -------
        hvp : torch.Tensor
            Hessian-vector product.
        """
        kl = self.policy.get_kl(states, old_policy)

        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * vector).sum()
        grads = torch.autograd.grad(kl_v, self.policy.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])

        return flat_grad_grad_kl

    def _surrogate_loss(self, states, actions, advantages, old_log_probs):
        """
        Compute surrogate loss for policy update.

        Parameters
        ----------
        states : torch.Tensor
            States tensor.
        actions : torch.Tensor
            Actions tensor.
        advantages : torch.Tensor
            Advantages tensor.
        old_log_probs : torch.Tensor
            Log probabilities of actions under the old policy.

        Returns
        -------
        loss : torch.Tensor
            Surrogate loss.
        """
        log_probs = self.policy.log_prob(states, actions)
        ratio = torch.exp(log_probs - old_log_probs)
        return (ratio * advantages).mean()

    def update(self, states, actions, rewards, masks):
        """
        Update policy and value networks.

        Parameters
        ----------
        states : np.ndarray
            States array.
        actions : np.ndarray
            Actions array.
        rewards : np.ndarray
            Rewards array.
        masks : np.ndarray
            Masks array.

        Returns
        -------
        dict
            Dictionary containing policy loss, value loss, KL divergence, and entropy.
        """
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        masks = torch.FloatTensor(masks)

        # Update value function
        with torch.no_grad():
            values = self.value(states)
        advantages, returns = self._compute_advantages(rewards, values, masks)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Value loss
        for _ in range(self.value_updates):
            value_loss = F.mse_loss(self.value(states), returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        # Policy gradient
        old_policy = PolicyNetwork(states.shape[1], actions.shape[1])
        old_policy.load_state_dict(self.policy.state_dict())

        with torch.no_grad():
            old_log_probs = old_policy.log_prob(states, actions).detach()

        # Compute policy gradient
        loss = self._surrogate_loss(states, actions, advantages, old_log_probs)
        grads = torch.autograd.grad(loss, self.policy.parameters())
        flat_grad = torch.cat([grad.view(-1) for grad in grads])

        # Compute search direction with conjugate gradient
        Ax = lambda x: self._hessian_vector_product(states, old_policy, x)  # noqa: E731
        step_dir = self._cojugate_gradients(Ax, flat_grad)

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
            new_loss = self._surrogate_loss(
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
