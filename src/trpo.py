import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
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
    def __init__(self, input_dim: int, hidden_dim: int = 64):
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

    def _cojugate_gradients(self, Ax, b, iters=10):
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

    def _hessian_vector_product(self, states, old_policy, vector):
        kl = self.policy.get_kl(states, old_policy)

        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * vector).sum()
        grads = torch.autograd.grad(kl_v, self.policy.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])

        return flat_grad_grad_kl

    def _surrogate_loss(self, states, actions, advantages, old_log_probs):
        log_probs = self.policy.log_prob(states, actions)
        ratio = torch.exp(log_probs - old_log_probs)
        return (ratio * advantages).mean()

    def update(self, states, actions, rewards, masks):
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
        step_dir = self._cojugate_gradients(Ax, flat_grad, self.cg_iters)

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
