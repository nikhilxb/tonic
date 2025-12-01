import typing as T

import torch

from tonic.torch import models, updaters


# Type alias for optimizer builder functions.
OptimizerBuilder = T.Callable[[T.List[torch.nn.Parameter]], torch.optim.Optimizer]


class VRegression:
  """Value function regression updater for state-value critics."""

  def __init__(
    self,
    loss: torch.nn.Module | None = None,
    optimizer: OptimizerBuilder | None = None,
    gradient_clip: float = 0,
  ):
    """Initialize the value function regression updater.
    
    Args:
      loss: Loss function for regression. Defaults to MSE loss.
      optimizer: Function that takes parameters and returns an optimizer.
        Defaults to Adam with lr=1e-3.
      gradient_clip: Maximum gradient norm for clipping. 0 disables clipping.
    """
    self.loss = loss or torch.nn.MSELoss()
    self.optimizer_builder = optimizer or (lambda params: torch.optim.Adam(params, lr=1e-3))
    self.gradient_clip = gradient_clip

  def initialize(self, model: torch.nn.Module) -> None:
    """Initialize the updater with the model.
    
    Args:
      model: Model containing the critic network.
    """
    self.model = model
    self.variables = models.trainable_variables(self.model.critic)
    self.optimizer = self.optimizer_builder(self.variables)

  def __call__(
    self,
    observations: torch.Tensor,
    returns: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Perform a value function regression update.
    
    Args:
      observations: State observations, shape [batch_size, obs_dim].
      returns: Target return values, shape [batch_size].
      
    Returns:
      Dictionary containing loss and predicted values.
    """
    self.optimizer.zero_grad()
    values = self.model.critic(observations)  # [batch_size]
    loss = self.loss(values, returns)

    loss.backward()
    if self.gradient_clip > 0:
      torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
    self.optimizer.step()

    return dict(loss=loss.detach(), v=values.detach())


class QRegression:
  """Action-value function regression updater for Q-critics."""

  def __init__(
    self,
    loss: torch.nn.Module | None = None,
    optimizer: OptimizerBuilder | None = None,
    gradient_clip: float = 0,
  ):
    """Initialize the Q-function regression updater.
    
    Args:
      loss: Loss function for regression. Defaults to MSE loss.
      optimizer: Function that takes parameters and returns an optimizer.
        Defaults to Adam with lr=1e-3.
      gradient_clip: Maximum gradient norm for clipping. 0 disables clipping.
    """
    self.loss = loss or torch.nn.MSELoss()
    self.optimizer_builder = optimizer or (lambda params: torch.optim.Adam(params, lr=1e-3))
    self.gradient_clip = gradient_clip

  def initialize(self, model: torch.nn.Module) -> None:
    """Initialize the updater with the model.
    
    Args:
      model: Model containing the critic network.
    """
    self.model = model
    self.variables = models.trainable_variables(self.model.critic)
    self.optimizer = self.optimizer_builder(self.variables)

  def __call__(
    self,
    observations: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Perform a Q-function regression update.
    
    Args:
      observations: State observations, shape [batch_size, obs_dim].
      actions: Actions taken, shape [batch_size, action_dim].
      returns: Target Q-values, shape [batch_size].
      
    Returns:
      Dictionary containing loss and predicted Q-values.
    """
    self.optimizer.zero_grad()
    values = self.model.critic(observations, actions)  # [batch_size]
    loss = self.loss(values, returns)

    loss.backward()
    if self.gradient_clip > 0:
      torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
    self.optimizer.step()

    return dict(loss=loss.detach(), q=values.detach())


class DeterministicQLearning:
  """Q-learning updater for deterministic policies (e.g., DDPG)."""

  def __init__(
    self,
    loss: torch.nn.Module | None = None,
    optimizer: OptimizerBuilder | None = None,
    gradient_clip: float = 0,
  ):
    """Initialize the deterministic Q-learning updater.
    
    Args:
      loss: Loss function for Q-value regression. Defaults to MSE loss.
      optimizer: Function that takes parameters and returns an optimizer.
        Defaults to Adam with lr=1e-3.
      gradient_clip: Maximum gradient norm for clipping. 0 disables clipping.
    """
    self.loss = loss or torch.nn.MSELoss()
    self.optimizer_builder = optimizer or (lambda params: torch.optim.Adam(params, lr=1e-3))
    self.gradient_clip = gradient_clip

  def initialize(self, model: torch.nn.Module) -> None:
    """Initialize the updater with the model.
    
    Args:
      model: Model containing critic, target_actor, and target_critic.
    """
    self.model = model
    self.variables = models.trainable_variables(self.model.critic)
    self.optimizer = self.optimizer_builder(self.variables)

  def __call__(
    self,
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_observations: torch.Tensor,
    rewards: torch.Tensor,
    discounts: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Perform a Q-learning update using target networks.
    
    Computes targets as r + gamma * Q_target(s', mu_target(s')).
    
    Args:
      observations: Current state observations, shape [batch_size, obs_dim].
      actions: Actions taken, shape [batch_size, action_dim].
      next_observations: Next state observations, shape [batch_size, obs_dim].
      rewards: Rewards received, shape [batch_size].
      discounts: Discount factors (gamma * (1 - done)), shape [batch_size].
      
    Returns:
      Dictionary containing loss and predicted Q-values.
    """
    with torch.no_grad():
      next_actions = self.model.target_actor(next_observations)  # [batch_size, action_dim]
      next_values = self.model.target_critic(next_observations, next_actions)  # [batch_size]
      returns = rewards + discounts * next_values  # [batch_size]

    self.optimizer.zero_grad()
    values = self.model.critic(observations, actions)  # [batch_size]
    loss = self.loss(values, returns)

    loss.backward()
    if self.gradient_clip > 0:
      torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
    self.optimizer.step()

    return dict(loss=loss.detach(), q=values.detach())


class DistributionalDeterministicQLearning:
  """Distributional Q-learning updater (e.g., D4PG)."""

  def __init__(
    self,
    optimizer: OptimizerBuilder | None = None,
    gradient_clip: float = 0,
  ):
    """Initialize the distributional Q-learning updater.
    
    Args:
      optimizer: Function that takes parameters and returns an optimizer.
        Defaults to Adam with lr=1e-3.
      gradient_clip: Maximum gradient norm for clipping. 0 disables clipping.
    """
    self.optimizer_builder = optimizer or (lambda params: torch.optim.Adam(params, lr=1e-3))
    self.gradient_clip = gradient_clip

  def initialize(self, model: torch.nn.Module) -> None:
    """Initialize the updater with the model.
    
    Args:
      model: Model containing distributional critic and target networks.
    """
    self.model = model
    self.variables = models.trainable_variables(self.model.critic)
    self.optimizer = self.optimizer_builder(self.variables)

  def __call__(
    self,
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_observations: torch.Tensor,
    rewards: torch.Tensor,
    discounts: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Perform a distributional Q-learning update.
    
    Uses categorical projection to update the value distribution.
    
    Args:
      observations: Current state observations, shape [batch_size, obs_dim].
      actions: Actions taken, shape [batch_size, action_dim].
      next_observations: Next state observations, shape [batch_size, obs_dim].
      rewards: Rewards received, shape [batch_size].
      discounts: Discount factors, shape [batch_size].
      
    Returns:
      Dictionary containing the cross-entropy loss.
    """
    with torch.no_grad():
      next_actions = self.model.target_actor(next_observations)  # [batch_size, action_dim]
      next_value_distributions = self.model.target_critic(next_observations, next_actions)
      values = next_value_distributions.values  # [batch_size, num_atoms]
      returns = rewards[:, None] + discounts[:, None] * values  # [batch_size, num_atoms]
      targets = next_value_distributions.project(returns)  # [batch_size, num_atoms]

    self.optimizer.zero_grad()
    value_distributions = self.model.critic(observations, actions)
    log_probabilities = torch.nn.functional.log_softmax(
      value_distributions.logits, dim=-1
    )  # [batch_size, num_atoms]
    # Cross-entropy loss between projected target and predicted distribution.
    loss = -(targets * log_probabilities).sum(dim=-1).mean()

    loss.backward()
    if self.gradient_clip > 0:
      torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
    self.optimizer.step()

    return dict(loss=loss.detach())


class TargetActionNoise:
  """Action noise for target policy smoothing (used in TD3)."""

  def __init__(
    self,
    scale: float = 0.2,
    clip: float = 0.5,
  ):
    """Initialize target action noise.
    
    Args:
      scale: Standard deviation of Gaussian noise.
      clip: Maximum absolute noise value (clipped to [-clip, clip]).
    """
    self.scale = scale
    self.clip = clip

  def __call__(self, actions: torch.Tensor) -> torch.Tensor:
    """Add clipped Gaussian noise to actions and clip to [-1, 1].
    
    Args:
      actions: Actions to add noise to, shape [batch_size, action_dim].
      
    Returns:
      Noisy actions clipped to [-1, 1], shape [batch_size, action_dim].
    """
    noises = self.scale * torch.randn_like(actions)  # [batch_size, action_dim]
    noises = torch.clamp(noises, -self.clip, self.clip)
    actions = actions + noises
    return torch.clamp(actions, -1, 1)


class TwinCriticDeterministicQLearning:
  """Twin critic Q-learning updater (e.g., TD3)."""

  def __init__(
    self,
    loss: torch.nn.Module | None = None,
    optimizer: OptimizerBuilder | None = None,
    target_action_noise: TargetActionNoise | None = None,
    gradient_clip: float = 0,
  ):
    """Initialize the twin critic Q-learning updater.
    
    Args:
      loss: Loss function for Q-value regression. Defaults to MSE loss.
      optimizer: Function that takes parameters and returns an optimizer.
        Defaults to Adam with lr=1e-3.
      target_action_noise: Noise to add to target actions for smoothing.
      gradient_clip: Maximum gradient norm for clipping. 0 disables clipping.
    """
    self.loss = loss or torch.nn.MSELoss()
    self.optimizer_builder = optimizer or (lambda params: torch.optim.Adam(params, lr=1e-3))
    self.target_action_noise = target_action_noise or TargetActionNoise(scale=0.2, clip=0.5)
    self.gradient_clip = gradient_clip

  def initialize(self, model: torch.nn.Module) -> None:
    """Initialize the updater with the model.
    
    Args:
      model: Model containing twin critics and target networks.
    """
    self.model = model
    variables_1 = models.trainable_variables(self.model.critic_1)
    variables_2 = models.trainable_variables(self.model.critic_2)
    self.variables = variables_1 + variables_2
    self.optimizer = self.optimizer_builder(self.variables)

  def __call__(
    self,
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_observations: torch.Tensor,
    rewards: torch.Tensor,
    discounts: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Perform a twin critic Q-learning update.
    
    Uses minimum of twin Q-values for target computation to reduce overestimation.
    
    Args:
      observations: Current state observations, shape [batch_size, obs_dim].
      actions: Actions taken, shape [batch_size, action_dim].
      next_observations: Next state observations, shape [batch_size, obs_dim].
      rewards: Rewards received, shape [batch_size].
      discounts: Discount factors, shape [batch_size].
      
    Returns:
      Dictionary containing loss and both Q-values.
    """
    with torch.no_grad():
      next_actions = self.model.target_actor(next_observations)  # [batch_size, action_dim]
      next_actions = self.target_action_noise(next_actions)  # Add smoothing noise
      next_values_1 = self.model.target_critic_1(next_observations, next_actions)  # [batch_size]
      next_values_2 = self.model.target_critic_2(next_observations, next_actions)  # [batch_size]
      # Use minimum to reduce overestimation bias.
      next_values = torch.min(next_values_1, next_values_2)  # [batch_size]
      returns = rewards + discounts * next_values  # [batch_size]

    self.optimizer.zero_grad()
    values_1 = self.model.critic_1(observations, actions)  # [batch_size]
    values_2 = self.model.critic_2(observations, actions)  # [batch_size]
    loss_1 = self.loss(values_1, returns)
    loss_2 = self.loss(values_2, returns)
    loss = loss_1 + loss_2

    loss.backward()
    if self.gradient_clip > 0:
      torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
    self.optimizer.step()

    return dict(loss=loss.detach(), q1=values_1.detach(), q2=values_2.detach())


class TwinCriticSoftQLearning:
  """Twin critic soft Q-learning updater for SAC."""

  def __init__(
    self,
    loss: torch.nn.Module | None = None,
    optimizer: OptimizerBuilder | None = None,
    entropy_coeff: float = 0.2,
    gradient_clip: float = 0,
  ):
    """Initialize the twin critic soft Q-learning updater.
    
    Args:
      loss: Loss function for Q-value regression. Defaults to MSE loss.
      optimizer: Function that takes parameters and returns an optimizer.
        Defaults to Adam with lr=3e-4.
      entropy_coeff: Temperature parameter for entropy regularization.
      gradient_clip: Maximum gradient norm for clipping. 0 disables clipping.
    """
    self.loss = loss or torch.nn.MSELoss()
    self.optimizer_builder = optimizer or (lambda params: torch.optim.Adam(params, lr=3e-4))
    self.entropy_coeff = entropy_coeff
    self.gradient_clip = gradient_clip

  def initialize(self, model: torch.nn.Module) -> None:
    """Initialize the updater with the model.
    
    Args:
      model: Model containing twin critics, actor, and target critics.
    """
    self.model = model
    variables_1 = models.trainable_variables(self.model.critic_1)
    variables_2 = models.trainable_variables(self.model.critic_2)
    self.variables = variables_1 + variables_2
    self.optimizer = self.optimizer_builder(self.variables)

  def __call__(
    self,
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_observations: torch.Tensor,
    rewards: torch.Tensor,
    discounts: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Perform a soft Q-learning update.
    
    Computes targets as r + gamma * (min(Q) - alpha * log(pi)).
    
    Args:
      observations: Current state observations, shape [batch_size, obs_dim].
      actions: Actions taken, shape [batch_size, action_dim].
      next_observations: Next state observations, shape [batch_size, obs_dim].
      rewards: Rewards received, shape [batch_size].
      discounts: Discount factors, shape [batch_size].
      
    Returns:
      Dictionary containing loss and both Q-values.
    """
    with torch.no_grad():
      next_distributions = self.model.actor(next_observations)
      # Use reparameterization trick for sampling.
      if hasattr(next_distributions, 'rsample_with_log_prob'):
        outs = next_distributions.rsample_with_log_prob()
        next_actions, next_log_probs = outs
      else:
        next_actions = next_distributions.rsample()  # [batch_size, action_dim]
        next_log_probs = next_distributions.log_prob(next_actions)
      next_log_probs = next_log_probs.sum(dim=-1)  # [batch_size]
      next_values_1 = self.model.target_critic_1(next_observations, next_actions)  # [batch_size]
      next_values_2 = self.model.target_critic_2(next_observations, next_actions)  # [batch_size]
      next_values = torch.min(next_values_1, next_values_2)  # [batch_size]
      # SAC target: r + gamma * (Q - alpha * log(pi)).
      returns = rewards + discounts * (next_values - self.entropy_coeff * next_log_probs)

    self.optimizer.zero_grad()
    values_1 = self.model.critic_1(observations, actions)  # [batch_size]
    values_2 = self.model.critic_2(observations, actions)  # [batch_size]
    loss_1 = self.loss(values_1, returns)
    loss_2 = self.loss(values_2, returns)
    loss = loss_1 + loss_2

    loss.backward()
    if self.gradient_clip > 0:
      torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
    self.optimizer.step()

    return dict(loss=loss.detach(), q1=values_1.detach(), q2=values_2.detach())


class ExpectedSARSA:
  """Expected SARSA updater using Monte Carlo sampling."""

  def __init__(
    self,
    num_samples: int = 20,
    loss: torch.nn.Module | None = None,
    optimizer: OptimizerBuilder | None = None,
    gradient_clip: float = 0,
  ):
    """Initialize the Expected SARSA updater.
    
    Args:
      num_samples: Number of action samples to approximate expectation.
      loss: Loss function for Q-value regression. Defaults to MSE loss.
      optimizer: Function that takes parameters and returns an optimizer.
        Defaults to Adam with lr=3e-4.
      gradient_clip: Maximum gradient norm for clipping. 0 disables clipping.
    """
    self.num_samples = num_samples
    self.loss = loss or torch.nn.MSELoss()
    self.optimizer_builder = optimizer or (lambda params: torch.optim.Adam(params, lr=3e-4))
    self.gradient_clip = gradient_clip

  def initialize(self, model: torch.nn.Module) -> None:
    """Initialize the updater with the model.
    
    Args:
      model: Model containing critic, target_actor, and target_critic.
    """
    self.model = model
    self.variables = models.trainable_variables(self.model.critic)
    self.optimizer = self.optimizer_builder(self.variables)

  def __call__(
    self,
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_observations: torch.Tensor,
    rewards: torch.Tensor,
    discounts: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Perform an Expected SARSA update.
    
    Approximates E[Q(s', a')] by averaging over sampled actions.
    
    Args:
      observations: Current state observations, shape [batch_size, obs_dim].
      actions: Actions taken, shape [batch_size, action_dim].
      next_observations: Next state observations, shape [batch_size, obs_dim].
      rewards: Rewards received, shape [batch_size].
      discounts: Discount factors, shape [batch_size].
      
    Returns:
      Dictionary containing loss and predicted Q-values.
    """
    # Approximate the expected next values using Monte Carlo sampling.
    with torch.no_grad():
      next_target_distributions = self.model.target_actor(next_observations)
      next_actions = next_target_distributions.rsample(
        (self.num_samples,)
      )  # [num_samples, batch, action_dim]
      next_actions = updaters.merge_first_two_dims(next_actions)  # [num_samples * batch, act_dim]
      next_observations = updaters.tile(
        next_observations, self.num_samples
      )  # [num_samples, batch, obs_dim]
      next_observations = updaters.merge_first_two_dims(
        next_observations
      )  # [num_samples*batch, obs_dim]
      next_values = self.model.target_critic(
        next_observations, next_actions
      )  # [num_samples * batch]
      next_values = next_values.view(self.num_samples, -1)  # [num_samples, batch]
      next_values = next_values.mean(dim=0)  # [batch_size]
      returns = rewards + discounts * next_values  # [batch_size]

    self.optimizer.zero_grad()
    values = self.model.critic(observations, actions)  # [batch_size]
    loss = self.loss(returns, values)

    loss.backward()
    if self.gradient_clip > 0:
      torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
    self.optimizer.step()

    return dict(loss=loss.detach(), q=values.detach())
