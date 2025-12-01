import typing as T

import torch

from tonic.torch import models, updaters


FLOAT_EPSILON = 1e-8


# Type alias for optimizer builder functions.
OptimizerBuilder = T.Callable[[T.List[torch.nn.Parameter]], torch.optim.Optimizer]


class StochasticPolicyGradient:
  """Vanilla policy gradient updater for stochastic policies (e.g., REINFORCE, A2C)."""

  def __init__(
    self,
    optimizer: OptimizerBuilder | None = None,
    entropy_coeff: float = 0,
    gradient_clip: float = 0,
  ):
    """Initialize the stochastic policy gradient updater.
    
    Args:
      optimizer: Function that takes parameters and returns an optimizer.
        Defaults to Adam with lr=3e-4.
      entropy_coeff: Coefficient for entropy regularization (encourages exploration).
      gradient_clip: Maximum gradient norm for clipping. 0 disables clipping.
    """
    self.optimizer_builder = optimizer or (lambda params: torch.optim.Adam(params, lr=3e-4))
    self.entropy_coeff = entropy_coeff
    self.gradient_clip = gradient_clip

  def initialize(self, model: torch.nn.Module) -> None:
    """Initialize the updater with the model.
    
    Args:
      model: Model containing the actor network.
    """
    self.model = model
    self.variables = models.trainable_variables(self.model.actor)
    self.optimizer = self.optimizer_builder(self.variables)

  def __call__(
    self,
    observations: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    log_probs: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Perform a policy gradient update.
    
    Args:
      observations: State observations, shape [batch_size, obs_dim].
      actions: Actions taken, shape [batch_size, action_dim].
      advantages: Advantage estimates, shape [batch_size].
      log_probs: Log probabilities of actions under old policy, shape [batch_size].
      
    Returns:
      Dictionary containing loss, KL divergence, entropy, and policy std.
    """
    # Skip update if all advantages are zero.
    if (advantages == 0.).all():
      loss = torch.as_tensor(0., dtype=torch.float32)
      kl = torch.as_tensor(0., dtype=torch.float32)
      with torch.no_grad():
        distributions = self.model.actor(observations)
        entropy = distributions.entropy().mean()
        std = distributions.stddev.mean()

    else:
      self.optimizer.zero_grad()
      distributions = self.model.actor(observations)  # [batch_size, action_dim]
      new_log_probs = distributions.log_prob(actions).sum(dim=-1)  # [batch_size]
      # Policy gradient loss: -E[A * log(pi(a|s))].
      loss = -(advantages * new_log_probs).mean()
      entropy = distributions.entropy().mean()
      if self.entropy_coeff != 0:
        loss -= self.entropy_coeff * entropy

      loss.backward()
      if self.gradient_clip > 0:
        torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
      self.optimizer.step()

      loss = loss.detach()
      kl = (log_probs - new_log_probs).mean().detach()
      entropy = entropy.detach()
      std = distributions.stddev.mean().detach()

    return dict(loss=loss, kl=kl, entropy=entropy, std=std)


class ClippedRatio:
  """PPO (Proximal Policy Optimization) updater with clipped surrogate objective."""

  def __init__(
    self,
    optimizer: OptimizerBuilder | None = None,
    ratio_clip: float = 0.2,
    kl_threshold: float = 0.015,
    entropy_coeff: float = 0,
    gradient_clip: float = 0,
  ):
    """Initialize the clipped ratio (PPO) updater.
    
    Args:
      optimizer: Function that takes parameters and returns an optimizer.
        Defaults to Adam with lr=3e-4.
      ratio_clip: Clipping range for probability ratios [1-clip, 1+clip].
      kl_threshold: KL divergence threshold for early stopping.
      entropy_coeff: Coefficient for entropy regularization (encourages exploration).
      gradient_clip: Maximum gradient norm for clipping. 0 disables clipping.
    """
    self.optimizer_builder = optimizer or (lambda params: torch.optim.Adam(params, lr=3e-4))
    self.ratio_clip = ratio_clip
    self.kl_threshold = kl_threshold
    self.entropy_coeff = entropy_coeff
    self.gradient_clip = gradient_clip

  def initialize(self, model: torch.nn.Module) -> None:
    """Initialize the updater with the model.
    
    Args:
      model: Model containing the actor network.
    """
    self.model = model
    self.variables = models.trainable_variables(self.model.actor)
    self.optimizer = self.optimizer_builder(self.variables)

  def __call__(
    self,
    observations: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    log_probs: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Perform a PPO update with clipped surrogate objective.
    
    Args:
      observations: State observations, shape [batch_size, obs_dim].
      actions: Actions taken, shape [batch_size, action_dim].
      advantages: Advantage estimates, shape [batch_size].
      log_probs: Log probabilities of actions under old policy, shape [batch_size].
      
    Returns:
      Dictionary containing loss, KL divergence, entropy, clip fraction, std, and stop flag.
    """
    # Skip update if all advantages are zero.
    if (advantages == 0.).all():
      loss = torch.as_tensor(0., dtype=torch.float32)
      kl = torch.as_tensor(0., dtype=torch.float32)
      clip_fraction = torch.as_tensor(0., dtype=torch.float32)
      with torch.no_grad():
        distributions = self.model.actor(observations)
        entropy = distributions.entropy().mean()
        std = distributions.stddev.mean()

    else:
      self.optimizer.zero_grad()
      distributions = self.model.actor(observations)  # [batch_size, action_dim]
      new_log_probs = distributions.log_prob(actions).sum(dim=-1)  # [batch_size]
      ratios_1 = torch.exp(new_log_probs - log_probs)  # [batch_size]
      surrogates_1 = advantages * ratios_1  # [batch_size]
      ratio_low = 1 - self.ratio_clip
      ratio_high = 1 + self.ratio_clip
      ratios_2 = torch.clamp(ratios_1, ratio_low, ratio_high)  # [batch_size]
      surrogates_2 = advantages * ratios_2  # [batch_size]
      # PPO clipped surrogate loss: -E[min(r*A, clip(r)*A)].
      loss = -(torch.min(surrogates_1, surrogates_2)).mean()
      entropy = distributions.entropy().mean()
      if self.entropy_coeff != 0:
        loss -= self.entropy_coeff * entropy

      loss.backward()
      if self.gradient_clip > 0:
        torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
      self.optimizer.step()

      loss = loss.detach()
      with torch.no_grad():
        kl = (log_probs - new_log_probs).mean()
      entropy = entropy.detach()
      clipped = ratios_1.gt(ratio_high) | ratios_1.lt(ratio_low)
      clip_fraction = torch.as_tensor(clipped, dtype=torch.float32).mean()
      std = distributions.stddev.mean().detach()

    return dict(
      loss=loss,
      kl=kl,
      entropy=entropy,
      clip_fraction=clip_fraction,
      std=std,
      stop=kl > self.kl_threshold,
    )


class TrustRegionPolicyGradient:
  """TRPO (Trust Region Policy Optimization) updater using natural gradients."""

  def __init__(
    self,
    optimizer: T.Any | None = None,
    entropy_coeff: float = 0,
  ):
    """Initialize the TRPO updater.
    
    Args:
      optimizer: Conjugate gradient optimizer. Defaults to ConjugateGradient().
      entropy_coeff: Coefficient for entropy regularization (encourages exploration).
    """
    self.optimizer = optimizer or updaters.ConjugateGradient()
    self.entropy_coeff = entropy_coeff

  def initialize(self, model: torch.nn.Module) -> None:
    """Initialize the updater with the model.
    
    Args:
      model: Model containing the actor network.
    """
    self.model = model
    self.variables = models.trainable_variables(self.model.actor)

  def __call__(
    self,
    observations: torch.Tensor,
    actions: torch.Tensor,
    log_probs: torch.Tensor,
    locs: torch.Tensor,
    scales: torch.Tensor,
    advantages: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Perform a TRPO update using natural gradients.
    
    Args:
      observations: State observations, shape [batch_size, obs_dim].
      actions: Actions taken, shape [batch_size, action_dim].
      log_probs: Log probabilities of actions under old policy, shape [batch_size].
      locs: Old policy distribution means, shape [batch_size, action_dim].
      scales: Old policy distribution stds, shape [batch_size, action_dim].
      advantages: Advantage estimates, shape [batch_size].
      
    Returns:
      Dictionary containing loss, KL divergence, and backtracking steps.
    """
    # Skip update if all advantages are zero.
    if (advantages == 0.).all():
      kl = torch.as_tensor(0., dtype=torch.float32)
      loss = torch.as_tensor(0., dtype=torch.float32)
      steps = torch.as_tensor(0, dtype=torch.int32)

    else:
      kl, loss, steps = self.optimizer.optimize(
        loss_function=lambda: self._loss(observations, actions, log_probs, advantages),
        constraint_function=lambda: self._kl(observations, locs, scales),
        variables=self.variables,
      )

    return dict(loss=loss, kl=kl, backtrack_steps=steps)

  def _loss(
    self,
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
  ) -> torch.Tensor:
    """Compute the surrogate policy loss.
    
    Args:
      observations: State observations, shape [batch_size, obs_dim].
      actions: Actions taken, shape [batch_size, action_dim].
      old_log_probs: Log probabilities under old policy, shape [batch_size].
      advantages: Advantage estimates, shape [batch_size].
      
    Returns:
      Scalar loss value.
    """
    distributions = self.model.actor(observations)
    log_probs = distributions.log_prob(actions).sum(dim=-1)  # [batch_size]
    ratios = torch.exp(log_probs - old_log_probs)  # [batch_size]
    loss = -(ratios * advantages).mean()
    if self.entropy_coeff != 0:
      entropy = distributions.entropy().mean()
      loss -= self.entropy_coeff * entropy
    return loss

  def _kl(
    self,
    observations: torch.Tensor,
    locs: torch.Tensor,
    scales: torch.Tensor,
  ) -> torch.Tensor:
    """Compute KL divergence between old and new policies.
    
    Args:
      observations: State observations, shape [batch_size, obs_dim].
      locs: Old policy distribution means, shape [batch_size, action_dim].
      scales: Old policy distribution stds, shape [batch_size, action_dim].
      
    Returns:
      Mean KL divergence.
    """
    distributions = self.model.actor(observations)
    old_distributions = type(distributions)(locs, scales)
    return torch.distributions.kl.kl_divergence(distributions, old_distributions).mean()


class DeterministicPolicyGradient:
  """Deterministic policy gradient updater for continuous control (e.g., DDPG)."""

  def __init__(
    self,
    optimizer: OptimizerBuilder | None = None,
    gradient_clip: float = 0,
  ):
    """Initialize the deterministic policy gradient updater.
    
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
      model: Model containing actor and critic networks.
    """
    self.model = model
    self.variables = models.trainable_variables(self.model.actor)
    self.optimizer = self.optimizer_builder(self.variables)

  def __call__(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
    """Perform a deterministic policy gradient update.
    
    Maximizes the Q-value of actions selected by the current policy.
    
    Args:
      observations: State observations, shape [batch_size, obs_dim].
      
    Returns:
      Dictionary containing the actor loss.
    """
    critic_variables = models.trainable_variables(self.model.critic)

    # Freeze critic during actor update.
    for var in critic_variables:
      var.requires_grad = False

    self.optimizer.zero_grad()
    actions = self.model.actor(observations)  # [batch_size, action_dim]
    values = self.model.critic(observations, actions)  # [batch_size]
    # Maximize Q(s, mu(s)) => minimize -Q(s, mu(s)).
    loss = -values.mean()

    loss.backward()
    if self.gradient_clip > 0:
      torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
    self.optimizer.step()

    # Unfreeze critic after actor update.
    for var in critic_variables:
      var.requires_grad = True

    return dict(loss=loss.detach())


class DistributionalDeterministicPolicyGradient:
  """Deterministic policy gradient with distributional critic (e.g., D4PG)."""

  def __init__(
    self,
    optimizer: OptimizerBuilder | None = None,
    gradient_clip: float = 0,
  ):
    """Initialize the distributional deterministic policy gradient updater.
    
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
      model: Model containing actor and distributional critic networks.
    """
    self.model = model
    self.variables = models.trainable_variables(self.model.actor)
    self.optimizer = self.optimizer_builder(self.variables)

  def __call__(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
    """Perform a distributional deterministic policy gradient update.
    
    Maximizes the expected value from the critic's value distribution.
    
    Args:
      observations: State observations, shape [batch_size, obs_dim].
      
    Returns:
      Dictionary containing the actor loss.
    """
    critic_variables = models.trainable_variables(self.model.critic)

    # Freeze critic during actor update.
    for var in critic_variables:
      var.requires_grad = False

    self.optimizer.zero_grad()
    actions = self.model.actor(observations)  # [batch_size, action_dim]
    value_distributions = self.model.critic(observations, actions)
    values = value_distributions.mean()  # [batch_size]
    # Maximize E[Z(s, mu(s))] => minimize -E[Z(s, mu(s))].
    loss = -values.mean()

    loss.backward()
    if self.gradient_clip > 0:
      torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
    self.optimizer.step()

    # Unfreeze critic after actor update.
    for var in critic_variables:
      var.requires_grad = True

    return dict(loss=loss.detach())


class TwinCriticSoftDeterministicPolicyGradient:
  """Soft actor-critic policy updater with twin critics (SAC)."""

  def __init__(
    self,
    optimizer: OptimizerBuilder | None = None,
    entropy_coeff: float = 0.2,
    gradient_clip: float = 0,
  ):
    """Initialize the SAC policy updater.
    
    Args:
      optimizer: Function that takes parameters and returns an optimizer.
        Defaults to Adam with lr=3e-4.
      entropy_coeff: Temperature parameter for entropy regularization.
      gradient_clip: Maximum gradient norm for clipping. 0 disables clipping.
    """
    self.optimizer_builder = optimizer or (lambda params: torch.optim.Adam(params, lr=3e-4))
    self.entropy_coeff = entropy_coeff
    self.gradient_clip = gradient_clip

  def initialize(self, model: torch.nn.Module) -> None:
    """Initialize the updater with the model.
    
    Args:
      model: Model containing actor and twin critic networks.
    """
    self.model = model
    self.variables = models.trainable_variables(self.model.actor)
    self.optimizer = self.optimizer_builder(self.variables)

  def __call__(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
    """Perform a soft actor-critic policy update.
    
    Maximizes Q(s,a) - alpha * log(pi(a|s)) using reparameterization trick.
    
    Args:
      observations: State observations, shape [batch_size, obs_dim].
      
    Returns:
      Dictionary containing the actor loss.
    """
    critic_1_variables = models.trainable_variables(self.model.critic_1)
    critic_2_variables = models.trainable_variables(self.model.critic_2)
    critic_variables = critic_1_variables + critic_2_variables

    # Freeze critics during actor update.
    for var in critic_variables:
      var.requires_grad = False

    self.optimizer.zero_grad()
    distributions = self.model.actor(observations)  # [batch_size, action_dim]
    # Use reparameterization trick for differentiable sampling.
    if hasattr(distributions, 'rsample_with_log_prob'):
      actions, log_probs = distributions.rsample_with_log_prob()
    else:
      actions = distributions.rsample()  # [batch_size, action_dim]
      log_probs = distributions.log_prob(actions)
    log_probs = log_probs.sum(dim=-1)  # [batch_size]
    values_1 = self.model.critic_1(observations, actions)  # [batch_size]
    values_2 = self.model.critic_2(observations, actions)  # [batch_size]
    values = torch.min(values_1, values_2)  # [batch_size]
    # SAC objective: maximize Q - alpha*log(pi).
    loss = (self.entropy_coeff * log_probs - values).mean()

    loss.backward()
    if self.gradient_clip > 0:
      torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
    self.optimizer.step()

    # Unfreeze critics after actor update.
    for var in critic_variables:
      var.requires_grad = True

    return dict(loss=loss.detach())


class MaximumAPosterioriPolicyOptimization:
  """MPO (Maximum a Posteriori Policy Optimization) updater."""

  def __init__(
    self,
    num_samples: int = 20,
    epsilon: float = 1e-1,
    epsilon_penalty: float = 1e-3,
    epsilon_mean: float = 1e-3,
    epsilon_std: float = 1e-6,
    initial_log_temperature: float = 1.0,
    initial_log_alpha_mean: float = 1.0,
    initial_log_alpha_std: float = 10.0,
    min_log_dual: float = -18.0,
    per_dim_constraining: bool = True,
    action_penalization: bool = True,
    actor_optimizer: OptimizerBuilder | None = None,
    dual_optimizer: OptimizerBuilder | None = None,
    gradient_clip: float = 0,
  ):
    """Initialize the MPO updater.
    
    Args:
      num_samples: Number of action samples for Q-value estimation.
      epsilon: KL constraint for temperature optimization.
      epsilon_penalty: KL constraint for action bound penalty.
      epsilon_mean: KL constraint for policy mean.
      epsilon_std: KL constraint for policy std.
      initial_log_temperature: Initial log temperature for E-step.
      initial_log_alpha_mean: Initial log Lagrange multiplier for mean constraint.
      initial_log_alpha_std: Initial log Lagrange multiplier for std constraint.
      min_log_dual: Minimum value for log dual variables (prevents numerical issues).
      per_dim_constraining: If True, apply KL constraints per action dimension.
      action_penalization: If True, penalize actions outside [-1, 1].
      actor_optimizer: Optimizer builder for actor parameters.
      dual_optimizer: Optimizer builder for dual variables.
      gradient_clip: Maximum gradient norm for clipping. 0 disables clipping.
    """
    self.num_samples = num_samples
    self.epsilon = epsilon
    self.epsilon_mean = epsilon_mean
    self.epsilon_std = epsilon_std
    self.initial_log_temperature = initial_log_temperature
    self.initial_log_alpha_mean = initial_log_alpha_mean
    self.initial_log_alpha_std = initial_log_alpha_std
    self.min_log_dual = torch.as_tensor(min_log_dual, dtype=torch.float32)
    self.action_penalization = action_penalization
    self.epsilon_penalty = epsilon_penalty
    self.per_dim_constraining = per_dim_constraining
    self.actor_optimizer_builder = actor_optimizer or (
      lambda params: torch.optim.Adam(params, lr=3e-4)
    )
    self.dual_optimizer_builder = dual_optimizer or (
      lambda params: torch.optim.Adam(params, lr=1e-2)
    )
    self.gradient_clip = gradient_clip

  def initialize(self, model: torch.nn.Module, action_space: T.Any) -> None:
    """Initialize the updater with model and action space.
    
    Args:
      model: Model containing actor, target_actor, and target_critic.
      action_space: Gym action space (used to determine action dimensionality).
    """
    self.model = model
    self.actor_variables = models.trainable_variables(self.model.actor)
    self.actor_optimizer = self.actor_optimizer_builder(self.actor_variables)

    # Initialize dual variables (Lagrange multipliers).
    self.dual_variables: list[torch.nn.Parameter] = []
    self.log_temperature = torch.nn.Parameter(
      torch.as_tensor([self.initial_log_temperature], dtype=torch.float32)
    )
    self.dual_variables.append(self.log_temperature)
    shape = [action_space.shape[0]] if self.per_dim_constraining else [1]
    self.log_alpha_mean = torch.nn.Parameter(
      torch.full(shape, self.initial_log_alpha_mean, dtype=torch.float32)
    )
    self.dual_variables.append(self.log_alpha_mean)
    self.log_alpha_std = torch.nn.Parameter(
      torch.full(shape, self.initial_log_alpha_std, dtype=torch.float32)
    )
    self.dual_variables.append(self.log_alpha_std)
    if self.action_penalization:
      self.log_penalty_temperature = torch.nn.Parameter(
        torch.as_tensor([self.initial_log_temperature], dtype=torch.float32)
      )
      self.dual_variables.append(self.log_penalty_temperature)
    self.dual_optimizer = self.dual_optimizer_builder(self.dual_variables)

  def __call__(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
    """Perform an MPO update using EM-style optimization.
    
    Args:
      observations: State observations, shape [batch_size, obs_dim].
      
    Returns:
      Dictionary containing policy losses, KL losses, dual losses, and dual variables.
    """

    def parametric_kl_and_dual_losses(
      kl: torch.Tensor,
      alpha: torch.Tensor,
      epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
      """Compute KL penalty and dual variable loss."""
      kl_mean = kl.mean(dim=0)
      kl_loss = (alpha.detach() * kl_mean).sum()
      alpha_loss = (alpha * (epsilon - kl_mean.detach())).sum()
      return kl_loss, alpha_loss

    def weights_and_temperature_loss(
      q_values: torch.Tensor,
      epsilon: float,
      temperature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
      """Compute softmax weights and temperature loss (E-step)."""
      tempered_q_values = q_values.detach() / temperature
      weights = torch.nn.functional.softmax(tempered_q_values, dim=0)
      weights = weights.detach()

      # Temperature loss (dual of the E-step).
      q_log_sum_exp = torch.logsumexp(tempered_q_values, dim=0)
      num_actions = torch.as_tensor(q_values.shape[0], dtype=torch.float32)
      log_num_actions = torch.log(num_actions)
      loss = epsilon + (q_log_sum_exp).mean() - log_num_actions
      loss = temperature * loss

      return weights, loss

    def independent_normals(
      distribution_1: T.Any,
      distribution_2: T.Any | None = None,
    ) -> T.Any:
      """Create independent normal distribution from base distributions."""
      distribution_2 = distribution_2 or distribution_1
      return torch.distributions.independent.Independent(
        torch.distributions.normal.Normal(distribution_1.mean, distribution_2.stddev), -1
      )

    # Clamp dual variables to prevent numerical issues.
    with torch.no_grad():
      self.log_temperature.data.copy_(
        torch.maximum(self.min_log_dual, self.log_temperature)
      )
      self.log_alpha_mean.data.copy_(torch.maximum(self.min_log_dual, self.log_alpha_mean))
      self.log_alpha_std.data.copy_(torch.maximum(self.min_log_dual, self.log_alpha_std))
      if self.action_penalization:
        self.log_penalty_temperature.data.copy_(
          torch.maximum(self.min_log_dual, self.log_penalty_temperature)
        )

      # Sample actions from target policy and evaluate with target critic.
      target_distributions = self.model.target_actor(observations)
      actions = target_distributions.sample((self.num_samples,))  # [num_samples, batch, act_dim]

      tiled_observations = updaters.tile(observations, self.num_samples)
      flat_observations = updaters.merge_first_two_dims(tiled_observations)
      flat_actions = updaters.merge_first_two_dims(actions)
      values = self.model.target_critic(flat_observations, flat_actions)  # [num_samples * batch]
      values = values.view(self.num_samples, -1)  # [num_samples, batch]

      assert isinstance(target_distributions, torch.distributions.normal.Normal)
      target_distributions = independent_normals(target_distributions)

    self.actor_optimizer.zero_grad()
    self.dual_optimizer.zero_grad()

    distributions = self.model.actor(observations)
    distributions = independent_normals(distributions)

    temperature = torch.nn.functional.softplus(self.log_temperature) + FLOAT_EPSILON
    alpha_mean = torch.nn.functional.softplus(self.log_alpha_mean) + FLOAT_EPSILON
    alpha_std = torch.nn.functional.softplus(self.log_alpha_std) + FLOAT_EPSILON
    weights, temperature_loss = weights_and_temperature_loss(values, self.epsilon, temperature)

    # Action penalization is quadratic beyond [-1, 1].
    if self.action_penalization:
      penalty_temperature = (
        torch.nn.functional.softplus(self.log_penalty_temperature) + FLOAT_EPSILON
      )
      diff_bounds = actions - torch.clamp(actions, -1, 1)
      action_bound_costs = -torch.norm(diff_bounds, dim=-1)  # [num_samples, batch]
      penalty_weights, penalty_temperature_loss = weights_and_temperature_loss(
        action_bound_costs, self.epsilon_penalty, penalty_temperature
      )
      weights += penalty_weights
      temperature_loss += penalty_temperature_loss

    # Decompose the policy into fixed-mean and fixed-std distributions.
    fixed_std_distribution = independent_normals(
      distributions.base_dist, target_distributions.base_dist
    )
    fixed_mean_distribution = independent_normals(
      target_distributions.base_dist, distributions.base_dist
    )

    # Compute the decomposed policy losses (M-step).
    policy_mean_losses = (
      fixed_std_distribution.base_dist.log_prob(actions).sum(dim=-1) * weights
    ).sum(dim=0)
    policy_mean_loss = -(policy_mean_losses).mean()
    policy_std_losses = (
      fixed_mean_distribution.base_dist.log_prob(actions).sum(dim=-1) * weights
    ).sum(dim=0)
    policy_std_loss = -policy_std_losses.mean()

    # Compute the decomposed KL between the target and online policies.
    if self.per_dim_constraining:
      kl_mean = torch.distributions.kl.kl_divergence(
        target_distributions.base_dist, fixed_std_distribution.base_dist
      )
      kl_std = torch.distributions.kl.kl_divergence(
        target_distributions.base_dist, fixed_mean_distribution.base_dist
      )
    else:
      kl_mean = torch.distributions.kl.kl_divergence(target_distributions, fixed_std_distribution)
      kl_std = torch.distributions.kl.kl_divergence(
        target_distributions, fixed_mean_distribution
      )

    # Compute the alpha-weighted KL-penalty and dual losses.
    kl_mean_loss, alpha_mean_loss = parametric_kl_and_dual_losses(
      kl_mean, alpha_mean, self.epsilon_mean
    )
    kl_std_loss, alpha_std_loss = parametric_kl_and_dual_losses(
      kl_std, alpha_std, self.epsilon_std
    )

    # Combine losses.
    policy_loss = policy_mean_loss + policy_std_loss
    kl_loss = kl_mean_loss + kl_std_loss
    dual_loss = alpha_mean_loss + alpha_std_loss + temperature_loss
    loss = policy_loss + kl_loss + dual_loss

    loss.backward()
    if self.gradient_clip > 0:
      torch.nn.utils.clip_grad_norm_(self.actor_variables, self.gradient_clip)
      torch.nn.utils.clip_grad_norm_(self.dual_variables, self.gradient_clip)
    self.actor_optimizer.step()
    self.dual_optimizer.step()

    dual_variables = dict(
      temperature=temperature.detach(),
      alpha_mean=alpha_mean.detach(),
      alpha_std=alpha_std.detach(),
    )
    if self.action_penalization:
      dual_variables['penalty_temperature'] = penalty_temperature.detach()

    return dict(
      policy_mean_loss=policy_mean_loss.detach(),
      policy_std_loss=policy_std_loss.detach(),
      kl_mean_loss=kl_mean_loss.detach(),
      kl_std_loss=kl_std_loss.detach(),
      alpha_mean_loss=alpha_mean_loss.detach(),
      alpha_std_loss=alpha_std_loss.detach(),
      temperature_loss=temperature_loss.detach(),
      **dual_variables,
    )
