import typing as T

import torch


class OffPolicyBuffer:
  """Replay buffer for off-policy algorithms (e.g., SAC, TD3, DDPG).
  
  Stores a large number of transitions from parallel environments for replay.
  Uses large buffer sizes (e.g., 1M transitions) to maintain diverse experience.
  Batches consist of (observation, action, reward, next_observation, discount) tuples
  randomly sampled from the entire buffer history.
  """

  def __init__(
    self,
    size: int = int(1e6),
    return_steps: int = 1,
    batch_iterations: int = 50,
    batch_size: int = 100,
    discount_factor: float = 0.99,
    steps_before_batches: int = int(1e4),
    steps_between_batches: int = 50,
  ):
    """Initialize the off-policy replay buffer.
    
    Args:
      size: Maximum total transitions across all environments.
      return_steps: Number of steps for n-step bootstrapped returns.
      batch_iterations: Number of batch iterations per training epoch.
      batch_size: Size of minibatches for SGD updates.
      discount_factor: Discount factor (gamma) for return computation.
      steps_before_batches: Minimum steps before buffer is first ready.
      steps_between_batches: Minimum steps between successive calls when buffer is ready.
    """
    self.full_max_size = size
    self.return_steps = return_steps
    self.batch_iterations = batch_iterations
    self.batch_size = batch_size
    self.discount_factor = discount_factor
    self.steps_before_batches = steps_before_batches
    self.steps_between_batches = steps_between_batches

  def initialize(self, seed: int) -> None:
    """Initialize buffer state and random number generator.
    
    Args:
      seed: Random seed for reproducible sampling.
    """
    self.rng = torch.Generator().manual_seed(seed)
    self.buffers: dict[str, torch.Tensor] | None = None
    self.index: int = 0
    self.size: int = 0
    self.last_steps: int = 0
    self.num_envs: int = 0
    self.max_size: int = 0

  def ready(self, steps: int) -> bool:
    """Check if buffer is ready for training.
    
    Args:
      steps: Current total environment steps.
      
    Returns:
      True if enough steps have elapsed since last training and initial warmup is done.
    """
    if steps < self.steps_before_batches:
      return False
    return (steps - self.last_steps) >= self.steps_between_batches

  def record(self, **kwargs: torch.Tensor) -> None:
    """Record a single timestep of transitions from parallel environments.
    
    Automatically computes discounts from terminations and accumulates n-step returns.
    
    Args:
      **kwargs: Named tensors with shape [num_envs, ...] containing transition data.
        Typically includes: observations, actions, rewards, next_observations, 
        terminations, resets.
    """
    # Compute discount factors from terminations if provided.
    if 'terminations' in kwargs:
      continuations = 1.0 - kwargs['terminations'].float()  # [num_envs]
      kwargs['discounts'] = continuations * self.discount_factor

    # Initialize buffers on first call.
    if self.buffers is None:
      self.num_envs = list(kwargs.values())[0].shape[0]  # [num_envs, ...]
      self.max_size = self.full_max_size // self.num_envs
      self.buffers = {}
      for key, val in kwargs.items():
        shape = (self.max_size,) + val.shape  # [max_size, num_envs, ...]
        self.buffers[key] = torch.empty(shape, dtype=val.dtype, device=val.device)

    # Store current timestep data.
    for key, val in kwargs.items():
      self.buffers[key][self.index] = val

    # Accumulate n-step returns by updating past entries.
    if self.return_steps > 1:
      self._accumulate_n_steps(kwargs)

    self.index = (self.index + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)

  def _accumulate_n_steps(self, kwargs: dict[str, torch.Tensor]) -> None:
    """Accumulate n-step returns by updating past buffer entries.
    
    For each of the past n-1 transitions, updates their rewards, discounts, and next
    observations to reflect n-step bootstrapping. Uses masks to avoid accumulation
    across episode boundaries (resets).
    
    Args:
      kwargs: Current timestep data including 'rewards', 'next_observations', 'discounts', 'resets'.
    """
    assert self.buffers is not None, "Buffers not initialized"
    
    rewards = kwargs['rewards']  # [num_envs]
    next_observations = kwargs['next_observations']  # [num_envs, observation_size]
    discounts = kwargs['discounts']  # [num_envs]
    # Masks track which environments haven't encountered episode boundaries (resets).
    masks = torch.ones(self.num_envs, dtype=torch.float32, device=rewards.device)

    for i in range(min(self.size, self.return_steps - 1)):
      index = (self.index - i - 1) % self.max_size
      # Zero out masks for environments that hit episode boundaries.
      masks *= (1 - self.buffers['resets'][index])  # [num_envs]
      
      # Update accumulated reward: R_t = r_t + gamma * R_{t+1}
      new_rewards = (
        self.buffers['rewards'][index] + self.buffers['discounts'][index] * rewards
      )
      self.buffers['rewards'][index] = (
        (1 - masks) * self.buffers['rewards'][index] + masks * new_rewards
      )
      
      # Update accumulated discount: gamma_t = gamma_t * gamma_{t+1}
      new_discounts = self.buffers['discounts'][index] * discounts
      self.buffers['discounts'][index] = (
        (1 - masks) * self.buffers['discounts'][index] + masks * new_discounts
      )
      
      # Update next observation to n-step ahead observation.
      self.buffers['next_observations'][index] = (
        (1 - masks)[:, None] * self.buffers['next_observations'][index] +
        masks[:, None] * next_observations
      )

  def get_batches(self, *keys: str, steps: int) -> T.Iterator[dict[str, torch.Tensor]]:
    """Generate random minibatches from the buffer.
    
    Samples uniformly from all stored transitions to produce training batches.
    Each batch contains complete transition tuples for Q-learning updates.
    
    Args:
      *keys: Names of buffers to include in each batch (e.g., 'observations', 'actions').
      steps: Current environment steps (used to track last training time).
      
    Yields:
      Dictionary mapping keys to batch tensors with shape [batch_size, ...].
    """
    assert self.buffers is not None, "Buffers not initialized"
    
    for _ in range(self.batch_iterations):
      total_size = self.size * self.num_envs
      # Generate random indices into the flattened buffer.
      indices = torch.randint(
        total_size, (self.batch_size,), generator=self.rng, dtype=torch.long
      )  # [batch_size]
      rows = indices // self.num_envs  # Time indices
      columns = indices % self.num_envs  # Environment indices
      yield {k: self.buffers[k][rows, columns] for k in keys}  # [batch_size, ...]

    self.last_steps = steps
