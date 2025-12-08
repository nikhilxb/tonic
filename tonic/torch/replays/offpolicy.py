import typing as T

import torch


class RecordValues(T.TypedDict, extra_items=torch.Tensor | dict[str, torch.Tensor]):
  observations: torch.Tensor | dict[str, torch.Tensor]
  actions: torch.Tensor | dict[str, torch.Tensor]
  rewards: torch.Tensor
  resets: torch.Tensor
  terminations: torch.Tensor
  next_observations: torch.Tensor | dict[str, torch.Tensor]
  discounts: T.NotRequired[torch.Tensor]


class BufferValues(T.TypedDict, extra_items=torch.Tensor | dict[str, torch.Tensor]):
  observations: torch.Tensor | dict[str, torch.Tensor]
  actions: torch.Tensor | dict[str, torch.Tensor]
  rewards: torch.Tensor
  resets: torch.Tensor
  terminations: torch.Tensor
  next_observations: torch.Tensor | dict[str, torch.Tensor]
  discounts: torch.Tensor


Keys = T.TypeVar('Keys', bound=T.LiteralString)


class OffPolicyBuffer(T.Generic[Keys]):
  """Replay buffer for off-policy algorithms (e.g., SAC, TD3, DDPG).
  
  Stores a large number of transitions from parallel environments for replay.
  Uses large buffer sizes (e.g., 1M transitions) to maintain diverse experience.
  Batches consist of (observation, action, reward, next_observation, discount) tuples
  randomly sampled from the entire buffer history.
  """

  def __init__(
    self,
    max_samples: int = 1_000_000,
    discount_factor: float = 0.99,
    return_steps: int = 1,
  ):
    """Initialize the off-policy replay buffer.
    
    Args:
      max_samples: Maximum total transition samples across all environments.
      discount_factor: Discount factor (gamma) for n-step bootstrapped return computation.
      return_steps: Number of steps for n-step bootstrapped returns.
    """
    self.max_samples = max_samples
    self.discount_factor = discount_factor
    self.return_steps = return_steps

    # Each buffer stores a tensor or dict of tensors, [max_steps, num_envs, ...].
    self._buffers: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}
    self._cumulative_steps: int = 0
    self._max_steps: int = 0
    self._num_steps: int = 0
    self._num_envs: int = 0
    self._index: int = 0

  def initialize(self, seed: int) -> None:
    """Initialize buffer state and random number generator.
    
    Args:
      seed: Random seed for reproducible sampling.
    """
    self._rng = torch.Generator().manual_seed(seed)

  def is_full(self) -> bool:
    """
    Returns:
      True if the buffer has `max_samples` stored transition samples.
    """
    return self._num_steps * self._num_envs == self.max_samples
  
  def num_steps(self) -> int:
    """
    Returns:
      Number of steps currently in the buffer.
    """
    return self._num_steps
  
  def num_samples(self) -> int:
    """
    Returns:
      Number of samples currently in the buffer (`num_samples = num_steps * num_envs`).
    """
    return self._num_steps * self._num_envs
  
  def cumulative_steps(self) -> int:
    """
    Returns:
      Cumulative number of steps recorded since initialization.
    """
    return self._cumulative_steps
  
  def cumulative_samples(self) -> int:
    """
    Returns:
      Cumulative number of samples recorded since initialization.
    """
    return self._cumulative_steps * self._num_envs

  def record(self, keyvals: RecordValues) -> None:
    """Record a single timestep of transitions from parallel environments.
    
    Automatically computes discounts from terminations and accumulates n-step returns.
    
    Args:
      keyvals: Transition data, tensors or dicts of tensors, [num_envs, ...].
    """
    # Compute discount factors from terminations.
    continuations = 1 -  keyvals['terminations'].float()  # [num_envs]
    keyvals['discounts'] = continuations * self.discount_factor

    # Initialize buffers on first call.
    if len(self._buffers) == 0:
      first_val = T.cast(torch.Tensor | dict[str, torch.Tensor], list(keyvals.values())[0])
      if isinstance(first_val, dict):
        self._num_envs = list(first_val.values())[0].shape[0]  # [num_envs, ...]
      else:
        self._num_envs = first_val.shape[0]  # [num_envs, ...]
      self._max_steps = self.max_samples // self._num_envs
      
      for key, val in keyvals.items():
        if isinstance(val, dict):
          # Unpacked: create nested dict of buffers
          self._buffers[key] = {
            k: torch.empty((self._max_steps,) + v.shape, dtype=v.dtype, device=v.device)
            for k, v in val.items()
          }
        else:
          # Packed: create single buffer
          shape = (self._max_steps,) + val.shape  # [max_steps, num_envs, ...]
          self._buffers[key] = torch.empty(shape, dtype=val.dtype, device=val.device)

    # Store current timestep data.
    for key, val in keyvals.items():
      if isinstance(val, dict):
        for k, v in val.items():
          self._buffers[key][k][self._index] = v  # type: ignore
      else:
        self._buffers[key][self._index] = val  # type: ignore

    # Accumulate n-step returns by updating past entries.
    if self.return_steps > 1:
      self._compute_nstep_returns(keyvals)

    self._index = (self._index + 1) % self._max_steps
    self._num_steps = min(self._num_steps + 1, self._max_steps)
    self._cumulative_steps += 1

  def _compute_nstep_returns(self, keyvals: RecordValues) -> None:
    """Compute n-step returns by updating past buffer entries.
    
    For each of the past n-1 transitions, updates their rewards, discounts, and next
    observations to reflect n-step bootstrapping. Uses masks to avoid accumulation
    across episode boundaries (resets).
    
    Args:
      keyvals: Current timestep data.
    """
    assert 'discounts' in keyvals, "Discounts must be computed before accumulating n-steps."
    rewards = keyvals['rewards']  # [num_envs]
    next_observations = keyvals['next_observations']  # tensor/dict, [num_envs, observation_size]
    discounts = keyvals['discounts']  # [num_envs]
    masks = torch.ones(self._num_envs, device=rewards.device)  # [num_envs]
    
    buffers = T.cast(BufferValues, self._buffers)
    for i in range(min(self._num_steps, self.return_steps - 1)):
      index = (self._index - i - 1) % self._max_steps
      # Zero out masks for environments that hit episode boundaries.
      masks *= (1 - buffers['resets'][index])  # [num_envs]
      
      # Update accumulated reward: R_t = r_t + gamma * R_{t+1}
      new_rewards = buffers['rewards'][index] + buffers['discounts'][index] * rewards
      buffers['rewards'][index] = (1 - masks) * buffers['rewards'][index] + masks * new_rewards
      
      # Update accumulated discount: gamma_t = gamma_t * gamma_{t+1}
      new_discounts = buffers['discounts'][index] * discounts
      buffers['discounts'][index] = (
        (1 - masks) * buffers['discounts'][index] + masks * new_discounts
      )
      
      # Update next observation to n-step ahead observation.
      if isinstance(next_observations, dict):
        # Unpacked: update each sub-tensor
        buffers_next_observations = T.cast(dict[str, torch.Tensor], buffers['next_observations'])
        for k in next_observations.keys():
          buffers_next_observations[k][index] = (
            (1 - masks)[:, None] * buffers_next_observations[k][index] +
            masks[:, None] * next_observations[k]
          )
      else:
        # Packed: update single tensor
        buffers_next_observations = T.cast(torch.Tensor, buffers['next_observations'])
        buffers_next_observations[index] = (
          (1 - masks)[:, None] * buffers_next_observations[index] +
          masks[:, None] * next_observations
        )

  def get_minibatch(self, *keys: Keys, size: int) -> dict[Keys, T.Any]:
    """Generate random minibatches from the buffer.
    
    Samples uniformly from all stored transitions to produce training batches.
    Each batch contains complete transition tuples for Q-learning updates.
    
    Args:
      *keys: Names of buffers to include in each batch (e.g., 'observations', 'actions').
      step: Current step number.
      
    Yields:
      Dictionary mapping keys to batch tensors with shape [size, ...].
    """
    assert len(self._buffers) > 0, "Buffers not initialized"
    
    num_samples = self._num_steps * self._num_envs
    # Indices of random subset from the full buffer, [size].
    indices = torch.randint(num_samples, (size,), generator=self._rng, dtype=torch.long)
    steps = indices // self._num_envs  # Step indices, 0 <= steps < self.num_steps.
    envs = indices % self._num_envs  # Environment indices, 0 <= envs < self.num_envs.
    
    minibatch: dict[Keys, T.Any] = {}
    for key in keys:
      val = self._buffers[key]
      if isinstance(val, dict):
        # Unpacked: index each sub-tensor
        minibatch[key] = {k: v[steps, envs] for k, v in val.items()}
      else:
        # Packed: index single tensor
        minibatch[key] = val[steps, envs]
    return minibatch
    
