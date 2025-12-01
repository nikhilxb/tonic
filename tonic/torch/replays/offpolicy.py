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
    size: int = 1_000_000,
    return_steps: int = 1,
    batch_iterations: int = 50,
    batch_size: int = 100,
    discount_factor: float = 0.99,
    steps_before_batches: int = 10_000,
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
    self.max_transitions = size
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
    # Each buffer stores a tensor or dict of tensors, [max_size, num_envs, ...].
    self.buffers: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}
    self.index: int = 0
    self.size: int = 0
    self.max_size: int = 0
    self.num_envs: int = 0
    self.last_step: int = 0

  def ready(self, step: int) -> bool:
    """Check if buffer is ready for training.
    
    Args:
      step: Current step number.
      
    Returns:
      True if enough steps have elapsed since last training and initial warmup is done.
    """
    if step < self.steps_before_batches:
      return False
    return (step - self.last_step) >= self.steps_between_batches

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
    if len(self.buffers) == 0:
      first_val = T.cast(torch.Tensor | dict[str, torch.Tensor], list(keyvals.values())[0])
      if isinstance(first_val, dict):
        self.num_envs = list(first_val.values())[0].shape[0]  # [num_envs, ...]
      else:
        self.num_envs = first_val.shape[0]  # [num_envs, ...]
      self.max_size = self.max_transitions // self.num_envs
      
      for key, val in keyvals.items():
        if isinstance(val, dict):
          # Unpacked: create nested dict of buffers
          self.buffers[key] = {
            k: torch.empty((self.max_size,) + v.shape, dtype=v.dtype, device=v.device)
            for k, v in val.items()
          }
        else:
          # Packed: create single buffer
          shape = (self.max_size,) + val.shape  # [max_size, num_envs, ...]
          self.buffers[key] = torch.empty(shape, dtype=val.dtype, device=val.device)

    # Store current timestep data.
    for key, val in keyvals.items():
      if isinstance(val, dict):
        for k, v in val.items():
          self.buffers[key][k][self.index] = v  # type: ignore
      else:
        self.buffers[key][self.index] = val  # type: ignore

    # Accumulate n-step returns by updating past entries.
    if self.return_steps > 1:
      self._accumulate_n_steps(keyvals)

    self.index = (self.index + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)

  def _accumulate_n_steps(self, keyvals: RecordValues) -> None:
    """Accumulate n-step returns by updating past buffer entries.
    
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
    # Masks track which environments haven't encountered episode boundaries (resets).
    masks = torch.ones(self.num_envs, device=rewards.device)  # [num_envs]
    
    buffers = T.cast(BufferValues, self.buffers)
    for i in range(min(self.size, self.return_steps - 1)):
      index = (self.index - i - 1) % self.max_size
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

  def get_batches(self, *keys: Keys, step: int) -> T.Iterator[dict[Keys, T.Any]]:
    """Generate random minibatches from the buffer.
    
    Samples uniformly from all stored transitions to produce training batches.
    Each batch contains complete transition tuples for Q-learning updates.
    
    Args:
      *keys: Names of buffers to include in each batch (e.g., 'observations', 'actions').
      step: Current step number.
      
    Yields:
      Dictionary mapping keys to batch tensors with shape [batch_size, ...].
    """
    assert len(self.buffers) > 0, "Buffers not initialized"
    
    for _ in range(self.batch_iterations):
      num_transitions = self.size * self.num_envs
      # Indices of random subset from the full buffer, [batch_size].
      indices = torch.randint(
        num_transitions, (self.batch_size,), generator=self.rng, dtype=torch.long
      )
      rows = indices // self.num_envs  # Step indices, 0 <= rows < self.size.
      cols = indices % self.num_envs  # Environment indices, 0 <= cols < self.num_envs.
      
      batch: dict[Keys, T.Any] = {}
      for key in keys:
        val = self.buffers[key]
        if isinstance(val, dict):
          # Unpacked: index each sub-tensor
          batch[key] = {k: v[rows, cols] for k, v in val.items()}
        else:
          # Packed: index single tensor
          batch[key] = val[rows, cols]
      yield batch
    
    self.last_step = step
