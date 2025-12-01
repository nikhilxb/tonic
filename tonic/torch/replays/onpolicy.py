import typing as T

import torch

from . import utils


class RecordValues(T.TypedDict, extra_items=torch.Tensor | dict[str, torch.Tensor]):
  observations: torch.Tensor | dict[str, torch.Tensor]
  actions: torch.Tensor | dict[str, torch.Tensor]
  rewards: torch.Tensor
  resets: torch.Tensor
  terminations: torch.Tensor
  next_observations: torch.Tensor | dict[str, torch.Tensor]


class BufferValues(T.TypedDict, extra_items=torch.Tensor | dict[str, torch.Tensor]):
  observations: torch.Tensor | dict[str, torch.Tensor]
  actions: torch.Tensor | dict[str, torch.Tensor]
  rewards: torch.Tensor
  resets: torch.Tensor
  terminations: torch.Tensor
  next_observations: torch.Tensor | dict[str, torch.Tensor]
  returns: torch.Tensor
  values: torch.Tensor
  next_values: torch.Tensor
  advantages: torch.Tensor


Keys = T.TypeVar('Keys', bound=T.LiteralString)


class OnPolicyBuffer(T.Generic[Keys]):
  """Replay buffer for on-policy algorithms (e.g., PPO, A2C).
  
  Stores recent transitions from parallel environments and discards them after training.
  Uses relatively small buffer sizes (e.g., 2048-4096 steps) since data becomes stale quickly.
  Batches consist of (observation, action, reward, value, advantage, return) tuples sampled
  from the most recent rollouts.
  """

  def __init__(
    self,
    size: int = 4096,
    batch_iterations: int = 80,
    batch_size: int | None = None,
    discount_factor: float = 0.99,
    trace_decay: float = 0.97,
  ):
    """Initialize the on-policy replay buffer.
    
    Args:
      size: Number of timesteps to collect per environment before training.
      batch_iterations: Number of batch iterations per training epoch.
      batch_size: Size of minibatches for SGD updates. If None, uses full buffer.
      discount_factor: Discount factor (gamma) for return computation.
      trace_decay: Decay factor (lambda) for GAE/lambda-return computation.
    """
    self.max_size = size
    self.batch_iterations = batch_iterations
    self.batch_size = batch_size
    self.discount_factor = discount_factor
    self.trace_decay = trace_decay

  def initialize(self, seed: int) -> None:
    """Initialize buffer state and random number generator.
    
    Args:
      seed: Random seed for reproducible batch shuffling.
    """
    self.rng = torch.Generator().manual_seed(seed)
    self.buffers: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}
    self.index: int = 0
    self.num_envs: int = 0

  def reset(self) -> None:
    """Reset buffer index to start recording from the beginning."""
    self.index = 0

  def ready(self) -> bool:
    """Check if buffer is full and ready for training.
    
    Returns:
      True if buffer has collected `max_size` timesteps.
    """
    return self.index >= self.max_size

  def record(self, keyvals: RecordValues) -> None:
    """Record a single timestep of transitions from parallel environments.
    
    Args:
      keyvals: Named tensors with shape [num_envs, ...] containing transition data.
        Typically includes: observations, actions, rewards, resets, terminations.
        Values can be either flat tensors or dicts of tensors (for unflat observations/actions).
    """
    if len(self.buffers) == 0:
      # Initialize buffers on first call using the provided data shapes.
      first_val = T.cast(torch.Tensor | dict[str, torch.Tensor], list(keyvals.values())[0])
      if isinstance(first_val, dict):
        self.num_envs = list(first_val.values())[0].shape[0]  # [num_envs, ...]
      else:
        self.num_envs = first_val.shape[0]  # [num_envs, ...]
      
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
    self.index += 1

  def get_full(self, *keys: Keys) -> dict[Keys, T.Any]:
    """Retrieve flattened buffer data for specified keys.
    
    Reshapes buffer from [num_steps, num_envs, ...] to [num_steps * num_envs, ...] where
    each batch element is a complete transition tuple.
    
    Args:
      *keys: Names of buffers to retrieve (e.g., 'observations', 'actions', 'advantages').
      
    Returns:
      Dictionary mapping keys to flattened tensors or dicts of tensors with shape [num_steps * num_envs, ...].
    """
    assert len(self.buffers) > 0, "Buffers not initialized"

    full: dict[Keys, T.Any] = {}
    for key in keys:
      val = self.buffers[key]
      if isinstance(val, dict):
        # Unpacked: flatten each sub-tensor
        full[key] = {k: utils.flatten_batch(v) for k, v in val.items()}
      else:
        # Packed: flatten single tensor
        full[key] = utils.flatten_batch(val)
    return full

  def get_batches(self, *keys: Keys) -> T.Iterator[dict[Keys, T.Any]]:
    """Generate training batches from the buffer.
    
    Yields either full buffer or shuffled minibatches for the specified number of iterations.
    Each batch contains complete transition tuples for policy gradient updates.
    
    Args:
      *keys: Names of buffers to include in each batch (e.g., 'observations', 'actions').
      
    Yields:
      Dictionary mapping keys to batch tensors with shape [batch_size, ...] where
      batch_size is either the full buffer size or the specified minibatch size.
    """
    full = self.get_full(*keys)

    if self.batch_size is None:
      # Yield full buffer for each iteration.
      for _ in range(self.batch_iterations):
        yield full  # [num_steps * num_envs, ...]
    else:
      # Yield shuffled minibatches.
      size = self.max_size * self.num_envs  # Total number of transitions
      all_indices = torch.arange(size, dtype=torch.long)  # [num_steps * num_envs]
      for _ in range(self.batch_iterations):
        shuffled_indices = all_indices[torch.randperm(size, generator=self.rng)]
        for i in range(0, size, self.batch_size):
          indices = shuffled_indices[i : i + self.batch_size]  # [batch_size]
          batch: dict[Keys, T.Any] = {}
          for key, val in full.items():
            if isinstance(val, dict):
              # Unpacked: index each sub-tensor
              batch[key] = {k: v[indices] for k, v in val.items()}
            else:
              # Packed: index single tensor
              batch[key] = val[indices]
          yield batch

  def compute_advantages(self, normalize: bool = False) -> None:
    """Compute advantages from returns and values.
    
    Args:
      normalize: If True, normalize advantages to have zero mean and unit std.
    """
    buffers = T.cast(BufferValues, self.buffers)
    advs = buffers['returns'] - buffers['values']  # [num_steps, num_envs]
    if normalize:
      std = advs.std()
      if std != 0:
        advs = (advs - advs.mean()) / std
    buffers['advantages'] = advs

  def compute_returns(
    self,
    values: torch.Tensor,
    next_values: torch.Tensor,
  ) -> None:
    """Compute lambda-returns using GAE and store value estimates in the buffer.
    
    Args:
      values: Value function estimates for current observations, shape [num_steps * num_envs].
      next_values: Value function estimates for next observations, shape [num_steps * num_envs].
    """
    buffers = T.cast(BufferValues, self.buffers)
    shape = buffers['rewards'].shape  # [num_steps, num_envs]
    buffers['values'] = values.reshape(shape)
    buffers['next_values'] = next_values.reshape(shape)
    buffers['returns'] = utils.lambda_returns(
      values=buffers['values'],
      next_values=buffers['next_values'],
      rewards=buffers['rewards'],
      resets=buffers['resets'],
      terminations=buffers['terminations'],
      discount_factor=self.discount_factor,
      trace_decay=self.trace_decay,
    )
