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
    max_steps: int = 4096,
    discount_factor: float = 0.99,
    trace_decay: float = 0.97,
  ):
    """Initialize the on-policy replay buffer.
    
    Args:
      max_steps: Maximum number of steps to record (`max_samples = max_steps * num_envs`).
      discount_factor: Discount factor (gamma) for return computation used in GAE.
      trace_decay: Decay factor (lambda) for return computation used in GAE.
    """
    self.max_steps = max_steps
    self.discount_factor = discount_factor
    self.trace_decay = trace_decay

    # Each buffer stores a tensor or dict of tensors, [max_steps, num_envs, ...].
    self._buffers: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}
    self._cumulative_steps: int = 0
    self._num_envs: int = 0
    self._index: int = 0

  def initialize(self, seed: int) -> None:
    """Initialize buffer state and random number generator.
    
    Args:
      seed: Random seed for reproducible batch shuffling.
    """
    self._rng = torch.Generator().manual_seed(seed)

  def reset(self) -> None:
    """Reset the buffer to empty state."""
    self._index = 0

  def is_full(self) -> bool:
    """
    Returns:
      True if buffer has `max_steps` stored steps.
    """
    return self._index == self.max_steps
 
  def num_steps(self) -> int:
    """
    Returns:
      Number of steps currently in the buffer.
    """
    return self._index
  
  def num_samples(self) -> int:
    """
    Returns:
      Number of samples currently in the buffer (`num_samples = num_steps * num_envs`) .
    """
    return self._index * self._num_envs
  
  def cumulative_steps(self) -> int:
    """
    Returns:
      Cumulative number of steps ever recorded into the buffer.
    """
    return self._cumulative_steps
  
  def cumulative_samples(self) -> int:
    """
    Returns:
      Cumulative number of samples ever recorded into the buffer
      (`cumulative_samples = max_steps * num_envs`).
    """
    return self._cumulative_steps * self._num_envs

  def record(self, keyvals: RecordValues) -> None:
    """Record a single timestep of transitions from parallel environments.
    
    Args:
      keyvals: Named tensors with shape [num_envs, ...] containing transition data.
        Typically includes: observations, actions, rewards, resets, terminations.
        Values can be either flat tensors or dicts of tensors (for unflat observations/actions).
    """
    assert self._index < self.max_steps, "Buffer already full"

    if len(self._buffers) == 0:
      # Initialize buffers on first call using the provided data shapes.
      first_val = T.cast(torch.Tensor | dict[str, torch.Tensor], list(keyvals.values())[0])
      if isinstance(first_val, dict):
        self._num_envs = list(first_val.values())[0].shape[0]  # [num_envs, ...]
      else:
        self._num_envs = first_val.shape[0]  # [num_envs, ...]
      
      for key, val in keyvals.items():
        if isinstance(val, dict):
          # Unpacked: create nested dict of buffers
          self._buffers[key] = {
            k: torch.empty((self.max_steps,) + v.shape, dtype=v.dtype, device=v.device)
            for k, v in val.items()
          }
        else:
          # Packed: create single buffer
          shape = (self.max_steps,) + val.shape  # [max_steps, num_envs, ...]
          self._buffers[key] = torch.empty(shape, dtype=val.dtype, device=val.device)
    
    # Store current timestep data.
    for key, val in keyvals.items():
      if isinstance(val, dict):
        for k, v in val.items():
          self._buffers[key][k][self._index] = v  # type: ignore
      else:
        self._buffers[key][self._index] = val  # type: ignore
    
    self._index += 1

  def get_full(self, *keys: Keys) -> dict[Keys, T.Any]:
    """Retrieve flattened buffer data for specified keys.
    
    Args:
      *keys: Names of buffers to retrieve (e.g., 'observations', 'actions', 'advantages').
      
    Returns:
      Dictionary mapping keys to flattened tensors or dicts of tensors with shape [max_steps * num_envs, ...].
    """
    assert len(self._buffers) > 0, "Buffer not initialized"
    assert self._index == self.max_steps, "Buffer not full"

    full: dict[Keys, T.Any] = {}
    for key in keys:
      val = self._buffers[key]
      if isinstance(val, dict):
        # Unpacked: flatten each sub-tensor
        full[key] = {k: utils.flatten_batch(v) for k, v in val.items()}
      else:
        # Packed: flatten single tensor
        full[key] = utils.flatten_batch(val)
    return full

  def get_minibatches(self, *keys: Keys, size: int | None) -> T.Iterator[dict[Keys, T.Any]]:
    """Generate training batches from the buffer.
    
    Yields either full buffer or shuffled minibatches for the specified number of iterations.
    Each batch contains complete transition tuples for policy gradient updates.
    
    Args:
      *keys: Names of buffers to include in each batch (e.g., 'observations', 'actions').
      size: Number of samples per minibatch. If None, yields the full buffer.
      
    Yields:
      Dictionary mapping keys to batch tensors with shape [size, ...], where `size` is specified or 
      equal to the full buffer `max_steps * num_envs`.
    """
    full = self.get_full(*keys)

    if size is None:
      # Yield full buffer.
      yield full
    else:
      # Yield shuffled minibatches.
      num_transitions = self.max_steps * self._num_envs
      assert num_transitions % size == 0, f"{num_transitions=} not divisible by {size=}"
      all_indices = torch.arange(num_transitions, dtype=torch.long)  # [num_steps * num_envs]
      shuffled_indices = all_indices[torch.randperm(num_transitions, generator=self._rng)]
      for i in range(0, num_transitions, size):
        indices = shuffled_indices[i : i + size]  # [size]
        minibatch: dict[Keys, T.Any] = {}
        for key, val in full.items():
          if isinstance(val, dict):
            # Unpacked: index each sub-tensor
            minibatch[key] = {k: v[indices] for k, v in val.items()}
          else:
            # Packed: index single tensor
            minibatch[key] = val[indices]
        yield minibatch

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
    buffers = T.cast(BufferValues, self._buffers)
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

  def compute_advantages(self, normalize: bool = False) -> None:
    """Compute advantages from returns and values.
    
    Args:
      normalize: If True, normalize advantages to have zero mean and unit std.
    """
    buffers = T.cast(BufferValues, self._buffers)
    advantages = buffers['returns'] - buffers['values']  # [num_steps, num_envs]
    if normalize:
      std = advantages.std()
      if std != 0:
        advantages = (advantages - advantages.mean()) / std
    buffers['advantages'] = advantages
