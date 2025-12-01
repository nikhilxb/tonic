import typing as T

import gym
import gym.spaces
import numpy as np
import torch


ArrayLike = float | T.Sequence[float] | np.ndarray


# ==================================================================================================
# Observation normalizers

class MeanStdNormalizer(torch.nn.Module):
  """Observation normalizer using mean and standard deviation."""

  def __init__(
    self,
    mean: ArrayLike = 0.,
    std: ArrayLike = 1.,
    clip: ArrayLike | None = None,
    freeze: bool = False,
  ):
    """
    Args:
      mean: Initial mean value.
      std: Initial standard deviation value.
      clip: Clipping range for normalized values, optional.
      freeze: If True, disable normalization updates.
    """
    super().__init__()
    self.mean = torch.nn.Buffer(torch.as_tensor(mean, dtype=torch.float), requires_grad=False)
    self.mean_sq = torch.nn.Buffer(torch.as_tensor(mean, dtype=torch.float)**2, requires_grad=False)
    self.count = torch.nn.Buffer(torch.ones(1, dtype=torch.int), requires_grad=False)
    self.std = torch.nn.Buffer(torch.as_tensor(std, dtype=torch.float), requires_grad=False)
    self._clip = None if clip is None else torch.as_tensor(clip, dtype=torch.float)
    self._new_sum = 0.
    self._new_sum_sq = 0.
    self._new_count = 0
    self._eps = 1e-2
    self._freeze = freeze

  def initialize(self, observation_space: gym.spaces.Box) -> None:
    """
    Args:
      observation_space: `Box` observation space.
    """
    assert observation_space.shape is not None
    shape = observation_space.shape
    mean = self.mean.broadcast_to(shape).clone()
    self.mean = mean
    self.mean_sq = mean.square()
    self.std = self.std.broadcast_to(shape).clone()

  @torch.no_grad()
  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    """
    Args:
      obs: Observation tensor `[batch_size, ...]`.
      
    Returns:
      Normalized observation `[batch_size, ...]`.
    """
    obs = (obs - self.mean) / self.std
    return (
      torch.clamp(obs, -self._clip, self._clip) if self._clip is not None else obs
    )

  @torch.no_grad()
  def unnormalize(self, obs: torch.Tensor) -> torch.Tensor:
    """
    Args:
      obs: Normalized observation tensor `[batch_size, ...]`.
      
    Returns:
      Unnormalized observation `[batch_size, ...]`.
    """
    return obs * self.std + self.mean

  @torch.no_grad()
  def record(self, obs: torch.Tensor) -> None:
    """
    Args:
      obs: Observation batch `[batch_size, ...]`.
    """
    if self._freeze: return
    # obs = [batch_size, ...]
    self._new_sum += obs.sum(dim=0)  # [...]
    self._new_sum_sq += obs.square().sum(dim=0)  # [...]
    self._new_count += obs.size(0)  # [...]

  @torch.no_grad()
  def update(self) -> None:
    """Update normalization statistics from recorded observations."""
    if self._freeze: return
    # Merge old and new state.
    new_count = self.count + self._new_count
    new_mean = self._new_sum / self._new_count
    new_mean_sq = self._new_sum_sq / self._new_count
    w_old = self.count / new_count
    w_new = self._new_count / new_count
    self.mean[:] = w_old * self.mean + w_new * new_mean
    self.mean_sq[:] = w_old * self.mean_sq + w_new * new_mean_sq
    self.count[:] = new_count

    # Compute derived quantities.
    var = torch.clamp(self.mean_sq - self.mean.square(), min=0.)
    self.std[:] = torch.clamp(torch.sqrt(var), min=self._eps)

    # Reset state.
    self._new_sum = 0.
    self._new_sum_sq = 0.
    self._new_count = 0


class NegPosNormalizer(torch.nn.Module):
  """Observation normalizer using negative-positive range mapping."""

  def __init__(
    self,
    min: ArrayLike = 0.,
    mid: ArrayLike = 0.,
    max: ArrayLike = 0.,
    freeze: bool = False,
  ):
    """
    Args:
      min: Initial minimum value.
      mid: Initial middle value.
      max: Initial maximum value.
      freeze: If True, disable normalization updates.
    """
    super().__init__()
    self.min = torch.nn.Buffer(torch.as_tensor(min, dtype=torch.float), requires_grad=False)
    self.mid = torch.nn.Buffer(torch.as_tensor(mid, dtype=torch.float), requires_grad=False)
    self.max = torch.nn.Buffer(torch.as_tensor(max, dtype=torch.float), requires_grad=False)

    self._new_min = torch.as_tensor(min, dtype=torch.float)
    self._new_max = torch.as_tensor(max, dtype=torch.float)
    self._freeze = freeze

  def initialize(self, observation_space: gym.spaces.Box) -> None:
    """
    Args:
      observation_space: `Box` observation space.
    """
    assert observation_space.shape is not None
    shape = observation_space.shape
    min = self.min.broadcast_to(shape).clone()
    mid = self.mid.broadcast_to(shape).clone()
    max = self.max.broadcast_to(shape).clone()
    min = torch.minimum(min, mid)
    max = torch.maximum(max, mid)
    self.min = min
    self.mid = mid
    self.max = max

  @torch.no_grad()
  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    """
    Args:
      obs: Observation tensor `[batch_size, ...]`.
      
    Returns:
      Normalized observation `[batch_size, ...]`.
    """
    # Convert [min, _mid, max] to [-1, 0, 1]:
    # val' = (-1 + (val - min) / (mid - min)) if val < mid else (0 + (val - mid) / (max - mid))
    obs = torch.clamp(obs, self.min, self.max)
    neg = (obs < self.mid).float() * ((obs - self.min) / (self.mid - self.min) - 1)
    pos = (obs >= self.mid).float() * (obs - self.mid) / (self.max - self.mid)
    obs = torch.nan_to_num(neg, 0, 0, 0) + torch.nan_to_num(pos, 0, 0, 0)
    return obs

  @torch.no_grad()
  def unnormalize(self, obs: torch.Tensor) -> torch.Tensor:
    """
    Args:
      obs: Normalized observation tensor `[batch_size, ...]`.
      
    Returns:
      Unnormalized observation `[batch_size, ...]`.
    """
    # Convert [-1, 0, 1] to [min, mid, max]:
    # val' = min + (val + 1) * (mid - min) if val < 0 else mid + val * (max - mid)
    obs = torch.clamp(obs, -1, 1)
    obs = ((obs < 0).float() * (self.min + (obs + 1) * (self.mid - self.min)) +
            (obs >= 0).float() * (self.mid + obs * (self.max - self.mid)))
    return obs

  @torch.no_grad()
  def record(self, obs: torch.Tensor) -> None:
    """
    Args:
      obs: Observation batch `[batch_size, ...]`.
    """
    if self._freeze: return
    # obs = [batch_size, ...]
    self._new_min = torch.min(obs, dim=0).values
    self._new_max = torch.max(obs, dim=0).values

  @torch.no_grad()
  def update(self) -> None:
    """Update normalization statistics from recorded observations."""
    if self._freeze: return
    self.min[:] = torch.minimum(self.min, self._new_min)
    self.max[:] = torch.maximum(self.max, self._new_max)


Normalizer = MeanStdNormalizer | NegPosNormalizer


def meanstd_builder(key: str) -> MeanStdNormalizer:
  return MeanStdNormalizer()


def posneg_builder(key: str) -> NegPosNormalizer:
  return NegPosNormalizer()


class UnflatNormalizer(torch.nn.Module):
  """Normalizer wrapper for `Dict` observations."""

  def __init__(
    self,
    normalizer_builder: T.Mapping[str, Normalizer | None] |
    T.Callable[[str], Normalizer | None] = posneg_builder,
  ):
    """
    Args:
      normalizer_builder: Mapping or callable to create normalizers per observation key.
    """
    super().__init__()
    self.normalizer_builder = normalizer_builder
    self.normalizers: dict[str, Normalizer] = torch.nn.ModuleDict()  # type: ignore

  def initialize(self, observation_space: gym.spaces.Dict) -> None:
    """
    Args:
      observation_space: `Dict` observation space.
    """
    assert isinstance(observation_space, gym.spaces.Dict)
    for key, o in observation_space.spaces.items():
      assert isinstance(o, gym.spaces.Box)
      if callable(self.normalizer_builder):
        normalizer = self.normalizer_builder(key)
      else:
        normalizer = self.normalizer_builder.get(key, None)
      if normalizer is None: continue
      normalizer.initialize(o)
      self.normalizers[key] = normalizer

  @torch.no_grad()
  def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Args:
      obs: `Dict` observations `{key: [batch_size, ...]}`.
      
    Returns:
      Normalized observations `{key: [batch_size, ...]}`.
    """
    obs = obs.copy()
    for key, o in obs.items():
      if key in self.normalizers:
        obs[key] = self.normalizers[key](o)
    return obs

  @torch.no_grad()
  def unnormalize(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Args:
      obs: Normalized `Dict` observations `{key: [batch_size, ...]}`.
      
    Returns:
      Unnormalized observations `{key: [batch_size, ...]}`.
    """
    obs = obs.copy()
    for key, o in obs.items():
      if key in self.normalizers:
        obs[key] = self.normalizers[key].unnormalize(o)
    return obs

  @torch.no_grad()
  def record(self, obs: dict[str, torch.Tensor]) -> None:
    """
    Args:
      obs: `Dict` observation batch `{key: [batch_size, ...]}`.
    """
    for key, vals in obs.items():
      if key in self.normalizers:
        self.normalizers[key].record(vals)

  @torch.no_grad()
  def update(self) -> None:
    """Update normalization statistics from recorded observations."""
    for normalizer in self.normalizers.values():
      normalizer.update()


ObservationNormalizer = Normalizer | UnflatNormalizer


# ==================================================================================================
# Return normalizers

class DiscountedMinMaxNormalizer(torch.nn.Module):
  """Return normalizer using discounted min-max scaling. The discount factor amplifies the
  normalization range to account for the maximum possible return over an infinite horizon,
  ensuring the sigmoid output maps to the full range of possible discounted returns."""
  # TODO: Unclear if this is the best approach for return normalization.

  def __init__(self, discount_factor: float):
    """
    Args:
      discount_factor: Discount factor for computing return bounds.
    """
    super().__init__()
    assert 0 <= discount_factor < 1
    self._coefficient = 1 / (1 - discount_factor)
    self.min = torch.nn.Buffer(torch.tensor(-1., dtype=torch.float), requires_grad=False)
    self.max = torch.nn.Buffer(torch.tensor(1., dtype=torch.float), requires_grad=False)
    
    self._new_min = torch.tensor(-1., dtype=torch.float)
    self._new_max = torch.tensor(1., dtype=torch.float)

  @torch.no_grad()
  def forward(self, val: torch.Tensor) -> torch.Tensor:
    """
    Args:
      val: Return value tensor `[batch_size]` (unbounded).
      
    Returns:
      Normalized return `[batch_size]` scaled to discounted bounds `[coefficient * min, coefficient * max]`.
    """
    val = torch.sigmoid(val)  # [batch_size] -> [0, 1]
    low = self._coefficient * self.min  # Discounted lower bound
    high = self._coefficient * self.max  # Discounted upper bound
    return low + val * (high - low)  # [batch_size]

  @torch.no_grad()
  def record(self, values: torch.Tensor) -> None:
    """
    Args:
      values: Return batch `[batch_size]`.
    """
    min_val = torch.min(values)  # []
    max_val = torch.max(values)  # []
    if min_val < self._new_min:
      self._new_min = min_val
    if max_val > self._new_max:
      self._new_max = max_val

  @torch.no_grad()
  def update(self) -> None:
    """Update normalization bounds from recorded returns."""
    self.min[:] = torch.minimum(self.min, self._new_min)
    self.max[:] = torch.maximum(self.max, self._new_max)
      

ReturnNormalizer = DiscountedMinMaxNormalizer