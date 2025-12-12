import typing as T

import torch
import gym.spaces

from .. import agent
from . import normalizers


# ==================================================================================================
# Distributions

class CategoricalValueDistribution:
  """Categorical distribution with explicit value support for distributional RL."""

  def __init__(self, values: torch.Tensor, logits: torch.Tensor):
    """Initialize categorical distribution.
    
    Args:
      values: Support values [num_atoms].
      logits: Logits for each atom [batch_size, num_atoms].
    """
    self.values = values  # [num_atoms]
    self.logits = logits  # [batch_size, num_atoms]
    self.probabilities = torch.nn.functional.softmax(logits, dim=-1)  # [batch_size, num_atoms]

  def mean(self) -> torch.Tensor:
    """Compute expected value under the distribution.
    
    Returns:
      Expected values [batch_size].
    """
    return (self.probabilities * self.values).sum(dim=-1)  # [batch_size]

  def project(self, returns: torch.Tensor) -> torch.Tensor:
    """Project target returns onto the categorical distribution support.
    
    Args:
      returns: Target return values [batch_size, num_steps].
      
    Returns:
      Projected probabilities [batch_size, num_atoms].
    """
    vmin, vmax = self.values[0], self.values[-1]  # []
    
    # Compute distances to neighboring support values.
    d_pos = torch.cat([self.values, vmin[None]], 0)[1:]  # [num_atoms]
    d_pos = (d_pos - self.values)[None, :, None]  # [1, num_atoms, 1]
    d_neg = torch.cat([vmax[None], self.values], 0)[:-1]  # [num_atoms]
    d_neg = (self.values - d_neg)[None, :, None]  # [1, num_atoms, 1]

    # Clip returns to support range.
    clipped_returns = torch.clamp(returns, vmin, vmax)  # [batch_size, num_steps]
    
    # Compute interpolation weights.
    delta_values = clipped_returns[:, None] - self.values[None, :, None]  # [batch_size, num_atoms, num_steps]
    delta_sign = (delta_values >= 0).float()  # [batch_size, num_atoms, num_steps]
    delta_hat = (
      (delta_sign * delta_values / d_pos) - ((1 - delta_sign) * delta_values / d_neg)
    )  # [batch_size, num_atoms, num_steps]
    delta_clipped = torch.clamp(1 - delta_hat, 0, 1)  # [batch_size, num_atoms, num_steps]

    return (delta_clipped * self.probabilities[:, None]).sum(dim=2)  # [batch_size, num_atoms]


# ==================================================================================================
# Heads

class ValueHead(torch.nn.Module):
  """Value function head that outputs a scalar value estimate."""

  def __init__(
    self,
    init_fn: T.Callable[[torch.nn.Module], None] | None = None,
  ):
    """Initialize the value head.
    
    Args:
      init_fn: Optional initialization function to apply to the layer.
    """
    super().__init__()
    self.init_fn = init_fn

  def initialize(
    self,
    input_size: int,
    return_normalizer: normalizers.ReturnNormalizer | None = None,
  ) -> None:
    self.return_normalizer = return_normalizer
    self.v_layer = torch.nn.Linear(input_size, 1)
    if self.init_fn:
      self.v_layer.apply(self.init_fn)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """Compute value estimate.
    
    Args:
      inputs: Input features [batch_size, input_size].
      
    Returns:
      Value estimates [batch_size].
    """
    out = self.v_layer(inputs)  # [batch_size, 1]
    out = torch.squeeze(out, -1)  # [batch_size]
    if self.return_normalizer:
      out = self.return_normalizer(out)  # [batch_size]
    return out


class DistributionalValueHead(torch.nn.Module):
  """Distributional value head for categorical value distributions (e.g. D4PG)."""

  def __init__(
    self,
    vmin: float,
    vmax: float,
    num_atoms: int,
    init_fn: T.Callable[[torch.nn.Module], None] | None = None,
  ):
    """Initialize the distributional value head.
    
    Args:
      vmin: Minimum value of the support.
      vmax: Maximum value of the support.
      num_atoms: Number of atoms in the categorical distribution.
      init_fn: Optional initialization function to apply to the layer.
    """
    super().__init__()
    self.num_atoms = num_atoms
    self.init_fn = init_fn
    self.values = torch.linspace(vmin, vmax, num_atoms).float()  # [num_atoms]

  def initialize(
    self,
    input_size: int,
    return_normalizer: normalizers.ReturnNormalizer | None = None,
  ) -> None:
    """Initialize the distributional layer.
    
    Args:
      input_size: Dimension of input features.
      return_normalizer: Must be None (not supported for distributional heads).
    """
    if return_normalizer:
      raise ValueError('Return normalizers cannot be used with distributional value head.')
    self.distributional_layer = torch.nn.Linear(input_size, self.num_atoms)
    if self.init_fn:
      self.distributional_layer.apply(self.init_fn)

  def forward(self, inputs: torch.Tensor) -> CategoricalValueDistribution:
    """Compute distributional value estimate.
    
    Args:
      inputs: Input features [batch_size, input_size].
      
    Returns:
      Categorical distribution over value support.
    """
    logits = self.distributional_layer(inputs)  # [batch_size, num_atoms]
    return CategoricalValueDistribution(values=self.values, logits=logits)


# ==================================================================================================
# Critic

class CriticEncoder(T.Protocol):
  def initialize(
    self,
    observation_space: agent.ObservationSpace,
    action_space: agent.ActionSpace,
    observation_normalizer: normalizers.ObservationNormalizer | None = None,
  ) -> int:
    ...

  def forward(self, *inputs: torch.Tensor | dict[str, torch.Tensor]) -> T.Any:
     ...

  def __call__(self, *inputs: torch.Tensor | dict[str, torch.Tensor]) -> T.Any:
     ...


class CriticTorso(T.Protocol):
  def initialize(self, input_size: int) -> int:
    ...

  def forward(self, inputs: torch.Tensor) -> T.Any:
    ...

  def __call__(self, inputs: torch.Tensor) -> T.Any:
    ...


class CriticHead(T.Protocol):
  def initialize(
    self,
    input_size: int,
    return_normalizer: normalizers.ReturnNormalizer | None = None,
  ) -> None:
    ...

  def forward(self, inputs: torch.Tensor) -> T.Any:
    ...

  def __call__(self, inputs: torch.Tensor) -> T.Any:
    ...


class CriticLike(T.Protocol):
  def initialize(
    self,
    observation_space: agent.ObservationSpace,
    action_space: agent.ActionSpace,
    observation_normalizer: normalizers.ObservationNormalizer | None = None,
    return_normalizer: normalizers.ReturnNormalizer | None = None,
  ) -> None:
    ...

  def reset(self) -> None:
    ...

  def forward(self, *inputs: torch.Tensor | dict[str, torch.Tensor]) -> T.Any:
    ...

  def __call__(self, *inputs: torch.Tensor | dict[str, torch.Tensor]) -> T.Any:
    ...


class Critic(torch.nn.Module):
  """Critic that uses `Box` or `Dict` observation and action spaces."""
  
  def __init__(
    self,
    encoder: CriticEncoder,
    torso: CriticTorso | None,
    head: CriticHead,
  ):
    super().__init__()
    self.encoder = encoder
    self.torso = torso
    self.head = head

  def initialize(
    self,
    observation_space: agent.ObservationSpace,
    action_space: agent.ActionSpace,
    observation_normalizer: normalizers.ObservationNormalizer | None = None,
    return_normalizer: normalizers.ReturnNormalizer | None = None,
  ) -> None:    
    size = self.encoder.initialize(observation_space, action_space, observation_normalizer)
    if self.torso is not None:
      size = self.torso.initialize(size)
    self.head.initialize(size, return_normalizer)

  def reset(self) -> None:
    pass

  def forward(self, *inputs: torch.Tensor | dict[str, torch.Tensor]) -> T.Any:
    out = self.encoder(*inputs)
    if self.torso is not None:
      out = self.torso(out)
    return self.head(out)
