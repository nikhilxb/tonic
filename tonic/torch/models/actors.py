import typing as T

import torch
import gym.spaces
import numpy as np

from . import normalizers


FLOAT_EPSILON = 1e-8


class SquashedMultivariateNormalDiag:
  """Tanh-squashed multivariate normal distribution for bounded action spaces (SAC)."""

  def __init__(self, loc: float | torch.Tensor, scale: float | torch.Tensor):
    """Initialize the squashed normal distribution.
    
    Args:
      loc: Mean of the pre-squash normal distribution [batch_size, action_dim].
      scale: Standard deviation of the pre-squash normal [batch_size, action_dim].
    """
    self._distribution = torch.distributions.normal.Normal(loc, scale)

  def rsample_with_log_prob(
    self,
    shape: tuple[int, ...] = (),
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample with reparameterization and compute log probability.
    
    Args:
      shape: Additional sample dimensions to prepend.
      
    Returns:
      Tuple of (squashed samples [*shape, batch_size, action_dim],
                log probabilities [*shape, batch_size, action_dim]).
    """
    samples = self._distribution.rsample(shape)  # [*shape, batch_size, action_dim]
    squashed_samples = torch.tanh(samples)  # [*shape, batch_size, action_dim]
    log_probs = self._distribution.log_prob(samples)  # [*shape, batch_size, action_dim]
    # Jacobian correction for tanh transformation.
    log_probs -= torch.log(1 - squashed_samples ** 2 + 1e-6)  # [*shape, batch_size, action_dim]
    return squashed_samples, log_probs

  def rsample(self, shape: tuple[int, ...] = ()) -> torch.Tensor:
    """Sample with reparameterization (for gradient estimation).
    
    Args:
      shape: Additional sample dimensions to prepend.
      
    Returns:
      Squashed samples [*shape, batch_size, action_dim].
    """
    samples = self._distribution.rsample(shape)  # [*shape, batch_size, action_dim]
    return torch.tanh(samples)  # [*shape, batch_size, action_dim]

  def sample(self, shape: tuple[int, ...] = ()) -> torch.Tensor:
    """Sample without reparameterization (no gradients).
    
    Args:
      shape: Additional sample dimensions to prepend.
      
    Returns:
      Squashed samples [*shape, batch_size, action_dim].
    """
    samples = self._distribution.sample(shape)  # [*shape, batch_size, action_dim]
    return torch.tanh(samples)  # [*shape, batch_size, action_dim]

  def log_prob(self, samples: torch.Tensor):
    """Not implemented - unsquashing introduces approximation errors."""
    raise NotImplementedError(
      'Unsquashed samples cannot be accurately recovered. Use `rsample_with_log_prob` directly.'
    )

  @property
  def loc(self) -> torch.Tensor:
    """Mean of the squashed distribution.
    
    Returns:
      Squashed mean [batch_size, action_dim].
    """
    return torch.tanh(self._distribution.mean)  # [batch_size, action_dim]


class DetachedScaleGaussianPolicyHead(torch.nn.Module):
  """Gaussian policy with learnable mean and fixed (detached) scale per action dimension."""

  def __init__(
    self,
    loc_activation: T.Callable[[], torch.nn.Module] = torch.nn.Tanh,
    loc_fn: T.Callable[[torch.nn.Module], None] | None = None,
    log_scale_init: float = 0.,
    scale_min: float = 1e-4,
    scale_max: float = 1.,
    distribution: T.Type[torch.distributions.normal.Normal] = torch.distributions.normal.Normal,
  ):
    """Initialize the policy head.
    
    Args:
      loc_activation: Activation function for the mean layer (e.g., Tanh for bounded actions).
      loc_fn: Optional initialization function for the mean layer.
      log_scale_init: Initial value for log(scale) parameter.
      scale_min: Minimum allowed scale (for numerical stability).
      scale_max: Maximum allowed scale.
      distribution: Distribution class to use (Normal or SquashedMultivariateNormalDiag).
    """
    super().__init__()
    self.loc_activation = loc_activation
    self.loc_fn = loc_fn
    self.log_scale_init = log_scale_init
    self.scale_min = scale_min
    self.scale_max = scale_max
    self.distribution = distribution

  def initialize(self, input_size: int, action_size: int) -> None:
    """Initialize layers.
    
    Args:
      input_size: Dimension of input features.
      action_size: Dimension of action space.
    """
    self.loc_layer = torch.nn.Sequential(
      torch.nn.Linear(input_size, action_size),
      self.loc_activation(),
    )
    if self.loc_fn:
      self.loc_layer.apply(self.loc_fn)
    # Scale is a learnable parameter, same for all batch elements.
    self.log_scale = torch.nn.Parameter(
      torch.as_tensor([[self.log_scale_init] * action_size], dtype=torch.float32)
    )  # [1, action_size]

  def forward(self, inputs: torch.Tensor) -> torch.distributions.Distribution:
    """Compute action distribution.
    
    Args:
      inputs: Input features [batch_size, input_size].
      
    Returns:
      Action distribution with mean [batch_size, action_size] and scale [batch_size, action_size].
    """
    loc = self.loc_layer(inputs)  # [batch_size, action_size]
    batch_size = inputs.shape[0]
    scale = torch.nn.functional.softplus(self.log_scale) + FLOAT_EPSILON  # [1, action_size]
    scale = torch.clamp(scale, self.scale_min, self.scale_max)  # [1, action_size]
    scale = scale.repeat(batch_size, 1)  # [batch_size, action_size]
    return self.distribution(loc, scale)


class GaussianPolicyHead(torch.nn.Module):
  """Gaussian policy with learnable mean and scale (both state-dependent)."""

  def __init__(
    self,
    loc_activation: T.Callable[[], torch.nn.Module] = torch.nn.Tanh,
    loc_fn: T.Callable[[torch.nn.Module], None] | None = None,
    scale_activation: T.Callable[[], torch.nn.Module] = torch.nn.Softplus,
    scale_min: float = 1e-4,
    scale_max: float = 1,
    scale_fn: T.Callable[[torch.nn.Module], None] | None = None,
    distribution: T.Type[torch.distributions.normal.Normal] = torch.distributions.normal.Normal,
  ):
    """Initialize the policy head.
    
    Args:
      loc_activation: Activation function for the mean layer.
      loc_fn: Optional initialization function for the mean layer.
      scale_activation: Activation function for the scale layer (e.g., Softplus for positivity).
      scale_min: Minimum allowed scale (for numerical stability).
      scale_max: Maximum allowed scale.
      scale_fn: Optional initialization function for the scale layer.
      distribution: Distribution class to use (Normal or SquashedMultivariateNormalDiag).
    """
    super().__init__()
    self.loc_activation = loc_activation
    self.loc_fn = loc_fn
    self.scale_activation = scale_activation
    self.scale_min = scale_min
    self.scale_max = scale_max
    self.scale_fn = scale_fn
    self.distribution = distribution

  def initialize(self, input_size: int, action_size: int) -> None:
    """Initialize layers.
    
    Args:
      input_size: Dimension of input features.
      action_size: Dimension of action space.
    """
    self.loc_layer = torch.nn.Sequential(
      torch.nn.Linear(input_size, action_size),
      self.loc_activation(),
    )
    if self.loc_fn:
      self.loc_layer.apply(self.loc_fn)
    self.scale_layer = torch.nn.Sequential(
      torch.nn.Linear(input_size, action_size),
      self.scale_activation(),
    )
    if self.scale_fn:
      self.scale_layer.apply(self.scale_fn)

  def forward(self, inputs: torch.Tensor) -> torch.distributions.Distribution:
    """Compute action distribution.
    
    Args:
      inputs: Input features [batch_size, input_size].
      
    Returns:
      Action distribution with mean and scale [batch_size, action_size].
    """
    loc = self.loc_layer(inputs)  # [batch_size, action_size]
    scale = self.scale_layer(inputs)  # [batch_size, action_size]
    scale = torch.clamp(scale, self.scale_min, self.scale_max)  # [batch_size, action_size]
    return self.distribution(loc, scale)


class DeterministicPolicyHead(torch.nn.Module):
  """Deterministic policy for DDPG-style algorithms."""

  def __init__(
    self,
    activation: T.Callable[[], torch.nn.Module] = torch.nn.Tanh,
    bias: bool = True,
    fn: T.Callable[[torch.nn.Module], None] | None = None,
  ):
    """Initialize the policy head.
    
    Args:
      activation: Activation function for the action layer (e.g., Tanh for bounded actions).
      bias: Whether to use bias in the linear layer.
      fn: Optional initialization function for the layer.
    """
    super().__init__()
    self.activation = activation
    self.bias = bias
    self.fn = fn

  def initialize(self, input_size: int, action_size: int) -> None:
    """Initialize the action layer.
    
    Args:
      input_size: Dimension of input features.
      action_size: Dimension of action space.
    """
    self.action_layer = torch.nn.Sequential(
      torch.nn.Linear(input_size, action_size, bias=self.bias),
      self.activation(),
    )
    if self.fn:
      self.action_layer.apply(self.fn)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """Compute deterministic action.
    
    Args:
      inputs: Input features [batch_size, input_size].
      
    Returns:
      Deterministic actions [batch_size, action_size].
    """
    return self.action_layer(inputs)  # [batch_size, action_size]


# ==================================================================================================
# Actor

class ActorEncoder(T.Protocol):
  def initialize(
    self,
    observation_space: gym.spaces.Box,
    action_space: gym.spaces.Box,
    observation_normalizer: normalizers.Normalizer | None = None,
  ) -> int:
    ...

  def forward(self, *inputs: torch.Tensor) -> T.Any:
    ...

  def __call__(self, *inputs: torch.Tensor) -> T.Any:
    ...


class ActorTorso(T.Protocol):
  def initialize(self, input_size: int) -> int:
    ...

  def forward(self, inputs: T.Any) -> T.Any:
    ...

  def __call__(self, inputs: T.Any) -> T.Any:
    ...


class ActorHead(T.Protocol):
  def initialize(self, input_size: int, action_size: int) -> None:
    ...

  def forward(self, inputs: T.Any) -> T.Any:
    ...

  def __call__(self, inputs: T.Any) -> T.Any:
    ...


class ActorLike(T.Protocol):
  def initialize(
    self,
    observation_space: gym.spaces.Box,
    action_space: gym.spaces.Box,
    observation_normalizer: normalizers.Normalizer | None = None,
  ) -> None:
    ...

  def reset(self) -> None:
    ...

  def forward(self, *inputs: torch.Tensor) -> T.Any:
    ...

  def __call__(self, *inputs: torch.Tensor | dict[str, torch.Tensor]) -> T.Any:
    ...


class Actor(torch.nn.Module):
  """Actor that uses `Box` observation and action spaces."""
  def __init__(
    self,
    encoder: ActorEncoder,
    torso: ActorTorso | None,
    head: ActorHead,
  ):
    super().__init__()
    self.encoder = encoder
    self.torso = torso
    self.head = head

  def initialize(
    self,
    observation_space: gym.spaces.Box,
    action_space: gym.spaces.Box,
    observation_normalizer: normalizers.Normalizer | None = None,
  ):
    assert isinstance(observation_space, gym.spaces.Box)
    assert isinstance(action_space, gym.spaces.Box)
    assert len(observation_space.shape) == 1, 'Observation must be 1D.'
    assert len(action_space.shape) == 1, 'Action must be 1D.'
    
    size = self.encoder.initialize(observation_space, action_space, observation_normalizer)
    if self.torso is not None:
      size = self.torso.initialize(size)
    self.head.initialize(size, action_space.shape[0])

  def reset(self):
    pass
  
  @T.overload
  def forward(self, observations: torch.Tensor, /) -> T.Any:
     ...
  @T.overload
  def forward(self, *inputs: torch.Tensor) -> T.NoReturn:
     ...
  def forward(self, *inputs: torch.Tensor):
    out = self.encoder(*inputs)
    if self.torso is not None:
      out = self.torso(out)
    return self.head(out)


# ==================================================================================================
# Unflat Actor

class UnflatActorEncoder(T.Protocol):
  def initialize(
    self,
    observation_space: gym.spaces.Dict,
    action_space: gym.spaces.Dict,
    observation_normalizer: normalizers.UnflatNormalizer | None = None,
  ) -> int:
    ...

  def forward(self, *inputs: dict[str, torch.Tensor]) -> T.Any:
    ...

  def __call__(self, *inputs: dict[str, torch.Tensor]) -> T.Any:
    ...


class UnflatActorLike(T.Protocol):
  def initialize(
    self,
    observation_space: gym.spaces.Dict,
    action_space: gym.spaces.Dict,
    observation_normalizer: normalizers.UnflatNormalizer | None = None,
  ) -> None:
    ...

  def reset(self) -> None:
    ...

  def forward(self, *inputs: dict[str, torch.Tensor]) -> T.Any:
    ...

  def __call__(self, *inputs: torch.Tensor | dict[str, torch.Tensor]) -> T.Any:
    ...


class UnflatActor(torch.nn.Module):
  """Actor that uses `Dict` observation and action spaces."""
  def __init__(
    self,
    encoder: UnflatActorEncoder,
    torso: ActorTorso | None,
    head: ActorHead,
  ):
    super().__init__()
    self.encoder = encoder
    self.torso = torso
    self.head = head

  def initialize(
    self,
    observation_space: gym.spaces.Dict,
    action_space: gym.spaces.Dict,
    observation_normalizer: normalizers.UnflatNormalizer | None = None,
  ) -> None:
    assert isinstance(observation_space, gym.spaces.Dict)
    assert isinstance(action_space, gym.spaces.Dict)
    assert all(isinstance(o, gym.spaces.Box) for o in observation_space.spaces.values())
    assert all(isinstance(a, gym.spaces.Box) for a in action_space.spaces.values())
    
    size = self.encoder.initialize(observation_space, action_space, observation_normalizer)
    if self.torso is not None:
      size = self.torso.initialize(size)
    action_size = sum(
      int(np.prod(space.shape)) if space.shape is not None else 1
      for space in action_space.spaces.values()
    )
    self.head.initialize(size, action_size)

  def reset(self) -> None:
    pass

  def forward(self, *inputs: dict[str, torch.Tensor]):
    out = self.encoder(*inputs)
    if self.torso is not None:
      out = self.torso(out)
    return self.head(out)
