import typing as T
import math

import torch
import gym.spaces


from .. import agent, space
from . import normalizers


FLOAT_EPSILON = 1e-8

# ==================================================================================================
# Distributions

Params = T.TypeVarTuple('Params')

class ActionDistribution(T.Protocol[T.Unpack[Params]]):
  """Protocol for action distributions that handle Box or Dict action spaces."""
  action_space: agent.ActionSpace
  
  def __init__(self, action_space: agent.ActionSpace, *params: T.Unpack[Params]):
    ...
  
  def sample(self, shape: tuple[int, ...] = ()) -> agent.Action:
    ...

  def sample_with_log_prob(self, shape: tuple[int, ...] = ()) -> tuple[agent.Action, torch.Tensor]:
    ...

  def rsample(self, shape: tuple[int, ...] = ()) -> agent.Action:
    ...

  def rsample_with_log_prob(self, shape: tuple[int, ...] = ()) -> tuple[agent.Action, torch.Tensor]:
    ...

  def log_prob(self, samples: agent.Action) -> torch.Tensor:
    ...

  def mean(self) -> agent.Action:
    ...

  def std(self) -> torch.Tensor:
    ...

  def entropy(self) -> torch.Tensor:
    ...


class NormalActionDistribution:
  """Normal distribution for action spaces (Box or Dict)."""

  def __init__(self, action_space: agent.ActionSpace, mean: torch.Tensor, std: torch.Tensor):
    """Initialize the normal distribution.
    
    Args:
      action_space: Action space for unpacking actions.
      mean: Mean of the normal distribution [batch_size, action_dim].
      std: Standard deviation of the normal [batch_size, action_dim].
    """
    self.action_space = action_space
    self.distribution = torch.distributions.normal.Normal(mean, std)

  def sample(self, shape: tuple[int, ...] = ()) -> agent.Action:
    """Sample without reparameterization (no gradients).
    
    Args:
      shape: Additional sample dimensions to prepend.
      
    Returns:
      Samples [*shape, batch_size, action_dim] or dict of tensors.
    """
    samples = self.distribution.sample(shape)  # [*shape, batch_size, action_dim]
    return space.unpack_tensors(self.action_space, samples)

  def sample_with_log_prob(self, shape: tuple[int, ...] = ()) -> tuple[agent.Action, torch.Tensor]:
    """Sample without reparameterization and compute log probability.
    
    Args:
      shape: Additional sample dimensions to prepend.
      
    Returns:
      Tuple of (samples [*shape, batch_size, action_dim] or dict,
                log probabilities [*shape, batch_size, action_dim]).
    """
    samples = self.distribution.sample(shape)  # [*shape, batch_size, action_dim]
    log_probs = self.distribution.log_prob(samples)  # [*shape, batch_size, action_dim]
    samples = space.unpack_tensors(self.action_space, samples)
    return samples, log_probs

  def rsample(self, shape: tuple[int, ...] = ()) -> agent.Action:
    """Sample with reparameterization (for gradient estimation).
    
    Args:
      shape: Additional sample dimensions to prepend.
      
    Returns:
      Samples [*shape, batch_size, action_dim] or dict of tensors.
    """
    samples = self.distribution.rsample(shape)  # [*shape, batch_size, action_dim]
    return space.unpack_tensors(self.action_space, samples)

  def rsample_with_log_prob(self, shape: tuple[int, ...] = ()) -> tuple[agent.Action, torch.Tensor]:
    """Sample with reparameterization and compute log probability.
    
    Args:
      shape: Additional sample dimensions to prepend.
      
    Returns:
      Tuple of (samples [*shape, batch_size, action_dim] or dict,
                log probabilities [*shape, batch_size, action_dim]).
    """
    samples = self.distribution.rsample(shape)  # [*shape, batch_size, action_dim]
    log_probs = self.distribution.log_prob(samples)  # [*shape, batch_size, action_dim]
    samples = space.unpack_tensors(self.action_space, samples)
    return samples, log_probs

  def log_prob(self, samples: agent.Action) -> torch.Tensor:
    """Compute log probability of samples.
    
    Args:
      samples: Samples [batch_size, action_dim] or dict of tensors.
      
    Returns:
      Log probabilities [batch_size, action_dim].
    """
    if isinstance(samples, dict):
      samples = space.pack_tensors(self.action_space, samples)
    return self.distribution.log_prob(samples)  # [batch_size, action_dim]

  def mean(self) -> agent.Action:
    """Mean of the distribution.
    
    Returns:
      Mean [batch_size, action_dim] or dict of tensors.
    """
    mean = self.distribution.mean  # [batch_size, action_dim]
    return space.unpack_tensors(self.action_space, mean)
  
  def std(self) -> torch.Tensor:
    """Standard deviation of the distribution.
    
    Returns:
      Standard deviation [batch_size, action_dim].
    """
    return self.distribution.stddev  # [batch_size, action_dim]
  
  def entropy(self) -> torch.Tensor:
    """Entropy of the distribution.
    
    Returns:
      Entropy [batch_size, action_dim].
    """
    return self.distribution.entropy()  # [batch_size, action_dim]


class SquashedNormalActionDistribution:
  """Tanh-squashed normal distribution for bounded action spaces (SAC)."""

  def __init__(self, action_space: agent.ActionSpace, mean: torch.Tensor, std: torch.Tensor):
    """Initialize the squashed normal distribution.
    
    Args:
      action_space: Action space for unpacking actions.
      mean: Mean of the pre-squash normal distribution [batch_size, action_dim].
      std: Standard deviation of the pre-squash normal [batch_size, action_dim].
    """
    self.action_space = action_space
    self._distribution = torch.distributions.normal.Normal(mean, std)

  def sample(self, shape: tuple[int, ...] = ()) -> agent.Action:
    """Sample without reparameterization (no gradients).
    
    Args:
      shape: Additional sample dimensions to prepend.
      
    Returns:
      Squashed samples [*shape, batch_size, action_dim] or dict of tensors.
    """
    samples = self._distribution.sample(shape)  # [*shape, batch_size, action_dim]
    squashed_samples = torch.tanh(samples)  # [*shape, batch_size, action_dim]
    return space.unpack_tensors(self.action_space, squashed_samples)

  def sample_with_log_prob(self, shape: tuple[int, ...] = ()) -> tuple[agent.Action, torch.Tensor]:
    """Sample without reparameterization and compute log probability.
    
    Args:
      shape: Additional sample dimensions to prepend.
      
    Returns:
      Tuple of (squashed samples [*shape, batch_size, action_dim] or dict,
                log probabilities [*shape, batch_size, action_dim]).
    """
    samples = self._distribution.sample(shape)  # [*shape, batch_size, action_dim]
    squashed_samples = torch.tanh(samples)  # [*shape, batch_size, action_dim]
    log_probs = self._distribution.log_prob(samples)  # [*shape, batch_size, action_dim]
    # Jacobian correction for tanh transformation.
    log_probs -= torch.log(1 - squashed_samples ** 2 + 1e-6)  # [*shape, batch_size, action_dim]
    squashed_samples = space.unpack_tensors(self.action_space, squashed_samples)
    return squashed_samples, log_probs

  def rsample(self, shape: tuple[int, ...] = ()) -> agent.Action:
    """Sample with reparameterization (for gradient estimation).
    
    Args:
      shape: Additional sample dimensions to prepend.
      
    Returns:
      Squashed samples [*shape, batch_size, action_dim] or dict of tensors.
    """
    samples = self._distribution.rsample(shape)  # [*shape, batch_size, action_dim]
    squashed_samples = torch.tanh(samples)  # [*shape, batch_size, action_dim]
    return space.unpack_tensors(self.action_space, squashed_samples)

  def rsample_with_log_prob(self, shape: tuple[int, ...] = ()) -> tuple[agent.Action, torch.Tensor]:
    """Sample with reparameterization and compute log probability.
    
    Args:
      shape: Additional sample dimensions to prepend.
      
    Returns:
      Tuple of (squashed samples [*shape, batch_size, action_dim] or dict,
                log probabilities [*shape, batch_size, action_dim]).
    """
    samples = self._distribution.rsample(shape)  # [*shape, batch_size, action_dim]
    squashed_samples = torch.tanh(samples)  # [*shape, batch_size, action_dim]
    log_probs = self._distribution.log_prob(samples)  # [*shape, batch_size, action_dim]
    # Jacobian correction for tanh transformation.
    log_probs -= torch.log(1 - squashed_samples ** 2 + 1e-6)  # [*shape, batch_size, action_dim]
    squashed_samples = space.unpack_tensors(self.action_space, squashed_samples)
    return squashed_samples, log_probs

  def log_prob(self, samples: agent.Action) -> torch.Tensor:
    """Not implemented, since unsquashed samples cannot be accurately recovered."""
    raise NotImplementedError(
      'Unsquashed samples cannot be accurately recovered. Use `rsample_with_log_prob` directly.'
    )

  def mean(self) -> agent.Action:
    """Mean of the squashed distribution.
    
    Returns:
      Squashed mean [batch_size, action_dim] or dict of tensors.
    """
    mean = torch.tanh(self._distribution.mean)  # [batch_size, action_dim]
    return space.unpack_tensors(self.action_space, mean)
  
  def std(self) -> torch.Tensor:
    """Not implemented, since standard deviation is not defined for squashed distributions."""
    raise NotImplementedError('Standard deviation is not defined for squashed distributions.')
  
  def entropy(self) -> torch.Tensor:
    """Not implemented, since entropy is not defined for squashed distributions."""
    raise NotImplementedError('Entropy is not defined for squashed distributions.')
  

# ==================================================================================================
# Heads

class StochasticDetachedStdPolicyHead(torch.nn.Module):
  """Stochastic policy with input-dependent mean (network) and input-independent standard deviation
  (vector). Each action dimension has its own independent mean and standard deviation."""

  def __init__(
    self,
    mean_activation: T.Callable[[], torch.nn.Module] = torch.nn.Tanh,
    mean_init_fn: T.Callable[[torch.nn.Module], None] | None = None,
    std_log_init: float = 0.,
    std_min: float = 1e-4,
    std_max: float = 1.,
    distribution: T.Type[ActionDistribution[torch.Tensor, torch.Tensor]] = NormalActionDistribution,
  ):
    """Initialize the policy head.
    
    Args:
      mean_activation: Activation function for the mean layer (e.g., Tanh for bounded actions).
      mean_init_fn: Optional initialization function for the mean layer.
      std_log_init: Initial value for log(std) parameter.
      std_min: Minimum allowed std (for numerical stability).
      std_max: Maximum allowed std.
      distribution: Action distribution constructor (e.g. `Normal` or `SquashedNormal`).
    """
    super().__init__()
    self.mean_activation = mean_activation
    self.mean_init_fn = mean_init_fn
    self.std_log_init = std_log_init
    self.std_min = std_min
    self.std_max = std_max
    self.distribution = distribution

  def initialize(self, input_size: int, action_space: agent.ActionSpace) -> None:
    """Initialize layers.
    
    Args:
      input_size: Dimension of input features.
      action_space: `Box` or `Dict` action space.
    """
    self.action_space = action_space
    if isinstance(action_space, gym.spaces.Dict):
      action_space = space.pack_space(action_space)
    assert action_space.shape is not None
    action_size = math.prod(action_space.shape)
    self.mean_layer = torch.nn.Sequential(
      torch.nn.Linear(input_size, action_size),
      self.mean_activation(),
    )
    if self.mean_init_fn:
      self.mean_layer.apply(self.mean_init_fn)
    # Std is a learnable parameter, same for all batch elements.
    self.std_log = torch.nn.Parameter(
      torch.as_tensor([[self.std_log_init] * action_size], dtype=torch.float32)
    )  # [1, action_size]

  def forward(self, inputs: torch.Tensor) -> ActionDistribution:
    """Compute action distribution.
    
    Args:
      inputs: Input features [batch_size, input_size].
      
    Returns:
      Action distribution with mean [batch_size, action_size] and std [batch_size, action_size].
    """
    batch_size = inputs.shape[0]
    mean = self.mean_layer(inputs)  # [batch_size, action_size]
    std = torch.nn.functional.softplus(self.std_log) + FLOAT_EPSILON  # [1, action_size]
    std = torch.clamp(std, self.std_min, self.std_max)  # [1, action_size]
    std = std.repeat(batch_size, 1)  # [batch_size, action_size]
    return self.distribution(self.action_space, mean, std)


class StochasticPolicyHead(torch.nn.Module):
  """Stochastic policy with input-dependent mean and standard deviation (both networks). Each action
  dimension has its own independent mean and standard deviation."""

  def __init__(
    self,
    mean_activation: T.Callable[[], torch.nn.Module] = torch.nn.Tanh,
    mean_init_fn: T.Callable[[torch.nn.Module], None] | None = None,
    std_activation: T.Callable[[], torch.nn.Module] = torch.nn.Softplus,
    std_min: float = 1e-4,
    std_max: float = 1,
    std_init_fn: T.Callable[[torch.nn.Module], None] | None = None,
    distribution: T.Type[ActionDistribution[torch.Tensor, torch.Tensor]] = NormalActionDistribution,
  ):
    """Initialize the policy head.
    
    Args:
      mean_activation: Activation function for the mean layer.
      mean_init_fn: Optional initialization function for the mean layer.
      std_activation: Activation function for the std layer (e.g., `Softplus` for positivity).
      std_min: Minimum allowed std (for numerical stability).
      std_max: Maximum allowed std.
      std_init_fn: Optional initialization function for the std layer.
      distribution: Action distribution constructor (e.g. `Normal` or `SquashedNormal`).
    """
    super().__init__()
    self.mean_activation = mean_activation
    self.mean_init_fn = mean_init_fn
    self.std_activation = std_activation
    self.std_min = std_min
    self.std_max = std_max
    self.std_init_fn = std_init_fn
    self.distribution = distribution

  def initialize(self, input_size: int, action_space: agent.ActionSpace) -> None:
    """Initialize layers.
    
    Args:
      input_size: Dimension of input features.
      action_space: `Box` or `Dict` action space.
    """
    self.action_space = action_space
    if isinstance(action_space, gym.spaces.Dict):
      action_space = space.pack_space(action_space)
    assert action_space.shape is not None
    action_size = math.prod(action_space.shape)
    self.mean_layer = torch.nn.Sequential(
      torch.nn.Linear(input_size, action_size),
      self.mean_activation(),
    )
    if self.mean_init_fn:
      self.mean_layer.apply(self.mean_init_fn)
    self.std_layer = torch.nn.Sequential(
      torch.nn.Linear(input_size, action_size),
      self.std_activation(),
    )
    if self.std_init_fn:
      self.std_layer.apply(self.std_init_fn)

  def forward(self, inputs: torch.Tensor) -> ActionDistribution:
    """Compute action distribution.
    
    Args:
      inputs: Input features [batch_size, input_size].
      
    Returns:
      Action distribution with mean and std [batch_size, action_size].
    """
    mean = self.mean_layer(inputs)  # [batch_size, action_size]
    std = self.std_layer(inputs)  # [batch_size, action_size]
    std = torch.clamp(std, self.std_min, self.std_max)  # [batch_size, action_size]
    return self.distribution(self.action_space, mean, std)


class DeterministicPolicyHead(torch.nn.Module):
  """Deterministic policy for DDPG-style algorithms."""

  def __init__(
    self,
    activation: T.Callable[[], torch.nn.Module] = torch.nn.Tanh,
    bias: bool = True,
    init_fn: T.Callable[[torch.nn.Module], None] | None = None,
  ):
    """Initialize the policy head.
    
    Args:
      activation: Activation function for the action layer (e.g., Tanh for bounded actions).
      bias: Whether to use bias in the linear layer.
      init_fn: Optional initialization function for the layer.
    """
    super().__init__()
    self.activation = activation
    self.bias = bias
    self.init_fn = init_fn

  def initialize(self, input_size: int, action_space: agent.ActionSpace) -> None:
    """Initialize the action layer.
    
    Args:
      input_size: Input features dimension.
      action_space: `Box` or `Dict` action space.
    """
    self.action_space = action_space
    if isinstance(action_space, gym.spaces.Dict):
      action_space = space.pack_space(action_space)
    action_size = math.prod(action_space.shape)
    self.action_layer = torch.nn.Sequential(
      torch.nn.Linear(input_size, action_size, bias=self.bias),
      self.activation(),
    )
    if self.init_fn:
      self.action_layer.apply(self.init_fn)

  def forward(self, inputs: torch.Tensor) -> agent.Action:
    """Compute deterministic action.
    
    Args:
      inputs: Input features [batch_size, input_size].
      
    Returns:
      Actions tensors or dict of tensors, [batch_size, action_size].
    """
    actions = self.action_layer(inputs)  # [batch_size, action_size]
    # Reshape box actions. Unpack and reshape dict actions.
    actions = space.unpack_tensors(self.action_space, actions)
    return actions


# ==================================================================================================
# Actor

class ActorEncoder(T.Protocol):
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


class ActorTorso(T.Protocol):
  def initialize(self, input_size: int) -> int:
    ...

  def forward(self, inputs: T.Any) -> T.Any:
    ...

  def __call__(self, inputs: T.Any) -> T.Any:
    ...


class ActorHead(T.Protocol):
  def initialize(self, input_size: int, action_space: agent.ActionSpace) -> None:
    ...

  def forward(self, inputs: T.Any) -> T.Any:
    ...

  def __call__(self, inputs: T.Any) -> T.Any:
    ...


class ActorLike(T.Protocol):
  def initialize(
    self,
    observation_space: agent.ObservationSpace,
    action_space: agent.ActionSpace,
    observation_normalizer: normalizers.ObservationNormalizer | None = None,
  ) -> None:
    ...

  def reset(self) -> None:
    ...

  def forward(self, *inputs: torch.Tensor | dict[str, torch.Tensor]) -> T.Any:
    ...

  def __call__(self, *inputs: torch.Tensor | dict[str, torch.Tensor]) -> T.Any:
    ...


class Actor(torch.nn.Module):
  """Actor that uses `Box` or `Dict` observation and action spaces."""
  
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
    observation_space: agent.ObservationSpace,
    action_space: agent.ActionSpace,
    observation_normalizer: normalizers.ObservationNormalizer | None = None,
  ) -> None:
    size = self.encoder.initialize(observation_space, action_space, observation_normalizer)
    if self.torso is not None:
      size = self.torso.initialize(size)
    self.head.initialize(size, action_space)

  def reset(self) -> None:
    pass
  
  def forward(self, *inputs: torch.Tensor | dict[str, torch.Tensor]) -> T.Any:
    out = self.encoder(*inputs)
    if self.torso is not None:
      out = self.torso(out)
    return self.head(out)
