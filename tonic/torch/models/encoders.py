import typing as T

import torch
import gym.spaces

from . import normalizers, utils


# ==================================================================================================
# Box encoders


class BoxObservationEncoder(torch.nn.Module):
  """Encoder for `Box` observations: normalize."""

  def initialize(
    self,
    observation_space: gym.spaces.Box | gym.spaces.Dict,
    action_space: gym.spaces.Box | gym.spaces.Dict,
    observation_normalizer: normalizers.ObservationNormalizer | None = None,
  ) -> int:
    """
    Args:
      observation_space: `Box` observation space.
      action_space: `Box` action space (unused).
      observation_normalizer: `Box` observation normalizer, optional.
      
    Returns:
      Observation vector size.
    """
    assert isinstance(observation_space, gym.spaces.Box)
    assert len(observation_space.shape) == 1, 'Observation must be 1D.'
    self.observation_normalizer = observation_normalizer
    observation_size = observation_space.shape[0]
    return observation_size

  @T.overload
  def forward(self, observations: torch.Tensor, /) -> torch.Tensor:
    ...
  @T.overload
  def forward(self, *inputs: torch.Tensor | dict[str, torch.Tensor]) -> T.NoReturn:
    ...
  def forward(self, *inputs) -> torch.Tensor:
    """
    Args:
      inputs: Observations `[batch_size, observation_size]`.
      
    Returns:
      Normalized observations `[batch_size, observation_size]`.
    """
    observations, = inputs
    if self.observation_normalizer:
      observations = self.observation_normalizer(observations)  # [batch, obs]
    return observations


class BoxObservationActionEncoder(torch.nn.Module):
  """Encoder for `Box` observations and actions: normalize, concatenate."""

  def initialize(
    self,
    observation_space: gym.spaces.Box | gym.spaces.Dict,
    action_space: gym.spaces.Box | gym.spaces.Dict,
    observation_normalizer: normalizers.ObservationNormalizer | None = None,
  ) -> int:
    """
    Args:
      observation_space: `Box` observation space.
      action_space: `Box` action space.
      observation_normalizer: `Box` observation normalizer, optional.
      
    Returns:
      Concatenated observation-action vector size.
    """
    assert isinstance(observation_space, gym.spaces.Box)
    assert isinstance(action_space, gym.spaces.Box)
    assert len(observation_space.shape) == 1, 'Observation must be 1D.'
    assert len(action_space.shape) == 1, 'Action must be 1D.'
    self.observation_normalizer = observation_normalizer
    observation_size = observation_space.shape[0]
    action_size = action_space.shape[0]
    return observation_size + action_size

  @T.overload
  def forward(self, observations: torch.Tensor, actions: torch.Tensor, /) -> torch.Tensor:
    ...
  @T.overload
  def forward(self, *inputs: torch.Tensor | dict[str, torch.Tensor]) -> T.NoReturn:
    ...
  def forward(self, *inputs) -> torch.Tensor:
    """
    Args:
      inputs: Observations `[batch_size, observation_size]`. Actions `[batch_size, action_size]`.
      
    Returns:
      Concatenated tensor `[batch_size, observation_size + action_size]`.
    """
    observations, actions = inputs
    if self.observation_normalizer:
      observations = self.observation_normalizer(observations)  # [batch, obs]
    return torch.cat([observations, actions], dim=-1)  # [batch, obs + act]


# ==================================================================================================
# Dict encoders


class DictObservationEncoder(torch.nn.Module):
  """Encoder for `Dict` observations: normalize, flatten."""

  def __init__(self, observation_prefix: str = ""):
    """
    Args:
      observation_prefix: Prefix of observation keys to encode. Default: Encode all observations.
    """
    super().__init__()
    self.observation_prefix = observation_prefix

  def initialize(
    self,
    observation_space: gym.spaces.Box | gym.spaces.Dict,
    action_space: gym.spaces.Box | gym.spaces.Dict,
    observation_normalizer: normalizers.ObservationNormalizer | None = None,
  ) -> int:
    """
    Args:
      observation_space: `Dict` observation space.
      action_space: `Dict` action space (unused).
      observation_normalizer: `Dict` observation normalizer, optional.
      
    Returns:
      Flattened observation vector size.
    """
    assert isinstance(observation_space, gym.spaces.Dict)
    assert all(isinstance(o, gym.spaces.Box) for o in observation_space.spaces.values())
    self.observation_normalizer = observation_normalizer
    self.observation_space = gym.spaces.Dict({
      k: v for k, v in observation_space.spaces.items() if k.startswith(self.observation_prefix)
    })
    observation_space_flat = utils.pack_space(self.observation_space)
    observation_size = observation_space_flat.shape[0]
    return observation_size

  @T.overload
  def forward(self, observations: dict[str, torch.Tensor], /) -> torch.Tensor:
    ...
  @T.overload
  def forward(self, *inputs: torch.Tensor | dict[str, torch.Tensor]) -> T.NoReturn:
    ...
  def forward(self, *inputs) -> torch.Tensor:
    """
    Args:
      inputs: `Dict` observations `{key: [batch_size, ...]}`.
      
    Returns:
      Flattened observations `[batch_size, observation_size]`.
    """
    observations = T.cast(dict[str, torch.Tensor], inputs[0])
    observations = {
      k: v for k, v in observations.items() if k.startswith(self.observation_prefix)
    }
    if self.observation_normalizer:
      observations = self.observation_normalizer(observations)  # {key: [batch, ...]}
    observations_flat = utils.pack_tensors(self.observation_space, observations)  # [batch, obs]
    return observations_flat


class DictObservationActionEncoder(torch.nn.Module):
  """Encoder for `Dict` observations and actions: normalize, flatten, concatenate."""

  def __init__(self, observation_prefix: str = ""):
    """
    Args:
      observation_prefix: Prefix of observation keys to encode. Default: Encode all observations.
    """
    super().__init__()
    self.observation_prefix = observation_prefix

  def initialize(
    self,
    observation_space: gym.spaces.Dict,
    action_space: gym.spaces.Dict,
    observation_normalizer: normalizers.DictNormalizer | None = None,
  ) -> int:
    """
    Args:
      observation_space: `Dict` observation space.
      action_space: `Dict` action space.
      observation_normalizer: `Dict` observation normalizer, optional.
      
    Returns:
      Concatenated observation-action vector size.
    """
    assert isinstance(observation_space, gym.spaces.Dict)
    assert isinstance(action_space, gym.spaces.Dict)
    assert all(isinstance(o, gym.spaces.Box) for o in observation_space.spaces.values())
    assert all(isinstance(a, gym.spaces.Box) for a in action_space.spaces.values())
    self.observation_normalizer = observation_normalizer
    self.observation_space = gym.spaces.Dict({
      k: v for k, v in observation_space.spaces.items() if k.startswith(self.observation_prefix)
    })
    self.action_space = action_space
    observation_space_flat = utils.pack_space(self.observation_space)
    action_space_flat = utils.pack_space(self.action_space)
    observation_size = observation_space_flat.shape[0]
    action_size = action_space_flat.shape[0]
    return observation_size + action_size

  @T.overload
  def forward(
    self,
    observations: dict[str, torch.Tensor],
    actions: dict[str, torch.Tensor],
    /,
  ) -> torch.Tensor:
    ...
  @T.overload
  def forward(self, *inputs: torch.Tensor | dict[str, torch.Tensor]) -> T.NoReturn:
    ...
  def forward(self, *inputs) -> torch.Tensor:
    """
    Args:
      inputs: Dict observations `{key: [batch_size, ...]}` and actions `{key: [batch_size, ...]}`.
      
    Returns:
      Concatenated tensor `[batch_size, observation_size + action_size]`.
    """
    observations, actions = T.cast(tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]], inputs)
    observations = {
      k: v for k, v in observations.items() if k.startswith(self.observation_prefix)
    }
    if self.observation_normalizer:
      observations = self.observation_normalizer(observations)  # {key: [batch, ...]}
    observations_flat = utils.pack_tensors(self.observation_space, observations)  # [batch, obs]
    actions_flat = utils.pack_tensors(self.action_space, actions)  # [batch, act]
    return torch.cat([observations_flat, actions_flat], dim=-1)  # [batch, obs + act]
