import typing as T

import torch
import gym.spaces

from .. import agent, space
from . import normalizers


class ObservationEncoder(torch.nn.Module):
  """Encoder for observations: normalize, pack."""

  def __init__(self, observation_prefix: str = ''):
    """
    Args:
      observation_prefix: Prefix of observation keys to encode (`Dict` only). Default: Encode all.
    """
    super().__init__()
    self.observation_prefix = observation_prefix

  def initialize(
    self,
    observation_space: agent.ObservationSpace,
    action_space: agent.ActionSpace,
    observation_normalizer: normalizers.ObservationNormalizer | None = None,
  ) -> int:
    """
    Args:
      observation_space: Observation space.
      action_space: Action space (unused).
      observation_normalizer: Observation normalizer, optional.
      
    Returns:
      Observation vector size.
    """
    self.observation_normalizer = observation_normalizer
    
    if isinstance(observation_space, gym.spaces.Box):
      assert len(observation_space.shape) == 1, 'Observation must be 1D.'
      self.observation_space = None
      observation_size = observation_space.shape[0]
    elif isinstance(observation_space, gym.spaces.Dict):
      assert all(isinstance(o, gym.spaces.Box) for o in observation_space.spaces.values())
      self.observation_space = gym.spaces.Dict({
        k: v for k, v in observation_space.spaces.items() if k.startswith(self.observation_prefix)
      })
      observation_space_box = space.pack_space(self.observation_space)
      observation_size = observation_space_box.shape[0]
    else:
      raise TypeError(f"Unsupported observation space type: {type(observation_space)}")
    
    return observation_size

  @T.overload
  def forward(self, observations: agent.Observation, /) -> torch.Tensor:
    ...
  @T.overload
  def forward(self, *inputs: torch.Tensor | dict[str, torch.Tensor]) -> T.NoReturn:
    ...
  def forward(self, *inputs) -> torch.Tensor:
    """
    Args:
      inputs: Observations `[batch, observation_size]`.
      
    Returns:
      Normalized observations `[batch, observation_size]`.
    """
    observations, = inputs
    
    if self.observation_space is None:
      # Box observations
      if self.observation_normalizer:
        observations = self.observation_normalizer(observations)  # [batch, obs]
      observations = T.cast(torch.Tensor, observations)
    else:
      # Dict observations
      observations = {
        k: v for k, v in observations.items() if k.startswith(self.observation_prefix)
      }
      if self.observation_normalizer:
        observations = self.observation_normalizer(observations)  # {key: [batch, ...]}
      observations = space.pack_tensors(self.observation_space, observations)  # [batch, obs] 
    
    return observations


class ObservationActionEncoder(torch.nn.Module):
  """Encoder for observations and actions: normalize, pack, concatenate."""

  def __init__(self, observation_prefix: str = ''):
    """
    Args:
      observation_prefix: Prefix of observation keys to encode (Dict only). Default: Encode all.
    """
    super().__init__()
    self.observation_prefix = observation_prefix

  def initialize(
    self,
    observation_space: agent.ObservationSpace,
    action_space: agent.ActionSpace,
    observation_normalizer: normalizers.ObservationNormalizer | None = None,
  ) -> int:
    """
    Args:
      observation_space: `Box` or `Dict` observation space.
      action_space: `Box` or `Dict` action space.
      observation_normalizer: Observation normalizer, optional.
      
    Returns:
      Concatenated observation-action vector size.
    """
    self.observation_normalizer = observation_normalizer
    
    # Validate observation space.
    if isinstance(observation_space, gym.spaces.Box):
      assert len(observation_space.shape) == 1, 'Observation must be 1D.'
      self.observation_space = None
      observation_size = observation_space.shape[0]
    elif isinstance(observation_space, gym.spaces.Dict):
      assert all(isinstance(o, gym.spaces.Box) for o in observation_space.spaces.values())
      self.observation_space = gym.spaces.Dict({
        k: v for k, v in observation_space.spaces.items() if k.startswith(self.observation_prefix)
      })
      observation_space_box = space.pack_space(self.observation_space)
      observation_size = observation_space_box.shape[0]
    else:
      raise TypeError(f"Unsupported observation space type: {type(observation_space)}")
    
    # Validate action space.
    if isinstance(action_space, gym.spaces.Box):
      assert len(action_space.shape) == 1, 'Action must be 1D.'
      self.action_space = None
      action_size = action_space.shape[0]
    elif isinstance(action_space, gym.spaces.Dict):
      assert all(isinstance(a, gym.spaces.Box) for a in action_space.spaces.values())
      self.action_space = action_space
      action_space_box = space.pack_space(self.action_space)
      action_size = action_space_box.shape[0]
    else:
      raise TypeError(f"Unsupported action space type: {type(action_space)}")
    
    return observation_size + action_size

  @T.overload
  def forward(self, observations: agent.Observation, actions: agent.Action, /) -> torch.Tensor:
    ...
  @T.overload
  def forward(self, *inputs: torch.Tensor | dict[str, torch.Tensor]) -> T.NoReturn:
    ...
  def forward(self, *inputs) -> torch.Tensor:
    """
    Args:
      inputs: Box observations and actions `[batch, ...]` or Dict observations and actions `{key: [batch, ...]}`.
      
    Returns:
      Normalized concatenated tensor `[batch, observation_size + action_size]`.
    """
    observations, actions = inputs
    
    if self.observation_space is None:
      # Box observations
      if self.observation_normalizer:
        observations = self.observation_normalizer(observations)  # [batch, obs]
      observations = T.cast(torch.Tensor, observations)
    else:
      # Dict observations
      observations = {
        k: v for k, v in observations.items() if k.startswith(self.observation_prefix)
      }
      if self.observation_normalizer:
        observations = self.observation_normalizer(observations)  # {key: [batch, ...]}
      observations = space.pack_tensors(self.observation_space, observations)  # [batch, obs]
    
    if self.action_space is None:
      # Box actions
      actions = T.cast(torch.Tensor, actions)
    else:
      # Dict actions
      actions = space.pack_tensors(self.action_space, actions)  # [batch, act]
    
    return torch.cat([observations, actions], dim=-1)  # [batch, obs + act]
