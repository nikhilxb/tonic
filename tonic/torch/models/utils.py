import collections
import functools
import typing as T

import gym.spaces
import gym.spaces.utils
import numpy as np
import torch


def trainable_variables(model: torch.nn.Module) -> list[torch.nn.Parameter]:
  return [p for p in model.parameters() if p.requires_grad]


# ==================================================================================================
# Flattening

V = T.TypeVar('V')

def flatten_space(space: gym.spaces.Space[V]) -> gym.spaces.Box:
  return gym.spaces.utils.flatten_space(space)

def flatten_ndarrays(space: gym.spaces.Space[V], value: V) -> np.ndarray:
  return gym.spaces.utils.flatten(space, value)

def unflatten_ndarrays(space: gym.spaces.Space[V], flat: np.ndarray) -> V:
  return gym.spaces.utils.unflatten(space, flat)

@functools.singledispatch
def flatten_tensors(space: gym.spaces.Space[V], value: V) -> torch.Tensor:
  """Flatten a data point from a space.

  This is useful when e.g. points from spaces must be passed to a neural
  network, which only understands flat arrays of floats.

  Accepts a space and a point from that space. Always returns a 1D array.
  Raises ``NotImplementedError`` if the space is not defined in
  ``gym.spaces``.
  """
  raise NotImplementedError(f'Unknown space: `{space}`')

@flatten_tensors.register(gym.spaces.Box)
def _flatten_tensors_box(space: gym.spaces.Box, value: torch.Tensor) -> torch.Tensor:
  return value.flatten()

@flatten_tensors.register(gym.spaces.Dict)
def _flatten_tensors_dict(space: gym.spaces.Dict, value: T.Mapping[str, torch.Tensor]) -> torch.Tensor:
  return torch.cat([flatten_tensors(sp, value[key]) for key, sp in space.spaces.items()])

@functools.singledispatch
def unflatten_tensors(space: gym.spaces.Space[V], value: torch.Tensor) -> V:
  """Unflatten a data point from a space.

  This is useful when e.g. points from spaces must be passed to a neural
  network, which only understands flat arrays of floats.

  Accepts a space and a flattened point. Returns a point with a structure
  that matches the space. Raises ``NotImplementedError`` if the space is not
  defined in ``gym.spaces``.
  """
  raise NotImplementedError(f'Unknown space: `{space}`')

@unflatten_tensors.register(gym.spaces.Box)
def _unflatten_tensors_box(space: gym.spaces.Box, value: torch.Tensor) -> torch.Tensor:
  return value.reshape(space.shape)

@unflatten_tensors.register(gym.spaces.Dict)
def _unflatten_tensors_dict(space: gym.spaces.Dict, value: torch.Tensor) -> collections.OrderedDict[str, T.Any]:
  chunks = torch.split(value, [gym.spaces.utils.flatdim(sp) for sp in space.spaces.values()])
  return collections.OrderedDict([
    (key, unflatten_tensors(sp, chunk)) for chunk, (key, sp) in zip(chunks, space.spaces.items())
  ])
