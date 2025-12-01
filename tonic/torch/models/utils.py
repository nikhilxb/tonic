import collections
import functools
import typing as T

import gym.spaces
import gym.spaces.utils
import numpy as np
import torch


def trainable_variables(model: torch.nn.Module) -> list[torch.nn.Parameter]:
  """Returns the parameters of the `torch.nn.Module` with `requires_grad=True`."""
  return [p for p in model.parameters() if p.requires_grad]


# ==================================================================================================
# Packing

V = T.TypeVar('V')

def pack_space(space: gym.spaces.Space[V]) -> gym.spaces.Box:
  return gym.spaces.utils.flatten_space(space)

def pack_ndarrays(space: gym.spaces.Space[V], value: V) -> np.ndarray:
  return gym.spaces.utils.flatten(space, value)

def unpack_ndarrays(space: gym.spaces.Space[V], flat: np.ndarray) -> V:
  return gym.spaces.utils.unflatten(space, flat)

@functools.singledispatch
def pack_tensors(space: gym.spaces.Space[V], value: V) -> torch.Tensor:
  """Pack a pytree of tensors (from  `Space`) into a single 1D tensor (into `Box`).
  
  Args:
    space: Structured space.
    value: Structured pytree of tensors to pack.
    
  Returns:
    Packed 1D tensor.
  """
  raise NotImplementedError(f'Unknown space: `{space}`')

@pack_tensors.register(gym.spaces.Box)
def _pack_tensors_box(space: gym.spaces.Box, value: torch.Tensor) -> torch.Tensor:
  return value.flatten()

@pack_tensors.register(gym.spaces.Dict)
def _pack_tensors_dict(space: gym.spaces.Dict, value: T.Mapping[str, torch.Tensor]) -> torch.Tensor:
  return torch.cat([pack_tensors(sp, value[key]) for key, sp in space.spaces.items()])

@functools.singledispatch
def unpack_tensors(space: gym.spaces.Space[V], value: torch.Tensor) -> V:
  """Unpack a single 1D tensor (from `Box`) into a pytree of tensors (into `Space`).
  
  Args:
    space: Structured space.
    value: Packed 1D tensor to unpack.
    
  Returns:
    Unpacked pytree of tensors.
  """
  raise NotImplementedError(f'Unknown space: `{space}`')

@unpack_tensors.register(gym.spaces.Box)
def _unpack_tensors_box(space: gym.spaces.Box, value: torch.Tensor) -> torch.Tensor:
  return value.reshape(space.shape)

@unpack_tensors.register(gym.spaces.Dict)
def _unpack_tensors_dict(space: gym.spaces.Dict, value: torch.Tensor) -> collections.OrderedDict[str, T.Any]:
  chunks = torch.split(value, [gym.spaces.utils.flatdim(sp) for sp in space.spaces.values()])
  return collections.OrderedDict([
    (key, unpack_tensors(sp, chunk)) for chunk, (key, sp) in zip(chunks, space.spaces.items())
  ])
