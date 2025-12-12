import typing as T

import torch


@T.overload
def map_tensors(
  fn: T.Callable[[torch.Tensor], torch.Tensor],
  x: torch.Tensor,
) -> torch.Tensor: ...
@T.overload
def map_tensors(
  fn: T.Callable[[torch.Tensor], torch.Tensor],
  x: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]: ...
@T.overload
def map_tensors(
  fn: T.Callable[..., torch.Tensor],
  x: torch.Tensor,
  *args: T.Any,
) -> torch.Tensor: ...
@T.overload
def map_tensors(
  fn: T.Callable[..., torch.Tensor],
  x: dict[str, torch.Tensor],
  *args: T.Any,
) -> dict[str, torch.Tensor]: ...
def map_tensors(fn, x, *args):
  """Apply a function to tensor(s), preserving dict structure if present.
  
  Args:
    fn: Function to apply to each tensor. Takes a tensor as first argument, 
        followed by any additional args.
    x: Single tensor or dict of tensors to transform.
    *args: Additional arguments to pass to fn after each tensor.
    
  Returns:
    Transformed tensor or dict of tensors with the same structure as x.
    
  Examples:
    >>> map_tensors(lambda t: t * 2, tensor)
    >>> map_tensors(lambda t: t * 2, {'a': tensor1, 'b': tensor2})
    >>> map_tensors(lambda t, scale: t * scale, tensor, 2.0)
    >>> map_tensors(lambda t, scale: t * scale, {'a': tensor1, 'b': tensor2}, 2.0)
  """
  if isinstance(x, dict):
    return {k: fn(v, *args) for k, v in x.items()}
  else:
    return fn(x, *args)
    

def tile_dim0(x: torch.Tensor, n: int) -> torch.Tensor:
  """Tile a tensor along a new dimension 0.
  
  Adds a new dimension at the front and repeats the tensor n times along it.
  
  Args:
    x: Input tensor of shape [d1, d2, ...].
    n: Number of times to repeat.
    
  Returns:
    Tiled tensor of shape [n, d1, d2, ...].
    
  Example:
    >>> x = torch.tensor([1, 2, 3])  # shape [3]
    >>> tile_dim0(x, 2)  # shape [2, 3], values [[1, 2, 3], [1, 2, 3]]
  """
  return x[None].repeat([n] + [1] * len(x.shape))


def merge_dim0_dim1(x: torch.Tensor) -> torch.Tensor:
  """Merge the first two dimensions of a tensor.
  
  Args:
    x: Input tensor of shape [d0, d1, d2, ...].
    
  Returns:
    Tensor of shape [d0 * d1, d2, ...].
    
  Example:
    >>> x = torch.randn(2, 3, 4)  # shape [2, 3, 4]
    >>> merge_dim0_dim1(x)  # shape [6, 4]
  """
  return x.view(x.shape[0] * x.shape[1], *x.shape[2:])
