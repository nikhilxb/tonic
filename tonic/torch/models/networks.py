import typing as T

import torch


class MLP(torch.nn.Module):
  """Multi-layer perceptron with configurable layers and activation functions."""

  def __init__(
    self,
    sizes: T.Sequence[int],
    activation: T.Callable[[], torch.nn.Module],
    fn: T.Callable[[torch.nn.Module], None] | None = None,
  ):
    """
    Args:
      sizes: Hidden layer sizes (e.g., `[256, 256]` for two hidden layers).
      activation: Activation function factory (e.g., `torch.nn.ReLU`).
      fn: Optional initialization function applied to all layers.
    """
    super().__init__()
    self.sizes = list(sizes)
    self.activation = activation
    self.fn = fn

  def initialize(self, input_size: int) -> int:
    """
    Args:
      input_size: Input feature dimension.
      
    Returns:
      Output feature dimension (size of last hidden layer).
    """
    sizes = [input_size] + self.sizes
    layers = []
    for i in range(len(sizes) - 1):
      layers.extend([
        torch.nn.Linear(sizes[i], sizes[i + 1]),
        self.activation(),
      ])
    self.layers = torch.nn.Sequential(*layers)
    if self.fn is not None:
      self.layers.apply(self.fn)
    return sizes[-1]

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """
    Args:
      inputs: Input tensor `[batch_size, input_size]`.
      
    Returns:
      Output tensor `[batch_size, output_size]`.
    """
    return self.layers(inputs)
