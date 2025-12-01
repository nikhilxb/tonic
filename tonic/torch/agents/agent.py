import abc
import os
import random

import numpy as np
import torch

from tonic import logger


Observation = torch.Tensor | dict[str, torch.Tensor]
Action = torch.Tensor | dict[str, torch.Tensor]


class Agent(abc.ABC):
  model: torch.nn.Module

  def initialize(self, seed: int):
    if seed is not None:
      np.random.seed(seed)
      random.seed(seed)
      torch.manual_seed(seed)

  @abc.abstractmethod
  def step(
    self,
    observations: Observation,
    steps: int,
  ) -> Action:
    """Returns actions during training."""
    pass

  def update(
    self,
    observations: Observation,
    rewards: torch.Tensor,
    resets: torch.Tensor,
    terminations: torch.Tensor,
    steps: int,
  ) -> None:
    """Informs the agent of the latest transitions during training."""
    pass

  @abc.abstractmethod
  def test_step(
    self,
    observations: Observation,
    steps: int,
  ) -> Action:
    """Returns actions during testing."""
    pass

  def test_update(
    self,
    observations: Observation,
    rewards: torch.Tensor,
    resets: torch.Tensor,
    terminations: torch.Tensor,
    steps: int,
  ) -> None:
    """Informs the agent of the latest transitions during testing."""
    pass

  def save(self, path: str) -> None:
    """Saves the agent weights to a checkpoint."""
    path = path + '.pt'
    logger.log(f'\nSaving weights to {path}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(self.model.state_dict(), path)

  def load(self, path: str) -> None:
    """Loads the agent weights from a checkpoint."""
    path = path + '.pt'
    logger.log(f'\nLoading weights from {path}')
    self.model.load_state_dict(torch.load(path))
