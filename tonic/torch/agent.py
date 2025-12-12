import abc
import os
import random

import numpy as np
import torch
import gym.spaces

from tonic import logger


Observation = torch.Tensor | dict[str, torch.Tensor]
ObservationSpace = gym.spaces.Box | gym.spaces.Dict
Action = torch.Tensor | dict[str, torch.Tensor]
ActionSpace = gym.spaces.Box | gym.spaces.Dict


class Agent(abc.ABC):
  model: torch.nn.Module

  def initialize(
    self,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
    seed: int,
  ) -> None:
    self.observation_space = observation_space
    self.action_space = action_space
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

  @abc.abstractmethod
  def step(self, observations: Observation) -> Action:
    """Returns actions during training."""
    pass

  @abc.abstractmethod
  def test_step(self, observations: Observation) -> Action:
    """Returns actions during testing."""
    pass

  def record(
    self,
    observations: Observation,
    actions: Action,
    rewards: torch.Tensor,
    resets: torch.Tensor,
    terminations: torch.Tensor,
    next_observations: Observation,
  ) -> None:
    """Informs the agent of the latest transitions during training."""
    pass

  def update(self) -> None:
    """Updates the parameters of the agent during training."""
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
