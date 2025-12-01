"""Container modules for actor and critic networks and related utilities.

These modules act as thin containers: they are not invoked directly but expose and manage their
submodules (e.g., actor, critic, target networks, normalizers) to serve 3 purposes:
1. Group related modules (actor, critic) together.
2. Manage shared normalizers (observation, return) across duplicated networks.
3. Create and update target networks for off-policy algorithms.

For consistency, all actor/critic models follow this container pattern even when normalizers are
not shared (e.g., actor-only models) or target networks are unnecessary (e.g., on-policy models).
"""
import copy
import typing as T

import torch
import gym.spaces

from . import actors, critics, normalizers, utils


class ActorOnly(torch.nn.Module):
  """Actor-only model for policy-only algorithms."""

  def __init__(
    self,
    actor: actors.ActorLike,
    observation_normalizer: normalizers.ObservationNormalizer | None = None,
  ):
    super().__init__()
    self.actor = actor
    self.observation_normalizer = observation_normalizer

  def initialize(
    self,
    observation_space: gym.spaces.Box | gym.spaces.Dict,
    action_space: gym.spaces.Box | gym.spaces.Dict,
  ) -> None:
    if self.observation_normalizer:
      self.observation_normalizer.initialize(observation_space)  # type: ignore
    self.actor.initialize(
      observation_space,
      action_space,
      self.observation_normalizer,
    )


class ActorCritic(torch.nn.Module):
  """Actor-critic model for on-policy algorithms (e.g., PPO, A2C)."""

  def __init__(
    self,
    actor: actors.ActorLike,
    critic: critics.CriticLike,
    observation_normalizer: normalizers.ObservationNormalizer | None = None,
    return_normalizer: normalizers.ReturnNormalizer | None = None,
  ):
    super().__init__()
    self.actor = actor
    self.critic = critic
    self.observation_normalizer = observation_normalizer
    self.return_normalizer = return_normalizer

  def initialize(
    self,
    observation_space: gym.spaces.Box | gym.spaces.Dict,
    action_space: gym.spaces.Box | gym.spaces.Dict,
  ) -> None:
    if self.observation_normalizer:
      self.observation_normalizer.initialize(observation_space)  # type: ignore
    self.actor.initialize(
      observation_space,
      action_space,
      self.observation_normalizer,
    )
    self.critic.initialize(
      observation_space,
      action_space,
      self.observation_normalizer,
      self.return_normalizer,
    )


class ActorCriticWithTargets(torch.nn.Module):
  """Actor-critic model with target networks for off-policy algorithms (e.g., DDPG, TD3, SAC)."""

  def __init__(
    self,
    actor: actors.ActorLike,
    critic: critics.CriticLike,
    observation_normalizer: normalizers.ObservationNormalizer | None = None,
    return_normalizer: normalizers.ReturnNormalizer | None = None,
    target_coeff: float = 0.005,
  ):
    super().__init__()
    self.actor = actor
    self.critic = critic
    self.target_actor = copy.deepcopy(actor)
    self.target_critic = copy.deepcopy(critic)
    self.observation_normalizer = observation_normalizer
    self.return_normalizer = return_normalizer
    self.target_coeff = target_coeff

  def initialize(
    self,
    observation_space: gym.spaces.Box | gym.spaces.Dict,
    action_space: gym.spaces.Box | gym.spaces.Dict,
  ) -> None:
    if self.observation_normalizer:
      self.observation_normalizer.initialize(observation_space)  # type: ignore
    self.actor.initialize(
      observation_space,
      action_space,
      self.observation_normalizer,
    )
    self.critic.initialize(
      observation_space,
      action_space,
      self.observation_normalizer,
      self.return_normalizer,
    )
    self.target_actor.initialize(
      observation_space,
      action_space,
      self.observation_normalizer,
    )
    self.target_critic.initialize(
      observation_space,
      action_space,
      self.observation_normalizer,
      self.return_normalizer,
    )
    self.online_variables = [
      *utils.trainable_variables(T.cast(torch.nn.Module, self.actor)),
      *utils.trainable_variables(T.cast(torch.nn.Module, self.critic)),
    ]
    self.target_variables = [
      *utils.trainable_variables(T.cast(torch.nn.Module, self.target_actor)),
      *utils.trainable_variables(T.cast(torch.nn.Module, self.target_critic)),
    ]
    for target in self.target_variables:
      target.requires_grad = False
    self.assign_targets()

  @torch.no_grad()
  def assign_targets(self) -> None:
    for o, t in zip(self.online_variables, self.target_variables):
      t.data.copy_(o.data)

  @torch.no_grad()
  def update_targets(self) -> None:
    for o, t in zip(self.online_variables, self.target_variables):
      t.data.mul_(1 - self.target_coeff)
      t.data.add_(self.target_coeff * o.data)


class ActorTwinCriticWithTargets(torch.nn.Module):
  """Actor-critic model with twin critics and target networks for off-policy algorithms (e.g., TD3, SAC)."""

  def __init__(
    self,
    actor: actors.ActorLike,
    critic: critics.CriticLike,
    observation_normalizer: normalizers.ObservationNormalizer | None = None,
    return_normalizer: normalizers.ReturnNormalizer | None = None,
    target_coeff: float = 0.005,
  ):
    super().__init__()
    self.actor = actor
    self.critic_1 = critic
    self.critic_2 = copy.deepcopy(critic)
    self.target_actor = copy.deepcopy(actor)
    self.target_critic_1 = copy.deepcopy(critic)
    self.target_critic_2 = copy.deepcopy(critic)
    self.observation_normalizer = observation_normalizer
    self.return_normalizer = return_normalizer
    self.target_coeff = target_coeff

  def initialize(
    self,
    observation_space: gym.spaces.Box | gym.spaces.Dict,
    action_space: gym.spaces.Box | gym.spaces.Dict,
  ) -> None:
    if self.observation_normalizer:
      self.observation_normalizer.initialize(observation_space)  # type: ignore
    self.actor.initialize(
      observation_space,
      action_space,
      self.observation_normalizer,
    )
    self.critic_1.initialize(
      observation_space,
      action_space,
      self.observation_normalizer,
      self.return_normalizer,
    )
    self.critic_2.initialize(
      observation_space,
      action_space,
      self.observation_normalizer,
      self.return_normalizer,
    )
    self.target_actor.initialize(
      observation_space,
      action_space,
      self.observation_normalizer,
    )
    self.target_critic_1.initialize(
      observation_space,
      action_space,
      self.observation_normalizer,
      self.return_normalizer,
    )
    self.target_critic_2.initialize(
      observation_space,
      action_space,
      self.observation_normalizer,
      self.return_normalizer,
    )
    self.online_variables = [
      *utils.trainable_variables(T.cast(torch.nn.Module, self.actor)),
      *utils.trainable_variables(T.cast(torch.nn.Module, self.critic_1)),
      *utils.trainable_variables(T.cast(torch.nn.Module, self.critic_2)),
    ]
    self.target_variables = [
      *utils.trainable_variables(T.cast(torch.nn.Module, self.target_actor)),
      *utils.trainable_variables(T.cast(torch.nn.Module, self.target_critic_1)),
      *utils.trainable_variables(T.cast(torch.nn.Module, self.target_critic_2)),
    ]
    for target in self.target_variables:
      target.requires_grad = False
    self.assign_targets()

  @torch.no_grad()
  def assign_targets(self) -> None:
    for o, t in zip(self.online_variables, self.target_variables):
      t.data.copy_(o.data)

  @torch.no_grad()
  def update_targets(self) -> None:
    for o, t in zip(self.online_variables, self.target_variables):
      t.data.mul_(1 - self.target_coeff)
      t.data.add_(self.target_coeff * o.data)
