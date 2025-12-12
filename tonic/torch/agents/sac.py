import typing as T

import torch

from tonic import logger
from tonic.torch import agent, explorations, models, replays, updaters


# ==================================================================================================
# Model

def sac_default_model():
  return models.ActorTwinCriticWithTargets(
    actor=models.Actor(
      encoder=models.BoxObservationEncoder(),
      torso=models.MLP((256, 256), torch.nn.ReLU),
      head=models.StochasticPolicyHead(
        mean_activation=torch.nn.Identity,
        distribution=models.SquashedNormalActionDistribution,
      ),
    ),
    critic=models.Critic(
      encoder=models.BoxObservationActionEncoder(),
      torso=models.MLP((256, 256), torch.nn.ReLU),
      head=models.ValueHead(),
    ),
    observation_normalizer=models.MeanStdNormalizer(),
  )


# ==================================================================================================
# Replay

SACKeys = replays.OffPolicyKeys
SACData = replays.OffPolicyData
SACStep = replays.OffPolicyStep


# ==================================================================================================
# Agent

class SAC(agent.Agent):
  """Soft Actor-Critic. https://arxiv.org/pdf/1801.01290.pdf"""
  model: models.ActorTwinCriticWithTargets

  def __init__(
    self,
    model: models.ActorTwinCriticWithTargets | None = None,
    exploration: explorations.NoActionNoise | None = None,
    actor_updater: updaters.TwinCriticSoftDeterministicPolicyGradient | None = None,
    critic_updater: updaters.TwinCriticSoftQLearning | None = None,
    # Dataset.
    replay_samples: int = 1_000_000,
    warmup_samples: int = 10_000,
    rollout_steps: int = 1,
    minibatch_samples: int = 128,
    minibatch_iterations: int = 50,
    # Returns.
    discount_factor: float = 0.99,
    return_steps: int = 1,
  ):
    self.model = model or sac_default_model()
    self.exploration = exploration or explorations.NoActionNoise()
    self.actor_updater = actor_updater or updaters.TwinCriticSoftDeterministicPolicyGradient()
    self.critic_updater = critic_updater or updaters.TwinCriticSoftQLearning()
    self.replay = replays.OffPolicyReplay[SACKeys, SACData, SACStep](
      max_samples=replay_samples,
      discount_factor=discount_factor,
      return_steps=return_steps,
    )
    self.replay_samples = replay_samples
    self.warmup_samples = warmup_samples
    self.rollout_steps = rollout_steps
    self.minibatch_samples = minibatch_samples
    self.minibatch_iterations = minibatch_iterations
    self.last_update_step = 0

  def initialize(
    self,
    observation_space: agent.ObservationSpace,
    action_space: agent.ActionSpace,
    seed: int,
  ) -> None:
    super().initialize(observation_space, action_space, seed)
    self.model.initialize(observation_space, action_space)
    self.replay.initialize(seed)
    self.exploration.initialize(lambda obs: self.model.actor(obs).sample(), action_space, seed)
    self.actor_updater.initialize(self.model)
    self.critic_updater.initialize(self.model)

  @T.override
  def step(self, observations: agent.Observation) -> agent.Action:
    # Sample stochastic actions for training (SAC uses stochastic policy).
    with torch.no_grad():
      return self.exploration(observations, self.replay.cumulative_samples())

  @T.override
  def test_step(self, observations: agent.Observation) -> agent.Action:
    # Use mean actions for testing.
    with torch.no_grad():
      return self.model.actor(observations).loc
    
  @T.override
  def record(
    self,
    observations: agent.Observation,
    actions: agent.Action,
    rewards: torch.Tensor,
    resets: torch.Tensor,
    terminations: torch.Tensor,
    next_observations: agent.Observation,
  ) -> None:
    # Record transition in the replay.
    self.replay.record({
      'observations': observations,
      'actions': actions,
      'rewards': rewards,
      'resets': resets,
      'terminations': terminations,
      'next_observations': next_observations,
    })

    # Record transition in the normalizers.
    if self.model.observation_normalizer:
      self.model.observation_normalizer.record(observations)  # type: ignore
    if self.model.return_normalizer:
      self.model.return_normalizer.record(rewards)
    
    # Record resets in exploration.
    self.exploration.record(resets)

  @T.override
  def update(self) -> None:
    # Skip update if not enough samples collected or rollout not complete.
    if self.replay.cumulative_samples() < self.warmup_samples: return
    if self.replay.cumulative_steps() < self.last_update_step + self.rollout_steps: return
    
    # Mark this update step.
    self.last_update_step = self.replay.cumulative_steps()

    # Update both the actor and critic multiple times.
    for i in range(self.minibatch_iterations):
      keys = ('observations', 'actions', 'next_observations', 'rewards', 'discounts')
      minibatch = self.replay.get_minibatch(*keys, size=self.minibatch_samples)
      # Update the critic first.
      critic_infos = self.critic_updater(
        observations=minibatch['observations'],
        actions=minibatch['actions'],
        next_observations=minibatch['next_observations'],
        rewards=minibatch['rewards'],
        discounts=minibatch['discounts'],
      )
      # Update the actor using the new critic.
      actor_infos = self.actor_updater(
        observations=minibatch['observations'],
      )
      # Update the target networks.
      self.model.update_targets()
      # Log training metrics.
      for k, v in actor_infos.items():
        logger.store('actor/' + k, v.numpy(force=True))
      for k, v in critic_infos.items():
        logger.store('critic/' + k, v.numpy(force=True))

    # Update the normalizers.
    if self.model.observation_normalizer:
      self.model.observation_normalizer.update()
    if self.model.return_normalizer:
      self.model.return_normalizer.update()
