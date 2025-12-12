import typing as T

import torch

from tonic import logger
from tonic.torch import agent, models, replays, updaters


# ==================================================================================================
# Model

def mpo_default_model():
  return models.ActorCriticWithTargets(
    actor=models.Actor(
      encoder=models.BoxObservationEncoder(),
      torso=models.MLP((256, 256), torch.nn.ReLU),
      head=models.StochasticPolicyHead(),
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

MPOKeys = replays.OffPolicyKeys
MPOData = replays.OffPolicyData
MPOStep = replays.OffPolicyStep


# ==================================================================================================
# Agent

class MPO(agent.Agent):
  """Maximum a Posteriori Policy Optimisation.
  MPO: https://arxiv.org/pdf/1806.06920.pdf
  MO-MPO: https://arxiv.org/pdf/2005.07513.pdf
  """
  model: models.ActorCriticWithTargets

  def __init__(
    self,
    model: models.ActorCriticWithTargets | None = None,
    actor_updater: updaters.MaximumAPosterioriPolicyOptimization | None = None,
    critic_updater: updaters.ExpectedSARSA | None = None,
    # Dataset.
    replay_samples: int = 1_000_000,
    warmup_samples: int = 10_000,
    rollout_steps: int = 1,
    minibatch_samples: int = 128,
    minibatch_iterations: int = 50,
    # Returns.
    discount_factor: float = 0.99,
    return_steps: int = 5,
  ):
    self.model = model or mpo_default_model()
    self.actor_updater = actor_updater or updaters.MaximumAPosterioriPolicyOptimization()
    self.critic_updater = critic_updater or updaters.ExpectedSARSA()
    self.replay = replays.OffPolicyReplay[MPOKeys, MPOData, MPOStep](
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
    self.actor_updater.initialize(self.model, action_space)
    self.critic_updater.initialize(self.model)

  @T.override
  def step(self, observations: agent.Observation) -> agent.Action:
    # Sample actions for training.
    with torch.no_grad():
      distributions: models.ActionDistribution = self.model.actor(observations)
      actions = distributions.sample()
    return actions

  @T.override
  def test_step(self, observations: agent.Observation) -> agent.Action:
    # Use mean actions for testing.
    with torch.no_grad():
      distributions: models.ActionDistribution = self.model.actor(observations)
      actions = distributions.mean()
    return actions
    
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
