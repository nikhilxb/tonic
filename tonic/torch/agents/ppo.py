import typing as T

import torch

from tonic import logger
from tonic.torch import agent, models, replays, updaters


# ==================================================================================================
# Model

def ppo_default_model():
  return models.ActorCritic(
    actor=models.Actor(
      encoder=models.BoxObservationEncoder(),
      torso=models.MLP((64, 64), torch.nn.Tanh),
      head=models.StochasticDetachedStdPolicyHead(),
    ),
    critic=models.Critic(
      encoder=models.BoxObservationEncoder(),
      torso=models.MLP((64, 64), torch.nn.Tanh),
      head=models.ValueHead(),
    ),
    observation_normalizer=models.MeanStdNormalizer(),
  )

# ==================================================================================================
# Replay

PPOKeys = replays.OnPolicyKeys | T.Literal['log_probs']


class PPOData(replays.OnPolicyData):
  log_probs: torch.Tensor


class PPOStep(replays.OnPolicyStep):
  log_probs: torch.Tensor


# ==================================================================================================
# Agent

class PPO(agent.Agent):
  """Proximal Policy Optimization. https://arxiv.org/pdf/1707.06347.pdf"""
  model: models.ActorCritic

  def __init__(
    self,
    model: models.ActorCritic | None = None,
    actor_updater: updaters.ClippedRatio | None = None,
    critic_updater: updaters.VRegression | None = None,
    # Dataset.
    rollout_steps: int = 32,
    minibatch_samples: int = 4096,
    minibatch_iterations: int = 10,
    # Returns.
    discount_factor: float = 0.99,
    trace_decay: float = 0.97,
    normalize_advantages: bool = True,
    recompute_advantages: bool = False,
  ):
    self.model = model or ppo_default_model()
    self.actor_updater = actor_updater or updaters.ClippedRatio()
    self.critic_updater = critic_updater or updaters.VRegression()
    self.replay = replays.OnPolicyReplay[PPOKeys, PPOData, PPOStep](
      max_steps=rollout_steps,
      discount_factor=discount_factor,
      trace_decay=trace_decay,
    )
    self.rollout_steps = rollout_steps
    self.minibatch_samples = minibatch_samples
    self.minibatch_iterations = minibatch_iterations
    self.discount_factor = discount_factor
    self.trace_decay = trace_decay
    self.normalize_advantages = normalize_advantages
    self.recompute_advantages = recompute_advantages

  def initialize(
    self,
    observation_space: agent.ObservationSpace,
    action_space: agent.ActionSpace,
    seed: int,
  ) -> None:
    super().initialize(observation_space, action_space, seed)
    self.model.initialize(observation_space, action_space)
    self.replay.initialize(seed)
    self.actor_updater.initialize(self.model)
    self.critic_updater.initialize(self.model)

  @T.override
  def step(self, observations: agent.Observation) -> agent.Action:
    # Sample actions and get their log-probabilities for training.
    with torch.no_grad():
      distributions: models.ActionDistribution = self.model.actor(observations)
      actions, log_probs = distributions.sample_with_log_prob()
      log_probs = log_probs.sum(dim=-1)
    
    # Keep values for the next record.
    self.log_probs = log_probs

    return actions

  @T.override
  def test_step(self, observations: agent.Observation) -> agent.Action:
    # Sample actions for testing.
    with torch.no_grad():
      return self.model.actor(observations).sample()
    
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
      'log_probs': self.log_probs,
    })

    # Record transition in the normalizers.
    if self.model.observation_normalizer:
      self.model.observation_normalizer.record(observations)  # type: ignore
    if self.model.return_normalizer:
      self.model.return_normalizer.record(rewards)

  @T.override
  def update(self) -> None:
    # Skip update if replay is not full.
    if not self.replay.is_full(): return

    # Track actor and critic iterations.
    actor_is_updating = True
    actor_iterations = 0
    critic_iterations = 0

    # Update both the actor and the critic multiple times.
    for i in range(self.minibatch_iterations):
      # Compute the lambda-returns over the full buffer. Must do this before splitting into 
      # shuffled minibatches because sequential ordering is assumed.
      if i == 0 or self.recompute_advantages:
        full = self.replay.get_full('observations', 'next_observations')
        with torch.no_grad():
          values = self.model.critic(full['observations'])
          next_values = self.model.critic(full['next_observations'])
        self.replay.compute_returns(values, next_values)
        self.replay.compute_advantages(normalize=self.normalize_advantages)
    
      keys = ('observations', 'actions', 'advantages', 'log_probs', 'returns')
      for minibatch in self.replay.get_minibatches(*keys, size=self.minibatch_samples):
        # Update the actor if not stopped early.
        if actor_is_updating:
          actor_infos = self.actor_updater(
            observations=minibatch['observations'],
            actions=minibatch['actions'],
            advantages=minibatch['advantages'],
            log_probs=minibatch['log_probs'],
          )
          actor_iterations += 1
          # Check if actor training should stop early.
          actor_is_updating = not actor_infos['stop'].item()
          # Log actor metrics.
          for k, v in actor_infos.items():
            logger.store('actor/' + k, v.numpy(force=True))

        # Always update the critic.
        critic_infos = self.critic_updater(
          observations=minibatch['observations'],
          returns=minibatch['returns'],
        )
        critic_iterations += 1
        # Log critic metrics.
        for k, v in critic_infos.items():
          logger.store('critic/' + k, v.numpy(force=True))

    # Log iteration counts.
    logger.store('actor/iterations', actor_iterations)
    logger.store('critic/iterations', critic_iterations)

    # Reset the replay for the next rollout.
    self.replay.reset()

    # Update the normalizers.
    if self.model.observation_normalizer:
      self.model.observation_normalizer.update()
    if self.model.return_normalizer:
      self.model.return_normalizer.update()
