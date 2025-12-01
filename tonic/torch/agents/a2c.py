import typing as T

import torch
import gym.spaces

from tonic import logger
from tonic.torch import agents, models, replays, updaters


def a2c_default_model():
    return models.ActorCritic(
        actor=models.Actor(
            encoder=models.BoxObservationEncoder(),
            torso=models.MLP((64, 64), torch.nn.Tanh),
            head=models.DetachedScaleGaussianPolicyHead()),
        critic=models.Critic(
            encoder=models.BoxObservationEncoder(),
            torso=models.MLP((64, 64), torch.nn.Tanh),
            head=models.ValueHead()),
        observation_normalizer=models.MeanStdNormalizer())


A2CBufferKeys = T.Literal[
    'observations', 'next_observations', 'actions', 'rewards', 'resets', 'terminations', 'returns', 'values', 'next_values', 'advantages', 'log_probs'
]


class A2C(agents.Agent):
    '''Advantage Actor Critic (aka Vanilla Policy Gradient).
    A3C: https://arxiv.org/pdf/1602.01783.pdf
    '''
    model: models.ActorCritic

    def __init__(
        self,
        model: models.ActorCritic | None = None,
        replay: replays.OnPolicyBuffer[A2CBufferKeys] | None = None,
        actor_updater: updaters.StochasticPolicyGradient | None = None,
        critic_updater: updaters.VRegression | None = None,
    ):
        self.model = model or a2c_default_model()
        self.replay = replay or replays.OnPolicyBuffer[A2CBufferKeys]()
        self.actor_updater = actor_updater or updaters.StochasticPolicyGradient()
        self.critic_updater = critic_updater or updaters.VRegression()

    def initialize(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Dict,
        seed: int,
    ):
        super().initialize(seed)
        self.model.initialize(observation_space, action_space)
        self.replay.initialize(seed)
        self.actor_updater.initialize(self.model)
        self.critic_updater.initialize(self.model)

    @T.override
    def step(self, observations: torch.Tensor | dict[str, torch.Tensor], step: int):
        # Sample actions and get their log-probabilities for training.
        actions, log_probs = self._step(observations)
        actions = actions.numpy()
        log_probs = log_probs.numpy()

        # Keep some values for the next update.
        self.last_observations = observations.copy()
        self.last_actions = actions.copy()
        self.last_log_probs = log_probs.copy()

        return actions

    @T.override
    def test_step(self, observations: torch.Tensor | dict[str, torch.Tensor], step: int):
        # Sample actions for testing.
        return self._test_step(observations).numpy()

    @T.override
    def update(
        self,
        observations: torch.Tensor | dict[str, torch.Tensor],
        rewards: torch.Tensor,
        resets: torch.Tensor,
        terminations: torch.Tensor,
        steps: int,
    ):
        # Store the last transitions in the replay.
        self.replay.record({
            'observations': self.last_observations,
            'actions': self.last_actions,
            'rewards': rewards,
            'resets': resets,
            'terminations': terminations,
            'next_observations': observations,
            'log_probs': self.last_log_probs,
        })

        # Prepare to update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.record(self.last_observations)
        if self.model.return_normalizer:
            self.model.return_normalizer.record(rewards)

        # Update the model if the replay is ready.
        if self.replay.ready():
            self._update()

    def _step(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            distributions = self.model.actor(observations)
            if hasattr(distributions, 'sample_with_log_prob'):
                actions, log_probs = distributions.sample_with_log_prob()
            else:
                actions = distributions.sample()
                log_probs = distributions.log_prob(actions)
            log_probs = log_probs.sum(dim=-1)
        return actions, log_probs

    def _test_step(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            return self.model.actor(observations).sample()

    def _evaluate(self, observations, next_observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        next_observations = torch.as_tensor(
            next_observations, dtype=torch.float32)
        with torch.no_grad():
            values = self.model.critic(observations)
            next_values = self.model.critic(next_observations)
        return values, next_values

    def _update(self):
        # Compute the lambda-returns.
        batch = self.replay.get_full('observations', 'next_observations')
        values, next_values = self._evaluate(**batch)
        values, next_values = values.numpy(), next_values.numpy()
        self.replay.compute_returns(values, next_values)
        self.replay.compute_advantages(normalize=True)

        # Update the actor once.
        batch = self.replay.get_full('observations', 'actions', 'advantages', 'log_probs')
        infos = self.actor_updater(**batch)
        for k, v in infos.items():
            logger.store('actor/' + k, v.numpy())

        # Update the critic multiple times.
        for minibatch in self.replay.get_batches('observations', 'returns'):
            infos = self.critic_updater(**minibatch)
            for k, v in infos.items():
                logger.store('critic/' + k, v.numpy())

        # Update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()

        # Reset the replay.
        self.replay.reset()
