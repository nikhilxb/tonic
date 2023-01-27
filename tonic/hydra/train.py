'''Script used to train agents.'''
import argparse
import os
import yaml
import typing as T
import dataclasses

import tonic
import omegaconf
import hydra


@dataclasses.dataclass
class TonicTrainConfig:
    header: T.Optional[str] = None
    agent: T.Any = None
    environment: T.Any = None
    test_environment: T.Any = None
    trainer: T.Any = None
    before_training: T.Optional[str] = None
    after_training: T.Optional[str] = None
    parallel: int = 1
    sequential: int = 1
    seed: int = 0
    checkpoint: T.Union[T.Literal['last', 'first'], int] = 'last'
    checkpoint_output_dir: T.Optional[str] = None


@hydra.main(version_base="1.2", config_name="train", config_path="configs")
def main(cfg: omegaconf.DictConfig):
    '''Trains an agent on an environment.'''
    cfg = omegaconf.OmegaConf.to_container(cfg.tonic)
    header = cfg['header']
    agent = cfg['agent']
    environment = cfg['environment']
    test_environment = cfg['test_environment']
    trainer = cfg['trainer']
    before_training = cfg['before_training']
    after_training = cfg['after_training']
    parallel = cfg['parallel']
    sequential = cfg['sequential']
    seed = cfg['seed']
    checkpoint = cfg['checkpoint']
    checkpoint_output_dir = cfg['checkpoint_output_dir']

    # Capture the arguments to save them, e.g. to play with the trained agent.
    args = dict(locals())
    del args['cfg']

    checkpoint_path = None

    # Process the checkpoint path same way as in tonic.play.
    if checkpoint_output_dir:
        tonic.logger.log(f'Loading experiment from {checkpoint_output_dir}')

        # Use no checkpoint, the agent is freshly created.
        if checkpoint == 'none' or agent is not None:
            tonic.logger.log('Not loading any weights')
        else:
            checkpoint_path = os.path.join(checkpoint_output_dir, 'checkpoints')
            if not os.path.isdir(checkpoint_path):
                tonic.logger.error(f'{checkpoint_path} is not a directory')
                checkpoint_path = None

            # List all the checkpoints.
            checkpoint_ids = []
            for file in os.listdir(checkpoint_path):
                if file[:5] == 'step_':
                    checkpoint_id = file.split('.')[0]
                    checkpoint_ids.append(int(checkpoint_id[5:]))

            if checkpoint_ids:
                if checkpoint == 'last':
                    # Use the last checkpoint.
                    checkpoint_id = max(checkpoint_ids)
                    checkpoint_path = os.path.join(checkpoint_path, f'step_{checkpoint_id}')
                elif checkpoint == 'first':
                    # Use the first checkpoint.
                    checkpoint_id = min(checkpoint_ids)
                    checkpoint_path = os.path.join(checkpoint_path, f'step_{checkpoint_id}')
                else:
                    # Use the specified checkpoint.
                    checkpoint_id = int(checkpoint)
                    if checkpoint_id in checkpoint_ids:
                        checkpoint_path = os.path.join(checkpoint_path, f'step_{checkpoint_id}')
                    else:
                        tonic.logger.error(f'Checkpoint {checkpoint_id} not found in {checkpoint_path}')
                        checkpoint_path = None
            else:
                tonic.logger.error(f'No checkpoint found in {checkpoint_path}')
                checkpoint_path = None

        # Load the experiment configuration.
        arguments_path = os.path.join(checkpoint_output_dir, 'config.yaml')
        with open(arguments_path, 'r') as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        config = argparse.Namespace(**config)

        header = header or config.header
        agent = agent or config.agent
        environment = environment or config.environment
        test_environment = test_environment or config.test_environment
        trainer = trainer or config.trainer

    # Run the header first, e.g. to load an ML framework.
    if header:
        exec(header)

    # Build the training environment.
    environment_cfg = environment
    environment = tonic.environments.distribute(
        lambda: hydra.utils.instantiate(environment_cfg), parallel, sequential)
    environment.initialize(seed=seed)

    # Build the testing environment.
    test_environment_cfg = test_environment or environment_cfg
    test_environment = tonic.environments.distribute(
        lambda: hydra.utils.instantiate(test_environment_cfg))
    test_environment.initialize(seed=seed + 10000)

    # Build the agent.
    agent = hydra.utils.instantiate(agent)
    agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        seed=seed
    )

    # Load the weights of the agent form a checkpoint.
    if checkpoint_path:
        agent.load(checkpoint_path)

    # Initialize the logger to save data to the output directory.
    output_dir = hydra.utils.HydraConfig.get()['runtime']['output_dir']
    tonic.logger.initialize(output_dir, script_path=__file__, config=args)

    # Build the trainer.
    trainer = hydra.utils.instantiate(trainer)
    trainer.initialize(
        agent=agent,
        environment=environment,
        test_environment=test_environment,
    )

    # Run some code before training.
    if before_training:
        exec(before_training)

    # Train.
    trainer.run()

    # Run some code after training.
    if after_training:
        exec(after_training)


if __name__ == '__main__':
    main()
