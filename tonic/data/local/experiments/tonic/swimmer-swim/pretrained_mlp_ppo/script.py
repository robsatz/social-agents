"""
Train agent using algorithms from Tonic.
"""
import argparse
import os
# import sys
# tonic_path = os.path.expanduser('~/ncap')
# sys.path.append(tonic_path)

# import tonic.tonic
# import tonic.tonic.torch



import tonic
import tonic.torch


# Register custom environments.
import neurobo.ncap.envs


def train(
  header,
  agent,
  environment,
  trainer,
  before_training,
  after_training,
  parallel,
  sequential,
  seed,
  name,
  note
):
  # Capture the arguments to save them, e.g. to play with the trained agent.
  args = dict(locals())

  # Run the header first, e.g. to load an ML framework.
  if header:
    exec(header)

  # Build the agent.
  agent = eval(agent)

  # Build the train and test environments.
  _environment = environment
  environment = tonic.environments.distribute(lambda: eval(_environment), parallel, sequential)
  test_environment = tonic.environments.distribute(lambda: eval(_environment))

  # Choose a name for the experiment.
  if hasattr(test_environment, 'name'):
    environment_name = test_environment.name
  else:
    environment_name = test_environment.__class__.__name__
  if not name:
    if hasattr(agent, 'name'):
      name = agent.name
    else:
      name = agent.__class__.__name__
    if parallel != 1 or sequential != 1:
      name += f'-{parallel}x{sequential}'
  environment.initialize(seed=seed)
  test_environment.initialize(seed=seed + 10000)
  
  agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space, seed=seed)

  # Initialize the logger to save data to the path environment/name/seed.
  path = os.path.join('data', 'local', 'experiments', 'tonic', environment_name, name)
  tonic.logger.initialize(path, script_path=__file__, config=args)

  # Build the trainer.
  trainer = eval(trainer)
  trainer.initialize(
    agent=agent,
    environment=environment,
    test_environment=test_environment,
    # seed=seed,
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
  # Argument parsing.
  parser = argparse.ArgumentParser()
  parser.add_argument('--header')
  parser.add_argument('--agent', required=True)
  parser.add_argument('--environment', '--env', required=True)
  parser.add_argument('--trainer', default='tonic.Trainer()')
  parser.add_argument('--before_training')
  parser.add_argument('--after_training')
  parser.add_argument('--parallel', type=int, default=1)
  parser.add_argument('--sequential', type=int, default=1)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--name')
  parser.add_argument('--note')
  args = vars(parser.parse_args())
  train(**args)
