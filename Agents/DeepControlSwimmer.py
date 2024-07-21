import dm_control as dm
import dm_control.suite.swimmer as swimmer
from dm_control.rl import control
from dm_control.utils import rewards
from dm_control import suite
from dm_control.suite.wrappers import pixels
import collections
import sys
import os
import os
import sys

# # Get the current working directory
# cwd = os.getcwd()

# # Split the path in a platform-independent way
# path_list = cwd.split(os.path.sep)

# # Initialize variable
# social_agents_path = ''

# # Iterate over the segments to find the 'social-agents' directory
# for part in path_list:
#     if part.lower() == 'social-agents':  # case-insensitive comparison
#         social_agents_path = os.path.join(*path_list[:path_list.index(part)+1])
#         sys.path.append(social_agents_path)
#         break  # Exit the loop once the path is found
# sys.path.append(social_agents_path)

# def modify_pythonpath(new_path):
    
#     # Check if PYTHONPATH already exists in the environment
#     current_pythonpath = os.environ.get('PYTHONPATH', '')
    
#     # If it exists, append the new path using the appropriate separator
#     if current_pythonpath:
#         # os.pathsep gives the separator used on this OS (';' for Windows, ':' for Unix)
#         new_pythonpath = new_path + os.pathsep + current_pythonpath
#     else:
#         new_pythonpath = new_path
    
#     # Set the new PYTHONPATH in the environment
#     os.environ['PYTHONPATH'] = new_pythonpath

# sys.path.insert(0, 'tasks/forwars_tasks')
# modify_pythonpath(social_agents_path)
# print(sys.path)

import sys
import os

# Add the parent directory to sys.path to resolve the relative imports
script_dir = os.path.dirname(__file__)  # Gets the directory where the script is located
parent_dir = os.path.dirname(script_dir)  # Gets the parent directory
sys.path.append(parent_dir)

from tasks.forwards_tasks import Swim, _SWIM_SPEED


# An agent with 6 joints which passed into 
@swimmer.SUITE.add() # added to domain swimmer
def swim(
  n_links=6,
  desired_speed=_SWIM_SPEED,
  time_limit=swimmer._DEFAULT_TIME_LIMIT,
  random=None,
  environment_kwargs={},
):
  '''Passed into suite.load()'''
  """Returns the Swim task for a n-link swimmer."""
  model_string, assets = swimmer.get_model_and_assets(n_links)
  physics = swimmer.Physics.from_xml_string(model_string, assets=assets)
  task = Swim(desired_speed=desired_speed, random=random)
  return control.Environment(
    physics,
    task,
    time_limit=time_limit,
    control_timestep=swimmer._CONTROL_TIMESTEP,
    **environment_kwargs,
  )

@swimmer.SUITE.add()
def swim_12_links(
  n_links=12,
  desired_speed=_SWIM_SPEED,
  time_limit=swimmer._DEFAULT_TIME_LIMIT,
  random=None,
  environment_kwargs={},
):
  """Returns the Swim task for a n-link swimmer."""
  model_string, assets = swimmer.get_model_and_assets(n_links)
  physics = swimmer.Physics.from_xml_string(model_string, assets=assets)
  task = Swim(desired_speed=desired_speed, random=random)
  return control.Environment(
    physics,
    task,
    time_limit=time_limit,
    control_timestep=swimmer._CONTROL_TIMESTEP,
    **environment_kwargs,
  )
