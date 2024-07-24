# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Procedurally generated Swimmer domain."""
import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from lxml import etree
import numpy as np

_DEFAULT_TIME_LIMIT = 30
_CONTROL_TIMESTEP = .03  # (Seconds)

SUITE = containers.TaggedTasks()


def get_model_and_assets(n_joints):
  """Returns a tuple containing the model XML string and a dict of assets.

  Args:
    n_joints: An integer specifying the number of joints in the swimmer.

  Returns:
    A tuple `(model_xml_string, assets)`, where `assets` is a dict consisting of
    `{filename: contents_string}` pairs.
  """
  print('importing')
  return _make_model(n_joints), common.ASSETS


@SUITE.add('benchmarking')
def swimmer6(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns a 6-link swimmer."""
  return _make_swimmer(6, time_limit, random=random,
                       environment_kwargs=environment_kwargs)


@SUITE.add('benchmarking')
def swimmer15(time_limit=_DEFAULT_TIME_LIMIT, random=None,
              environment_kwargs=None):
  """Returns a 15-link swimmer."""
  return _make_swimmer(15, time_limit, random=random,
                       environment_kwargs=environment_kwargs)


def swimmer(n_links=3, time_limit=_DEFAULT_TIME_LIMIT,
            random=None, environment_kwargs=None):
  """Returns a swimmer with n links."""
  return _make_swimmer(n_links, time_limit, random=random,
                       environment_kwargs=environment_kwargs)


def _make_swimmer(n_joints, time_limit=_DEFAULT_TIME_LIMIT, random=None,
                  environment_kwargs=None):
  """Returns a swimmer control environment."""
  model_string, assets = get_model_and_assets(n_joints)
  physics = Physics.from_xml_string(model_string, assets=assets)
  task = Swimmer(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


def _make_model(n_bodies):
  """Generates an xml string defining a swimmer with `n_bodies` bodies."""
  if n_bodies < 3:
    raise ValueError('At least 3 bodies required. Received {}'.format(n_bodies))
  script_dir = os.path.dirname(__file__)  # Get the directory of the current script
  resources_dir = os.path.join(script_dir, 'resources')  # Go to resources directory
  file_path = os.path.join(resources_dir, 'swimmer.xml')

  # Read the swimmer.xml file
  with open(file_path, 'r') as file:
      mjcf = etree.parse(file)
  
  root = mjcf.getroot()  # Get the root element

  head_bodies = root.findall('./worldbody/body')
  
  actuator = etree.SubElement(root, 'actuator')
  sensor = etree.SubElement(root, 'sensor')

  for body_id, parent in enumerate(head_bodies):
    body_id+=1
    for body_index in range(n_bodies - 1):
      site_name = 'site_{}_{}'.format(body_id, body_index)
      child = _make_body(body_id=body_id, body_index=body_index)
      child.append(etree.Element('site', name=site_name))
      joint_name = 'joint_{}_{}'.format(body_id, body_index)
      joint_limit = 360.0/n_bodies
      joint_range = '{} {}'.format(-joint_limit, joint_limit)
      child.append(etree.Element('joint', {'name': joint_name,
                                          'range': joint_range}))
      motor_name = 'motor_{}_{}'.format(body_id, body_index)
      actuator.append(etree.Element('motor', name=motor_name, joint=joint_name))
      velocimeter_name = 'velocimeter_{}_{}'.format(body_id, body_index)
      sensor.append(etree.Element('velocimeter', name=velocimeter_name,
                                  site=site_name))
      gyro_name = 'gyro_{}_{}'.format(body_id, body_index)
      sensor.append(etree.Element('gyro', name=gyro_name, site=site_name))
      parent.append(child)
      parent = child

  # Move tracking cameras further away from the swimmer according to its length.
  cameras = mjcf.findall('./worldbody/body/camera')
  scale = n_bodies / 6.0
  for cam in cameras:
    if cam.get('mode') == 'trackcom':
      old_pos = cam.get('pos').split(' ')
      new_pos = ' '.join([str(float(dim) * scale) for dim in old_pos])
      cam.set('pos', new_pos)

  print('Two Bodies Imported')

  return etree.tostring(mjcf, pretty_print=True)


def _make_body(body_index, body_id):
  """Generates an xml string defining a single physical body."""
  body_name = 'segment_{}_{}'.format(body_id, body_index)
  visual_name = 'visual_{}_{}'.format(body_id, body_index)
  inertial_name = 'inertial_{}_{}'.format(body_id, body_index)
  body = etree.Element('body', name=body_name)
  body.set('pos', '0 .1 0')
  etree.SubElement(body, 'geom', {'class': 'visual', 'name': visual_name})
  etree.SubElement(body, 'geom', {'class': 'inertial', 'name': inertial_name})
  return body


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the swimmer domain."""

  def nose_to_target(self, head_name, nose_name):
    """Returns a vector from nose to target in local coordinate of the head."""
    nose_to_target = (self.named.data.geom_xpos['target'] -
                      self.named.data.geom_xpos[nose_name])
    head_orientation = self.named.data.xmat[head_name].reshape(3, 3)
    return nose_to_target.dot(head_orientation)[:2]

  def nose_to_target_dist(self, head_name, nose_name):
    """Returns the distance from the nose to the target."""
    return np.linalg.norm(self.nose_to_target(head_name, nose_name))

  def body_velocities(self):
    """Returns local body velocities: x, y linear, z rotational for 2 swimmers."""
    print(self.named.data.sensordata)
    # Extract the portion of sensordata after index 21
    xvel_local = self.named.data.sensordata[21:]

    head_size = 6
    sensor_size = 30  # 15 velocimeter values + 15 gyro values

    swimmer1_head = xvel_local[:head_size]
    swimmer1_sensors = xvel_local[head_size:head_size + sensor_size]

    swimmer2_head = xvel_local[head_size + sensor_size:head_size + 2 * sensor_size]
    swimmer2_sensors = xvel_local[head_size + 2 * sensor_size:]

    swimmer1 = np.concatenate((swimmer1_head, swimmer1_sensors))
    swimmer2 = np.concatenate((swimmer2_head, swimmer2_sensors))

    swimmer1 = swimmer1.reshape((-1, 6))
    swimmer2 = swimmer2.reshape((-1, 6))

    print(swimmer1, swimmer2)

    vx_vy_wz = [0, 1, 5]  # Indices for linear x,y vels and rotational z vel.
    swim1_xvel = swimmer1[:, vx_vy_wz].ravel()
    swim2_xvel = swimmer2[:, vx_vy_wz].ravel()
    xvel = np.concatenate((swim1_xvel, swim2_xvel))
    return xvel

  def joints(self):
    """Returns all internal joint angles (excluding root joints) for each swimmer."""
    qpos = self.data.qpos[3:]


class Swimmer(base.Task):
  """A swimmer `Task` to reach the target or just swim."""

  def __init__(self, random=None):
    """Initializes an instance of `Swimmer`.

    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    super().__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    Initializes the swimmer orientation to [-pi, pi) and the relative joint
    angle of each joint uniformly within its range.

    Args:
      physics: An instance of `Physics`.
    """
    # Random joint angles:
    randomizers.randomize_limited_and_rotational_joints(physics, self.random)
    # Random target position.
    close_target = self.random.rand() < .2  # Probability of a close target.
    target_box = .3 if close_target else 2
    xpos, ypos = self.random.uniform(-target_box, target_box, size=2)
    physics.named.model.geom_pos['target', 'x'] = xpos
    physics.named.model.geom_pos['target', 'y'] = ypos
    physics.named.model.light_pos['target_light', 'x'] = xpos
    physics.named.model.light_pos['target_light', 'y'] = ypos

    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of joint angles, body velocities and target."""
    obs = collections.OrderedDict()
    obs['joints'] = physics.joints()
    obs['to_target'] = physics.nose_to_target()
    obs['body_velocities'] = physics.body_velocities()
    return obs

  def get_reward(self, physics):
    """Returns a smooth reward."""
    target_size = physics.named.model.geom_size['target', 0]
    return rewards.tolerance(physics.nose_to_target_dist('head1', 'nose1'),
                             bounds=(0, target_size),
                             margin=5*target_size,
                             sigmoid='long_tail')
