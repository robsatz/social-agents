from cust_utils.video_utils import render
from acme import wrappers
from cust_utils.video_utils import display_video
import numpy as np

""" Tests a DeepMind control suite environment by executing a series of random actions """
def test_dm_control(env):
    env = wrappers.CanonicalSpecWrapper(env, clip=True)
    env = wrappers.SinglePrecisionWrapper(env)

    spec = env.action_spec()
    timestep = env.reset()
    frames = [render(env)]

    for _ in range(60):
        action = np.random.uniform(low=spec.minimum, high=spec.maximum, size=spec.shape)
        physics = env.physics
        # set_joint_damping(physics, damping_value)
        timestep = env.step(action)
        frames.append(render(env))
    return display_video(frames)