import numpy as np
from acme import wrappers
from utils.video_utils import render
from utils.video_utils import display_video

""" Tests a DeepMind control suite environment by executing a series of random actions """


def test_dm_control(env, n_frames):
    env = wrappers.CanonicalSpecWrapper(env, clip=True)
    env = wrappers.SinglePrecisionWrapper(env)

    spec = env.action_spec()
    timestep = env.reset()
    frames = [render(env)]

    for _ in range(n_frames):
        action = np.random.uniform(
            low=spec.minimum, high=spec.maximum, size=spec.shape)
        physics = env.physics
        # set_joint_damping(physics, damping_value)
        timestep = env.step(action)
        frames.append(render(env))
    return display_video(frames)
