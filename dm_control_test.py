import numpy as np
from acme import wrappers
from dm_control import suite, rl
from social_agents.utils.video_utils import render
from social_agents.utils.video_utils import display_video, display_frames
import social_agents.agents.swimmer as swimmer
from social_agents.tasks.forwards_tasks import Swim
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
        timestep = env.step(action)
        frames.append(render(env))
    return display_frames(frames, output_dir='output_videos', output_filename='animation.gif')

_SWIM_SPEED = 0.1

if __name__ == "__main__":
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
        return rl.control.Environment(
            physics,
            task,
            time_limit=time_limit,
            control_timestep=swimmer._CONTROL_TIMESTEP,
            **environment_kwargs,
        )

    env = suite.load('swimmer', 'swim_12_links', task_kwargs={'random': 1})
    test_dm_control(env, n_frames=100)
