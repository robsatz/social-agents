import numpy as np
import argparse
import os
# import yaml
import typing as T
import imageio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

import os
import tonic


import matplotlib.pyplot as plt
import matplotlib.animation as animation

def display_frames(frames, fps=30, output_dir='output_videos', output_filename='animation.gif'):
    """
    Displays a sequence of frames as an animated GIF within a Jupyter Notebook.
    
    Parameters:
    - frames (Iterable[np.ndarray]): An iterable of frames, where each frame is a numpy array.
    - fps (int): Frames per second for the video display.
    - output_dir (str): Directory to save the output GIF.
    - output_filename (str): Name of the output GIF file.
    
    Returns:
    HTML object: An HTML video element that can be displayed in a Jupyter Notebook.
    """
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Path to save the GIF
    filepath = os.path.join(output_dir, output_filename)
    
    # Create the figure and axis for the animation
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    # Display the first frame
    im = ax.imshow(frames[0])
    
    # Update function for animation
    def update(frame):
        im.set_data(frame)
        return [im]
    
    # Create and save the animation using PillowWriter
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=1000/fps)
    writer = animation.PillowWriter(fps=fps)
    ani.save(filepath, writer=writer)
    plt.close(fig)
    
    # Return the HTML for displaying the GIF
    return HTML(f'<img src="{filepath}" />')

def write_video(
    filepath: os.PathLike,
    frames: T.Iterable[np.ndarray],
    fps: int = 60,
    macro_block_size: T.Optional[int] = None,
    quality: int = 10,
    verbose: bool = False,
    **kwargs,
):
    """
    Saves a sequence of frames as a video file.

    Parameters:
    - filepath (os.PathLike): Path to save the video file.
    - frames (Iterable[np.ndarray]): An iterable of frames, where each frame is a numpy array.
    - fps (int, optional): Frames per second, defaults to 60.
    - macro_block_size (Optional[int], optional): Macro block size for video encoding, can affect compression efficiency.
    - quality (int, optional): Quality of the output video, higher values indicate better quality.
    - verbose (bool, optional): If True, prints the file path where the video is saved.
    - **kwargs: Additional keyword arguments passed to the imageio.get_writer function.

    Returns:
    None. The video is written to the specified filepath.
    """

    with imageio.get_writer(filepath,
                            # fps=fps,
                            # macro_block_size=macro_block_size,
                            # quality=quality,
                            **kwargs) as video:
        if verbose:
            print('Saving video to:', filepath)
        for frame in frames:
            video.append_data(frame)


def display_video(
    frames: T.Iterable[np.ndarray],
    filename='output_videos/temp.mp4',
    fps=60,
    **kwargs,
):
    """
    Displays a video within a Jupyter Notebook from an iterable of frames.

    Parameters:
    - frames (Iterable[np.ndarray]): An iterable of frames, where each frame is a numpy array.
    - filename (str, optional): Temporary filename to save the video before display, defaults to 'output_videos/temp.mp4'.
    - fps (int, optional): Frames per second for the video display, defaults to 60.
    - **kwargs: Additional keyword arguments passed to the write_video function.

    Returns:
    HTML object: An HTML video element that can be displayed in a Jupyter Notebook.
    """

    # Write video to a temporary file.
    filepath = os.path.abspath(filename)
    write_video(filepath, frames, fps=fps, verbose=False, **kwargs)

    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    # Switch to headless 'Agg' to inhibit figure rendering.
    matplotlib.use('Agg')
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]
    interval = 1000/fps
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    return HTML(anim.to_html5_video())


def render(env):
    """ Renders the current environment state to an image """
    return env.physics.render(camera_id=0, width=640, height=480)


def play_model(path, checkpoint='last', environment='default', seed=None, header=None):
    """
      Plays a model within an environment and renders the gameplay to a video.

      Parameters:
      - path (str): Path to the directory containing the model and checkpoints.
      - checkpoint (str): Specifies which checkpoint to use ('last', 'first', or a specific ID). 'none' indicates no checkpoint.
      - environment (str): The environment to use. 'default' uses the environment specified in the configuration file.
      - seed (int): Optional seed for reproducibility.
      - header (str): Optional Python code to execute before initializing the model, such as importing libraries.
      """

    if checkpoint == 'none':
        # Use no checkpoint, the agent is freshly created.
        checkpoint_path = None
        tonic.logger.log('Not loading any weights')
    else:
        checkpoint_path = os.path.join(path, 'checkpoints')
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
                checkpoint_path = os.path.join(
                    checkpoint_path, f'step_{checkpoint_id}')
            elif checkpoint == 'first':
                # Use the first checkpoint.
                checkpoint_id = min(checkpoint_ids)
                checkpoint_path = os.path.join(
                    checkpoint_path, f'step_{checkpoint_id}')
            else:
                # Use the specified checkpoint.
                checkpoint_id = int(checkpoint)
                if checkpoint_id in checkpoint_ids:
                    checkpoint_path = os.path.join(
                        checkpoint_path, f'step_{checkpoint_id}')
                else:
                    tonic.logger.error(
                        f'Checkpoint {checkpoint_id} not found in {checkpoint_path}')
                    checkpoint_path = None
        else:
            tonic.logger.error(f'No checkpoint found in {checkpoint_path}')
            checkpoint_path = None

    # Load the experiment configuration.
    arguments_path = os.path.join(path, 'config.yaml')
    with open(arguments_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = argparse.Namespace(**config)

    # Run the header first, e.g. to load an ML framework.
    try:
        if config.header:
            exec(config.header)
        if header:
            exec(header)
    except:
        pass

    # Build the agent.
    agent = eval(config.agent)

    # Build the environment.
    if environment == 'default':
        environment = tonic.environments.distribute(
            lambda: eval(config.environment))
    else:
        environment = tonic.environments.distribute(lambda: eval(environment))
    if seed is not None:
        environment.seed(seed)

    # Initialize the agent.
    agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        seed=seed,
    )

    # Load the weights of the agent form a checkpoint.
    if checkpoint_path:
        agent.load(checkpoint_path)

    steps = 0
    test_observations = environment.start()
    frames = [environment.render(
        'rgb_array', camera_id=0, width=640, height=480)[0]]
    score, length = 0, 0

    while True:
        # Select an action.
        actions = agent.test_step(test_observations, steps)
        assert not np.isnan(actions.sum())

        # Take a step in the environment.
        test_observations, infos = environment.step(actions)
        frames.append(environment.render(
            'rgb_array', camera_id=0, width=640, height=480)[0])
        agent.test_update(**infos, steps=steps)

        score += infos['rewards'][0]
        length += 1

        if infos['resets'][0]:
            break
    video_path = os.path.join(path, 'video.mp4')
    print('Reward for the run: ', score)
    return display_video(frames, video_path)
