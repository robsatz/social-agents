import subprocess
import platform
from setuptools import setup, find_packages
from setuptools.command.install import install


class CustomInstallCommand(install):
    def run(self):
        # Determine the platform
        system = platform.system()
        print(f"Detected platform: {system}")
        if system == 'Linux':
            # Linux-specific commands
            subprocess.run(["sudo", "apt-get", "install",
                           "-y", "libglew-dev"], check=True)
            subprocess.run(["sudo", "apt-get", "install",
                           "-y", "libglfw3"], check=True)
            subprocess.run(["sudo", "apt", "install", "ffmpeg"], check=True)
        elif system == 'Darwin':  # macOS
            # macOS-specific commands
            subprocess.run(["brew", "install", "glew"], check=True)
            subprocess.run(["brew", "install", "glfw"], check=True)
        elif system == 'Windows':
            # Windows-specific commands
            subprocess.run(["pip", "install", "-q", "ffmpeg"], check=True)
            subprocess.run(
                ["pip", "install", "-q", "dm-acme[envs]"], check=True)
            subprocess.run(
                ["pip", "install", "-q", "dm_control>=1.0.16"], check=True)
        else:
            print(f"Unsupported platform: {system}")

        # Proceed with the standard installation
        install.run(self)


setup(
    name='tonic',
    description='Tonic RL Library',
    url='https://github.com/fabiopardo/tonic',
    version='0.3.0',
    author='Fabio Pardo',
    author_email='f.pardo@imperial.ac.uk',
    install_requires=[
        'gym', 'matplotlib', 'numpy', 'pandas', 'pyyaml', 'termcolor'],
    license='MIT',
    python_requires='>=3.6',
    keywords=['tonic', 'deep learning', 'reinforcement learning'],
    packages=find_packages(include=['tonic'])  # the line being added
)

setup(
    name='social_agents',
    version='0.1',
    packages=find_packages(where='social_agents'),
    package_dir={'': 'social_agents'},
    cmdclass={
        'install': CustomInstallCommand,
    },
)
