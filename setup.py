import subprocess
import platform
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

class Installer(develop):
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
        else:
            print(f"Unsupported platform: {system}")

        subprocess.run(["pip", "install", "-q", "ffmpeg"], check=True)
        subprocess.run(
            ["pip", "install", "-q", "dm-acme[envs]"], check=True)
        subprocess.run(
            ["pip", "install", "-q", "dm_control>=1.0.16"], check=True)
        super().run()

setup(
    name='social_agents',
    version='0.1',
    packages=find_packages(include=['social_agents', 'social_agents.*']),
    cmdclass={
        'develop': Installer,
    },
)
