from setuptools import setup

setup(
    name = 'fault_kinematics',
    version = '0.1',
    description = "A simple fault kinematics library",
    author = 'Joe Kington',
    author_email = 'joferkington@gmail.com',
    license = 'LICENSE',
    url = 'https://github.com/joferkington/fault_kinematics',
    packages = ['fault_kinematics'],
    install_requires = [
        'numpy >= 1.1',
        'scipy >= 0.7']
)
