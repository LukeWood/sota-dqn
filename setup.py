from setuptools import setup

VERSION = open("VERSION", "r").read()

setup(
    name='sota_dqn',
    packages=['sota_dqn'],
    version=VERSION,
    description='State of the art opinionanted DQN training and inference',
    long_description=open("README.md", "r").read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    url='https://github.com/lukewood/sota-dqn',
    author='Luke Wood',
    author_email='lukewoodcs@gmail.com',
    license='MIT',
)
