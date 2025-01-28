from setuptools import setup, find_packages
import sys, os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'financial_gym'))

directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'financial_gym', 'envs')


setup(
    name='financial-gym',
    version='1.0',
    packages=find_packages(),
    install_requires=['gym', "alpaca-py", "ray", "ray[rllib]", "tensorflow", "tensorflow-macos", "tensorflow-metal" "mplfinance"],  
    author='Gabriele Ansaldo',
    author_email='ansaldo.g@northeastern.edu',
)
