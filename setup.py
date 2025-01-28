from setuptools import setup, find_packages
import sys, os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'financial_gym'))

directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'financial_gym', 'envs')


setup(
    name='financial-gym',
    version='1.0',
    packages=find_packages(),
    install_requires=['gym==0.22', "alpaca-py==0.6.1", "ray==2.1.0", "ray[rllib]==2.1.0", "tensorflow", "mplfinance==0.12.9b5"],  
    author='Gabriele Ansaldo',
    author_email='ansaldo.g@northeastern.edu',
)
