import gym, sys, argparse
import numpy as np
from time import sleep

def viewer(env_name):
    env = gym.make("financial_gym/"+env_name)

    while True:
        done = False
        env.reset()
        env.render()
        action = env.action_space.sample()
        while not done:
            observation, reward, done, info = env.step(env.action_space.sample())
            env.render()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Financial Gym Environment Viewer')
    parser.add_argument('--env', default='GridWorld-v0',
                        help='Default Environment: GridWorld-v0')
    args = parser.parse_args()

    viewer(args.env)