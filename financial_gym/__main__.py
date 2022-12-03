import argparse
from .env_viewer import viewer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Financial Gym Environment Viewer')
    parser.add_argument('--env', default='GridWorld-v0',
                        help='Default Environment: GridWorld-v0')
    args = parser.parse_args()

    viewer(args.env)
