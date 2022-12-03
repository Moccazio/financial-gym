import sys, shutil, os, glob, argparse, gym, financial_gym
import financial_gym
import numpy as np
import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.algorithms.dqn as dqn
import multiprocessing
import matplotlib.pyplot as plt

def get_agent(algo, env_name, seed=1):
    num_processes = multiprocessing.cpu_count()
    if algo == 'ppo':
        config = ppo.DEFAULT_CONFIG.copy()
        config["framework"] = "tf2"
        config['train_batch_size'] = 19200
        config['num_sgd_iter'] = 50
        config['sgd_minibatch_size'] = 128
        config['lambda'] = 0.95
        config['model']['fcnet_hiddens'] = [256, 256]
        config['model']['fcnet_activation'] = "tanh"

    if algo == 'dqn':
        config = dqn.DEFAULT_CONFIG.copy()
        config["framework"] = "tf2"
        config["hiddens"] = [512,512,512]

    config["log_level"] = "ERROR"
    config["eager_tracing"] = True
    config['num_workers'] = num_processes
    config['num_cpus_per_worker'] = 0
    config['seed'] = seed

    if algo == 'ppo':
        agent = ppo.PPOTrainer(config, env=env_name)
    if algo == 'dqn':
        # agent = dqn.DQNTrainer(config, env=env_name)
        agent = dqn.DQN(config=config, env=env_name)
    return agent

def load_policy(algo, env_name, policy_path, seed=1):
    agent = get_agent(algo, env_name, seed)
    # Find the most recent policy in the directory
    directory = os.path.join(policy_path, algo, env_name)
    files = [f.split('_')[-1] for f in glob.glob(os.path.join(directory, 'checkpoint_*'))]
    files_ints = [int(f) for f in files]
    if files:
        checkpoint_max = max(files_ints)
        checkpoint_num = files_ints.index(checkpoint_max)
        checkpoint_path = os.path.join(directory, 'checkpoint_%s' % files[checkpoint_num])
        agent.restore(checkpoint_path)
        return agent, checkpoint_path
    return agent, None

def train_policy(env_name, algo, timesteps_total=1000, save_dir='./ray_trained_models/', policy_path='./ray_trained_models/', learning_curve=True, seed=1):
    if learning_curve:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        learning_curve_data = {'timesteps': [],
                                'min': [],
                                'max': [],
                                'mean': []}
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    agent, checkpoint_path = load_policy(algo, env_name, policy_path, seed=1)
    timesteps = 0
    while timesteps < timesteps_total:
        result = agent.train()
        timesteps = result['timesteps_total']

        print()
        print(f"Iteration: {result['training_iteration']}, total timesteps: {timesteps}, total time: {result['time_total_s']:.1f}, total mean reward (min/max): {result['episode_reward_mean']:.1f} ({result['episode_reward_min']:.1f}/{result['episode_reward_max']:.1f})")
        sys.stdout.flush()  

        # Delete the old saved policy
        if checkpoint_path is not None:
            shutil.rmtree(os.path.dirname(checkpoint_path), ignore_errors=True)
        # Save the recently trained policy
        checkpoint_path = agent.save(os.path.join(save_dir, algo, env_name))

        # Plot learning curve
        learning_curve_data['timesteps'].append(timesteps)
        learning_curve_data['min'].append(result['episode_reward_min'])
        learning_curve_data['max'].append(result['episode_reward_max'])
        learning_curve_data['mean'].append(result['episode_reward_mean'])
        if learning_curve:
            ax.clear()
            ax.plot(learning_curve_data['timesteps'], learning_curve_data['mean'], color='black', label='Mean')
            ax.plot(learning_curve_data['timesteps'], learning_curve_data['min'], color='r', label='Min')
            ax.plot(learning_curve_data['timesteps'], learning_curve_data['max'], color='g', label='Max')
            ax.fill_between(learning_curve_data['timesteps'], learning_curve_data['min'], learning_curve_data['max'], color='black',alpha=0.2)
            plt.legend()
            plt.pause(0.5)
    if learning_curve:
        plt.show()

def render_policy(env_name, algo, policy_path='./ray_trained_models/', n_episodes=1, seed=1):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    env = gym.make("financial_gym/"+env_name)
    test_agent, _ = load_policy(algo, env_name, policy_path, seed=1)
    for episode in range(n_episodes):
        done = False
        observation = env.reset()
        env.render()
        while not done:
            # Compute the next action based on the trained policy
            test_action = test_agent.compute_single_action(observation)
            # Step the environment forward using the action from the trained policy
            observation, reward, done, info = env.step(test_action)
            env.render()

def evaluate_policy(env_name, algo, policy_path='./ray_trained_models/', n_episodes=100, seed=1):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    env = gym.make("financial_gym/"+env_name)
    test_agent, _ = load_policy(algo, env_name, policy_path, seed=1)
    rewards = []
    equity = []
    profits = []
    for episode in range(n_episodes):
        done = False
        observation = env.reset()
        while not done:
            # Compute the next action based on the trained policy
            test_action = test_agent.compute_single_action(observation)
            # Step the environment forward using the action from the trained policy
            observation, reward, done, info = env.step(test_action)
        
        equity.append(info['equity'])
        profits.append(info['profit'])
        rewards.append(reward)
    
    print('\n', '-'*50, '\n')
    print('Reward Mean:', np.mean(rewards))
    print('Reward Std:', np.std(rewards))
    print('\n', '-'*50, '\n')
    
    print('\n', '-'*50, '\n')
    print('Equity Mean:', np.mean(equity))
    print('Equity Std:', np.std(equity))
    print('\n', '-'*50, '\n')

    print('\n', '-'*50, '\n')
    print('Profit Mean:', np.mean(profits))
    print('Profit Std:', np.std(profits))
    print('\n', '-'*50, '\n')

    sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reinforcement Learning Utility for Financial Gym')
    parser.add_argument('--env', default='TradeWorld-v0',
                        help='Environment to train on (default: TradeWorld-v0)')
    parser.add_argument('--algo', default='ppo',
                        help='Reinforcement learning algorithm from ray[rllib]')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Used to train new policies')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Used to render a single rollout of a trained policy')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Used to evaluate trained policies over n_episodes')
    parser.add_argument('--train_timesteps', type=int, default=10000,
                        help='Number of simulation timesteps to train a policy (default: 10000)')
    parser.add_argument('--save_dir', default='./ray_trained_models/',
                        help='Directory to save trained policy in (default ./ray_trained_models/)')
    parser.add_argument('--policy_path', default='./ray_trained_models/',
                        help='Path name to saved policy checkpoint (NOTE: Use this to continue training an existing policy, or to evaluate a trained policy)')
    parser.add_argument('--render_episodes', type=int, default=1,
                        help='Number of rendering episodes (default: 1)')
    parser.add_argument('--eval_episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    args = parser.parse_args()

    if args.train:
        train_policy(args.env, args.algo, timesteps_total=args.train_timesteps, save_dir=args.save_dir, policy_path=args.policy_path, seed=args.seed)
    if args.render:
        render_policy(args.env, args.algo, policy_path=args.policy_path, n_episodes=args.render_episodes, seed=args.seed)
    if args.evaluate:
        evaluate_policy(args.env, args.algo, policy_path=args.policy_path, n_episodes=args.eval_episodes, seed=args.seed)


# 1. Save the policy in the right directory with correct name DONE
# 2. Add function to load policy and keep training (train with live data too) DONE
# 3. Add function to render with historical data as well as live data
# 4. Add function to evaluate the policy (evaluate with historical new data as well as live data too), probably not possible with live/doesnt make sense (useful to check possible huge losses)
