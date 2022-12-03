from gym.envs.registration import register
from ray.tune.registry import register_env
from financial_gym.envs.trading_world import TradeWorldEnv

env_names = ["TradeWorld"]

# Register environments
for env_name in env_names:
    id = 'financial_gym/%s-v0' % (env_name)
    register(
        id= id,
        entry_point='financial_gym.envs:%sEnv' % (env_name),
        max_episode_steps=200,
    )

select_env = "TradeWorld-v0"
register_env(select_env, lambda config: TradeWorldEnv())