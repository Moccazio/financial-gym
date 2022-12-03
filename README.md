# Financial Gym


## Install

``` 
virtualenv venv
source venv/bin/activate
git clone https://github.com/gabriansa/financial-gym
cd financial-gym
pip3 install -e .
```

## Commands

Render environment (with random actions):
```
python3 -m financial_gym --env "TradeWorld-v0"
```

Train environment:
```
python3 -m financial_gym.ray_util --env "TradeWorld-v0" --train --train_timesteps 1000000
```

Render trained environment:
```
python3 -m financial_gym.ray_util --env "TradeWorld-v0" --render
```

Evaluate trained environment:
```
python3 -m financial_gym.ray_util --env "TradeWorld-v0" --evaluate --eval_episodes 50
```
