# Financial Gym


## Install

```
conda env = python 3.10 or higher
conda activate env 
git clone https://github.com/Moccazio/financial-gym
cd financial-gym
pip3 install -e .
```
or
```
conda env = python 3.10 or higher
conda activate env 
pip install git+https://github.com/Moccazio/financial-gym
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
