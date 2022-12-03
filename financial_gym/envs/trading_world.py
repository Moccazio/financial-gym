import gym
from gym import spaces
import numpy as np
import random

import pandas as pd

import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from financial_gym.envs.trading_util import TradingUtil
from financial_gym.envs.render_util import RenderUtil

import warnings
import datetime


class TradeWorldEnv(gym.Env):

    def __init__(self, data_type='historical', N_past_obs=10):
        super(TradeWorldEnv, self).__init__()   
        self.N_past_obs = N_past_obs
        obs_len = self.N_past_obs*5+3
        obs_low_bound = -np.inf
        obs_high_bound = np.inf
        self.observation_space = spaces.Box(low=np.array([obs_low_bound]*obs_len), high=np.array([obs_high_bound]*obs_len))

        # We have 3 actions, corresponding to "Hold", "Buy", "Sell"
        self.action_space = spaces.Discrete(3)
        self._what_action = {
            0: 'Hold',
            1: 'Buy',
            2: 'Sell'}

        self.trading_util = TradingUtil(client_type='historical')
        
    def _get_obs(self):
        index = self.iteration + self.N_past_obs
        _, self.timestamp_dp, self.open_dp, self.close_dp, self.high_dp, self.low_dp, self.volume_dp, _, _ = self.trading_util.get_data_points(bars_df=self.historical_bars_df, index=index)

        observation = np.array([*self.data['open'][-self.N_past_obs:], 
                                *self.data['high'][-self.N_past_obs:], 
                                *self.data['low'][-self.N_past_obs:], 
                                *self.data['close'][-self.N_past_obs:], 
                                *self.data['volume'][-self.N_past_obs:],
                                self.wallet, 
                                self.portfolio, 
                                self.equity])

        # Collect obs data
        self.save_obs_data()

        return observation
        
    def reset(self):
        self.initial_wallet_size = 10000
        self.wallet = self.initial_wallet_size
        self.portfolio = 0
        self.equity = self.initial_wallet_size

        self.historical_bars_df = self.retrieve_data()
        
        # Initialize dictionary for collecting data
        timestamp_init_list = self.historical_bars_df.iloc[:self.N_past_obs]['timestamp'].tolist()
        open_init_list = self.historical_bars_df.iloc[:self.N_past_obs]['open'].tolist()
        close_init_list = self.historical_bars_df.iloc[:self.N_past_obs]['close'].tolist()
        high_init_list = self.historical_bars_df.iloc[:self.N_past_obs]['high'].tolist()
        low_init_list = self.historical_bars_df.iloc[:self.N_past_obs]['low'].tolist()
        volume_init_list = self.historical_bars_df.iloc[:self.N_past_obs]['volume'].tolist()
        self.data = {'timestamp': timestamp_init_list,
                     'open': open_init_list,
                     'close': close_init_list,
                     'high': high_init_list,
                     'low': low_init_list,
                     'volume' : volume_init_list,
                     'wallet_history': [self.wallet for _ in range(self.N_past_obs)],
                     'portfolio_history': [self.portfolio for _ in range(self.N_past_obs)],
                     'equity_history': [self.equity for _ in range(self.N_past_obs)],
                     'buy_history': [float('nan') for _ in range(self.N_past_obs)],
                     'sell_history': [float('nan') for _ in range(self.N_past_obs)]}

        self.max_iteration = len(self.historical_bars_df) - self.N_past_obs
        self.iteration = 0

        return self._get_obs()

    def step(self, action):
        # Take a step
        reward = self.take_step(action)

        self.iteration += 1

        # Collect step data
        self.save_step_data()

        # Get observations
        observation = self._get_obs()

        # Get done
        done = self.iteration >= self.max_iteration-1

        # Get info
        info = {'equity': self.equity,
                'portfolio': self.portfolio,
                'wallet': self.wallet,
                'profit': self.profit}

        return observation, reward, done, info

    def take_step(self, action):
        self.action_type = self._what_action[action]
        self.current_price = self.close_dp
        
        percentage_to_buy = 1
        percentage_to_sell = 1

        if self.action_type == "Buy":
            how_much_to_spend = self.wallet*percentage_to_buy
            self.portfolio += how_much_to_spend/self.current_price
            self.wallet += -how_much_to_spend

        step_profit = 0
        if self.action_type == "Sell":
            how_much_to_sell = self.portfolio*percentage_to_sell
            self.wallet += how_much_to_sell*self.current_price
            self.portfolio += -how_much_to_sell
            step_profit = self.wallet - self.data["wallet_history"][-1]


        self.equity = self.wallet + self.portfolio*self.current_price
        self.profit = self.wallet - self.initial_wallet_size

        # reward = self.equity --> Why not working?
        reward = step_profit

        return reward

    def render(self, mode=None):
        if self.iteration==0:
            self.render_util = RenderUtil()
        self.render_util.update_plots(data=self.data)

    def retrieve_data(self):
        start_time = "2020-12-21 00:00:00 +0000"
        # end_time = "2020-12-21 23:59:00 +0000"
        end_time = "2020-12-21 03:59:00 +0000"
        trading_util = TradingUtil(client_type='historical')
        self.historical_bars_df = trading_util.get_historical_data(start_time=start_time, end_time=end_time)
        return self.historical_bars_df

    def save_obs_data(self):
        # Collecting data
        self.data['timestamp'].append(self.timestamp_dp)
        self.data['open'].append(self.open_dp)
        self.data['close'].append(self.close_dp)
        self.data['high'].append(self.high_dp)
        self.data['low'].append(self.low_dp)
        self.data['volume'].append(self.volume_dp)
        self.data['wallet_history'].append(self.wallet)
        self.data['portfolio_history'].append(self.portfolio)
        self.data['equity_history'].append(self.equity)
        self.data['buy_history'].append(float('nan'))
        self.data['sell_history'].append(float('nan'))

    def save_step_data(self):
        self.data['buy_history'][-1] = self.current_price if (self.action_type=="Buy") else float('nan')
        self.data['sell_history'][-1] = self.current_price if (self.action_type=="Sell") else float('nan')