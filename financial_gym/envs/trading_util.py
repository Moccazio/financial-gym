from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class TradingUtil():
    
    def __init__(self, client_type='historical', paper=True):
        if client_type == 'historical':
            # Get Historical Client
            self.historical_client = CryptoHistoricalDataClient()
        elif client_type == 'live':
            # Get Trading Client
            API_KEY = "PKXKF95OOZ2VE7EM65PT"
            SECRET_KEY = "3d3baEV3wLoBX6rxUxrE4THDFybElMQnvpiPnWcn"
            self.trading_client = TradingClient(API_KEY, SECRET_KEY, paper=paper)

    def get_historical_data(self, start_time, end_time):
        start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S %z')
        end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S %z')

        request_params = CryptoBarsRequest(
                            symbol_or_symbols=["BTC/USD"],
                            timeframe=TimeFrame.Minute,
                            start=start_time,
                            end=end_time
                    )

        historical_bars = self.historical_client.get_crypto_bars(request_params)

        # convert to dataframe
        historical_bars_df = historical_bars.df

        historical_bars_df.reset_index(inplace=True)
        return historical_bars_df

    def get_live_data(self):
        pass

    def get_data_points(self, bars_df, index):
        symbol = bars_df.iloc[index]['symbol']
        timestamp = bars_df.iloc[index]['timestamp']
        open_data_point = bars_df.iloc[index]['open']
        close_data_point = bars_df.iloc[index]['close']
        high_data_point = bars_df.iloc[index]['high']
        low_data_point = bars_df.iloc[index]['low']
        volume_data_point = bars_df.iloc[index]['volume']
        trade_count_data_point = bars_df.iloc[index]['trade_count']
        vwap_data_point = bars_df.iloc[index]['vwap']

        return symbol, timestamp, open_data_point, close_data_point, high_data_point, low_data_point, volume_data_point, trade_count_data_point, vwap_data_point
        

    def buy(self, symbol="BTC/USD", quantity=0, transaction_type='simulation'):
        if transaction_type in ['paper', 'real']:
            # Setting parameters for buy order
            market_order_data = MarketOrderRequest(
                                symbol=symbol,
                                quantity=1,
                                side=OrderSide.BUY,
                                time_in_force=TimeInForce.UTC
                            )
            market_order = self.trading_client.submit_order(market_order_data)
        elif transaction_type == 'simulation':
            pass

    def sell(self, symbol="BTC/USD", quantity=0, transaction_type='simulation'):
        if transaction_type in ['paper', 'real']:
            # Setting parameters for sell order
            market_order_data = MarketOrderRequest(
                                symbol=symbol,
                                quantity=1,
                                side=OrderSide.SELL,
                                time_in_force=TimeInForce.UTC
                            )
            market_order = self.trading_client.submit_order(market_order_data)
        elif transaction_type == 'simulation':
            pass