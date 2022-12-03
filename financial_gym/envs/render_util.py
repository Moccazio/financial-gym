import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.patches as mpatches
import warnings
import pandas as pd

class RenderUtil():
    # Ignore warnings
    warnings.filterwarnings(action='ignore', category=DeprecationWarning) 

    def __init__(self, pause_time=2):
        self.pause_time = pause_time

        # Rendering stuff
        self.fig = mpf.figure(figsize=(12,9))
        self.ax1 = self.fig.add_subplot(2,1,1) # Candlestick
        self.ax_volume = self.ax1.twinx()
        self.ax2 = self.fig.add_subplot(2,3,4) # Profit 
        self.ax3 = self.fig.add_subplot(2,3,5) # Portfolio 
        self.ax4 = self.fig.add_subplot(2,3,6) # Equity 

        self.ax2.grid(False)
        self.ax3.grid(False)
        self.ax4.grid(False)
    
    def update_plots(self, data):
        self.ax1.clear()
        self.ax_volume.clear()
        self.ax1.grid(False)
        self.ax_volume.grid(False)

        data_copy = data.copy()

        data_df = pd.DataFrame(data_copy).set_index('timestamp')

        # Profit plot
        curent_wallet = data_df['wallet_history'].iloc[-1]
        initial_wallet = data_df['wallet_history'].iloc[0]
        self.ax2.set_title('Current Profit: $' + str(round((curent_wallet - initial_wallet), 4)))
        self.ax2.plot(range(len(data_df)), data_df['wallet_history'], color='black')

        # Portfolio plot
        current_portfolio = data_df['portfolio_history'].iloc[-1]
        self.ax3.set_title('Current Portfolio Size: ' + str(round(current_portfolio,4)))
        self.ax3.plot(range(len(data_df)), data_df['portfolio_history'], color='black')

        # Equity plot
        current_equity = data_df['equity_history'].iloc[-1]
        self.ax4.set_title('Current Equity: $' + str(round(current_equity, 4)))
        self.ax4.plot(range(len(data_df)), data_df['equity_history'], color='black')

        # Candlestick and Volume plot
        ap = [mpf.make_addplot(data_df['buy_history'], type='scatter', marker='^', markersize=100, color='g', ax=self.ax1),
              mpf.make_addplot(data_df['sell_history'], type='scatter', marker='v', markersize=100, color='r', ax=self.ax1)]
        # mpf.plot(data_df,type='candle',ax=self.ax1,axtitle='Candlestick Graph & Volume',xrotation=15, volume=self.ax_volume, addplot=ap, style='classic')
        mpf.plot(data_df,type='candle',ax=self.ax1,axtitle='Candlestick Graph & Volume',xrotation=15, addplot=ap, style='classic')

        patch_buy = mpatches.Patch(color='g', label='Buy')
        patch_sell = mpatches.Patch(color='r', label='Sell')          
        self.ax1.legend(handles=[patch_buy, patch_sell], loc='upper left')
        self.ax1.yaxis.set_label_position("left")
        self.ax1.yaxis.tick_left()

        # Update plot
        plt.pause(self.pause_time)
