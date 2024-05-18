"""
Implementation of reward and trading policy from the paper titled "Improved pairs trading strategy using two-level reinforcement
    learning framework" by Zhizhao Xu and Chao Luo (https://www.sciencedirect.com/science/article/abs/pii/S0952197623013325)
"""

from .base_trading_strategies import TradingStrategies
import numpy as np

class RLTradingStrategy(TradingStrategies):
    """Trading Strategy for Reinforcement Learning based models."""
    def __init__(self, transaction_cost):
        self.current_position = None  # 'short', 'long', or None
        self.transaction_cost = transaction_cost
        self.pwt_short = 0  # Placeholder for short position price
        self.pwt_long = 0  # Placeholder for long position price
        self.all_profit_or_loss = []
        self.sharpe_ratios = []
        self.total_profit_or_loss = 0
        self.no_trade = 0
        self.num_trade = 0

    def execute_trade(self, spread, price, ol, sl, index):
        """Execute a trading action based on given spread and price data."""
        spread_wt = spread
        # Calculate transaction cost for this trade
        c = self.transaction_cost
        return_t = 0

        # Check for opening conditions
        if self.current_position is None and ol < abs(spread_wt) < sl:
            if spread_wt > 0:
                self.current_position = 'short'
                self.pwt_short = price[index]  # Assuming spread_wt as the entering price
            elif spread_wt < 0:
                self.current_position = 'long'
                self.pwt_long = price[index]  # Assuming spread_wt as the entering price
            return 0  # No return as this is an opening action

        # Check for closing conditions
        elif self.current_position == 'short' and spread_wt < 0:
            self.current_position = None  # Position closed
            return_t = self.pwt_short - price[index] - 2 * c
        elif self.current_position == 'long' and spread_wt > 0:
            self.current_position = None  # Position closed
            return_t = price[index] - self.pwt_long - 2 * c

        # Check for stop-loss conditions
        elif self.current_position == 'short' and spread_wt > sl:
            self.current_position = None  # Stop-loss triggered
            return_t = self.pwt_short - price[index] - 2 * c
        elif self.current_position == 'long' and spread_wt < -sl:
            self.current_position = None  # Stop-loss triggered
            return_t = price[index] - self.pwt_long - 2 * c

        self.evaluate_trade(return_t)

        if(self.pwt_long != 0):
            return 1000 * (return_t / self.pwt_long)
        else:
            return 1000 * (return_t / price[index])

    def reward(self, spread, action, price, index):
        """Reward for the RL agent"""
        open_threshold = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.0}
        close_threshold = {0: 1.5, 1: 2.0, 2: 2.5, 3: 3.0, 4: 3.5, 5: 4.0}
        spread_wt = spread[index]
        ol = open_threshold[np.argmax(action[0])]
        sl = close_threshold[np.argmax(action[1])]
        trade = (np.argmax(action[2]) == 1)

        if trade:
            return self.execute_trade(spread_wt, price, ol, sl, index)

        return 0

    def evaluate_trade(self, return_t):
        """Evaluates the trade at the current time step."""
        if(return_t != 0):
            self.total_profit_or_loss += return_t
            self.all_profit_or_loss.append(return_t)
            self.num_trade += 1
        else:
            self.no_trade += 1

    def calculate_sharpe_ratios(self):
        """Calculates the Sharpe ratio for each threshold."""
        for j in range(len(self.all_profit_or_loss)):
            excess_returns = self.all_profit_or_loss[j]
            computed_mean = float(np.mean(excess_returns))
            computed_stddev = float(np.std(excess_returns))
            self.sharpe_ratios.append(computed_mean / computed_stddev)