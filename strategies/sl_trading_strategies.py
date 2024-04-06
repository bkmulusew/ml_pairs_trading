from .base_trading_strategies import TradingStrategies
from metrics import ConfusionMatrix
import numpy as np

class SLTradingStrategy(TradingStrategies):
    """Trading Strategy for Supervised Learning based models."""
    def __init__(self, strategy_name, trade_thresholds):
        self.strategy_name = strategy_name
        self.trade_thresholds = trade_thresholds
        self.all_profit_or_loss = [[] for _ in range(len(trade_thresholds))]
        self.sharpe_ratios = []
        self.total_profit_or_loss = [0.0] * len(trade_thresholds)
        self.no_trade = [0] * len(trade_thresholds)
        self.num_trade = [0] * len(trade_thresholds)
        self.confusion_matrices = [ConfusionMatrix() for _ in range(len(trade_thresholds))]

    def evaluate_trade(self, i, prev_ratio, curr_ratio, predicted_next_ratio, actual_next_ratio, numerator_prices, denominator_prices):
        """Evaluates the trade at the current time step."""
        for j, threshold in enumerate(self.trade_thresholds):
            predicted_ratio_change = predicted_next_ratio - curr_ratio
            base_ratio_change = curr_ratio - prev_ratio
            ground_truth_ratio_change = actual_next_ratio - curr_ratio

            trade_direction = self.determine_trade_direction(base_ratio_change, predicted_ratio_change, ground_truth_ratio_change, threshold)

            profit = self.calculate_profit(i, trade_direction, curr_ratio, numerator_prices, denominator_prices)

            self.all_profit_or_loss[j].append(profit)
            self.total_profit_or_loss[j] += profit
            if trade_direction != 'no_trade':
                self.update_confusion_matrix(j, ground_truth_ratio_change, predicted_ratio_change, base_ratio_change)
                self.num_trade[j] += 1
            else:
                self.no_trade[j] += 1

    def calculate_profit(self, i, trade_direction, curr_ratio, numerator_prices, denominator_prices):
        """Calculates the profit or loss of the trade."""
        if trade_direction == 'buy_numerator':
            return (numerator_prices[i] - numerator_prices[i - 1]) + ((1 / curr_ratio) * (denominator_prices[i - 1] - denominator_prices[i]))
        elif trade_direction == 'buy_denominator':
            return ((1 / curr_ratio) * (denominator_prices[i] - denominator_prices[i - 1])) + (numerator_prices[i - 1] - numerator_prices[i])
        return 0

    def calculate_sharpe_ratios(self):
        """Calculates the Sharpe ratio for each threshold."""
        for j in range(len(self.all_profit_or_loss)):
            excess_returns = self.all_profit_or_loss[j]
            computed_mean = float(np.mean(excess_returns))
            computed_stddev = float(np.std(excess_returns))
            self.sharpe_ratios.append(computed_mean / computed_stddev)

    def time_series_bootstrap(self, profits, block_size, n_iterations):
        """Generates a bootstrap samples to estimate the standard deviation, standard error, and confidence interval."""
        n = len(profits)
        total_profits = []

        for _ in range(n_iterations):
            bootstrap_series = []
            while len(bootstrap_series) < n:
                start = np.random.randint(0, n)
                length = min(np.random.geometric(1.0 / block_size), n - len(bootstrap_series))
                bootstrap_series.extend(profits[start:start+length] if start+length <= n else profits[start:] + profits[:start+length-n])
            total_profits.append(sum(bootstrap_series[:n]))

        return total_profits

    def calculate_stats(self, multiDimProfits):
        """Calculates the standard deviation, standard error, and confidence interval for the total profits."""
        stat_standard_dev = []
        stat_standard_error = []
        stat_confidence_interval = []
        for threshold, profits in zip(self.trade_thresholds, multiDimProfits):
            block_size = int(len(profits) ** (1/3))
            n_iterations = 1000
            bootstrapped_total_profits = self.time_series_bootstrap(profits, block_size, n_iterations)
            bootstrap_std_dev = np.std(bootstrapped_total_profits)
            bootstrap_standard_error = bootstrap_std_dev / np.sqrt(n_iterations)
            confidence_interval = np.percentile(bootstrapped_total_profits, [2.5, 97.5])
            stat_standard_dev.append(bootstrap_std_dev)
            stat_standard_error.append(bootstrap_standard_error)
            stat_confidence_interval.append(confidence_interval)
        return {'standard_dev': stat_standard_dev, 'standard_error': stat_standard_error, 'confidence_interval': stat_confidence_interval}

    def update_confusion_matrix(self, j, ground_truth_ratio_change, predicted_ratio_change, base_ratio_change):
        """Updates the confusion matrix based on the ground truth and predicted changes."""
        if(self.strategy_name == 'mean reversion'):
            self.confusion_matrices[j].update(ground_truth_ratio_change, -1 * base_ratio_change)
        elif(self.strategy_name == 'pure forcasting'):
            self.confusion_matrices[j].update(ground_truth_ratio_change, predicted_ratio_change)
        elif(self.strategy_name == 'hybrid'):
            if ground_truth_ratio_change > 0 and (base_ratio_change < 0 and predicted_ratio_change > 0):
                self.confusion_matrices[j].true_positive += 1
            elif ground_truth_ratio_change < 0 and (base_ratio_change < 0 and predicted_ratio_change > 0):
                self.confusion_matrices[j].false_positive += 1
            elif ground_truth_ratio_change > 0 and (base_ratio_change > 0 and predicted_ratio_change < 0):
                self.confusion_matrices[j].false_negative += 1
            elif ground_truth_ratio_change < 0 and (base_ratio_change > 0 and predicted_ratio_change < 0):
                self.confusion_matrices[j].true_negative += 1
            elif (base_ratio_change == 0 or predicted_ratio_change == 0):
                self.confusion_matrices[j].no_change += 1

    def determine_trade_direction(self, base_ratio_change, predicted_ratio_change, ground_truth_ratio_change, threshold):
        """Determines the trade direction based on the base ratio change, the predicted ratio change, and ground truth ratio change."""
        if(self.strategy_name == 'mean reversion'):
            if base_ratio_change > threshold:
                return 'buy_denominator'
            elif base_ratio_change < -threshold:
                return 'buy_numerator'
            else:
                return 'no_trade'
        elif(self.strategy_name == 'pure forcasting'):
            if predicted_ratio_change > threshold:
                return 'buy_numerator'
            elif predicted_ratio_change < -threshold:
                return 'buy_denominator'
            else:
                return 'no_trade'
        elif(self.strategy_name == 'hybrid'):
            if base_ratio_change < -threshold and predicted_ratio_change > threshold:
                return 'buy_numerator'
            elif base_ratio_change > threshold and predicted_ratio_change < -threshold:
                return 'buy_denominator'
            else:
                return 'no_trade'
        elif(self.strategy_name == 'ground truth'):
            if ground_truth_ratio_change > threshold:
                return 'buy_numerator'
            elif ground_truth_ratio_change < -threshold:
                return 'buy_denominator'
            else:
                return 'no_trade'
        else:
            return 'no_trade'

    def display_total_profit(self):
        print(f"Total Profits - {self.strategy_name} strategy = {self.total_profit_or_loss}")
        print(f"Best Trade Threshold - {self.strategy_name} strategy = {self.trade_thresholds[self.total_profit_or_loss.index(max(self.total_profit_or_loss))]}")

    def display_profit_per_trade(self):
        print(f"Profits per Trade - {self.strategy_name} strategy = {[x / y for x, y in zip(self.total_profit_or_loss, self.num_trade)]}")

    def display_stat_total_profit(self):
        stat = self.calculate_stats(self.all_profit_or_loss)
        print(f"Total Profits Std Dev - {self.strategy_name} strategy = {stat['standard_dev']}")
        print(f"Total Profits Std Error - {self.strategy_name} strategy = {stat['standard_error']}")
        print(f"Total Profits Confidence - {self.strategy_name} strategy = {stat['confidence_interval']}")

    def display_stat_profit_per_trade(self):
        profit_per_trade = [[profit / num for profit in profits] for profits, num in zip(self.all_profit_or_loss, self.num_trade)]
        stat = self.calculate_stats(profit_per_trade)
        print(f"Profits per Trade Std Dev - {self.strategy_name} strategy = {stat['standard_dev']}")
        print(f"Profits per Trade Std Error - {self.strategy_name} strategy = {stat['standard_error']}")
        print(f"Profits per Trade Confidence Interval - {self.strategy_name} strategy = {stat['confidence_interval']}")

    def display_sharpe_ratios(self):
        print(f"Sharpe Ratios - {self.strategy_name} strategy = {self.sharpe_ratios}")
        print(f"Safest Trade Threshold - {self.strategy_name} strategy = {self.trade_thresholds[self.sharpe_ratios.index(max(self.sharpe_ratios))]}")

    def display_num_trades(self):
        print(f"{self.strategy_name} strategy: trades made = {self.num_trade}, trades not made = {self.no_trade}")

    def display_confusion_matrix(self):
        for j in range(len(self.confusion_matrices)):
            print(f"{self.strategy_name} strategy v/s ground truth for thresold = {self.trade_thresholds[j]}: TP = {self.confusion_matrices[j].true_positive}, FP = {self.confusion_matrices[j].false_positive}, FN = {self.confusion_matrices[j].false_negative}, TN = {self.confusion_matrices[j].true_negative}")