from strategies import SLTradingStrategy

class TradingSimulator:
    def initialize_strategies(self, trade_thresholds):
        """Initializes the strategies."""
        pure_forcasting_strategy = SLTradingStrategy("pure forcasting", trade_thresholds)
        mean_reversion_strategy = SLTradingStrategy("mean reversion", trade_thresholds)
        hybrid_strategy = SLTradingStrategy("hybrid", trade_thresholds)
        ground_truth_strategy = SLTradingStrategy("ground truth", trade_thresholds)
        return pure_forcasting_strategy, mean_reversion_strategy, hybrid_strategy, ground_truth_strategy

    def simulate_trading_with_strategies(self, true_values, predicted_values, numerator_prices, denominator_prices, trade_thresholds):
        """Simulates trading with the strategies."""
        pure_forcasting_strategy, mean_reversion_strategy, hybrid_strategy, ground_truth_strategy = self.initialize_strategies(trade_thresholds)

        for i in range(2, len(true_values)):
            curr_ratio = true_values[i - 1]
            prev_ratio = true_values[i - 2]
            predicted_next_ratio = predicted_values[i]
            actual_next_ratio = true_values[i]

            pure_forcasting_strategy.evaluate_trade(i, prev_ratio, curr_ratio, predicted_next_ratio, actual_next_ratio, numerator_prices, denominator_prices)
            mean_reversion_strategy.evaluate_trade(i, prev_ratio, curr_ratio, predicted_next_ratio, actual_next_ratio, numerator_prices, denominator_prices)
            hybrid_strategy.evaluate_trade(i, prev_ratio, curr_ratio, predicted_next_ratio, actual_next_ratio, numerator_prices, denominator_prices)
            ground_truth_strategy.evaluate_trade(i, prev_ratio, curr_ratio, predicted_next_ratio, actual_next_ratio, numerator_prices, denominator_prices)

        pure_forcasting_strategy.calculate_sharpe_ratios()
        mean_reversion_strategy.calculate_sharpe_ratios()
        hybrid_strategy.calculate_sharpe_ratios()
        ground_truth_strategy.calculate_sharpe_ratios()

        print (f"_____Total Profits_____")
        pure_forcasting_strategy.display_total_profit()
        mean_reversion_strategy.display_total_profit()
        hybrid_strategy.display_total_profit()
        ground_truth_strategy.display_total_profit()
        print (f"\n")

        print (f"_____Profits per Trade_____")
        pure_forcasting_strategy.display_profit_per_trade()
        mean_reversion_strategy.display_profit_per_trade()
        hybrid_strategy.display_profit_per_trade()
        ground_truth_strategy.display_profit_per_trade()
        print (f"\n")

        print (f"_____Total Profits Statistics_____")
        pure_forcasting_strategy.display_stat_total_profit()
        mean_reversion_strategy.display_stat_total_profit()
        hybrid_strategy.display_stat_total_profit()
        ground_truth_strategy.display_stat_total_profit()
        print (f"\n")

        print (f"_____Profits per Trade Statistics_____")
        pure_forcasting_strategy.display_stat_profit_per_trade()
        mean_reversion_strategy.display_stat_profit_per_trade()
        hybrid_strategy.display_stat_profit_per_trade()
        ground_truth_strategy.display_stat_profit_per_trade()
        print (f"\n")

        print (f"_____Sharpe Ratios_____")
        pure_forcasting_strategy.display_sharpe_ratios()
        mean_reversion_strategy.display_sharpe_ratios()
        hybrid_strategy.display_sharpe_ratios()
        ground_truth_strategy.display_sharpe_ratios()
        print (f"\n")

        print (f"_____Number of Trades_____")
        pure_forcasting_strategy.display_num_trades()
        mean_reversion_strategy.display_num_trades()
        hybrid_strategy.display_num_trades()
        ground_truth_strategy.display_num_trades()
        print (f"\n")

        print (f"_____Confusion Matrix_____")
        pure_forcasting_strategy.display_confusion_matrix()
        mean_reversion_strategy.display_confusion_matrix()
        hybrid_strategy.display_confusion_matrix()
        print (f"\n")