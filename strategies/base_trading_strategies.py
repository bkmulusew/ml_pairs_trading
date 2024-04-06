from abc import ABC, abstractmethod

class TradingStrategies(ABC):

    @abstractmethod
    def evaluate_trade(self):
        pass

    @abstractmethod
    def calculate_sharpe_ratios(self):
        pass