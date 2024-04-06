from abc import ABC, abstractmethod

class FinancialForecastingModel(ABC):

    @abstractmethod
    def initalize_model(self):
        pass

    @abstractmethod
    def split_and_scale_data(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict_future_values(self):
        pass

    @abstractmethod
    def generate_predictions(self):
        pass