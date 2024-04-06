import numpy as np

class ModelEvaluationMetrics:
    def calculate_smape(self, actual, predicted):
        """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
        actual, predicted = np.asarray(actual), np.asarray(predicted)
        denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
        diff = np.abs(actual - predicted) / denominator
        diff[denominator == 0] = 0.0  # avoid division by zero
        return 100 * np.mean(diff)

    def calculate_mape(self, actual, predicted):
        """Calculates the Mean Absolute Percentage Error (MAPE)."""
        actual, predicted = np.asarray(actual), np.asarray(predicted)
        diff = np.abs((actual - predicted) / actual)
        diff[actual == 0] = 0.0  # avoid division by zero
        return 100 * np.mean(diff)

    def calculate_mase(self, actual, predicted):
        """Calculates the Mean Absolute Scaled Error (MASE)."""
        actual, predicted = np.asarray(actual), np.asarray(predicted)
        n = len(actual)
        d = np.abs(np.diff(actual)).sum() / (n - 1)
        errors = np.abs(actual - predicted)
        return errors.mean() / d

    def calculate_rmse(self, actual, predicted):
        """Calculates the Root Mean Squared Error (RMSE)."""
        actual, predicted = np.asarray(actual), np.asarray(predicted)
        return np.sqrt(((predicted - actual) ** 2).mean())

    def calculate_prediction_error(self, predicted_values, true_values):
        """Calculates the prediction error."""
        smape = self.calculate_smape(predicted_values, true_values)
        mape = self.calculate_mape(predicted_values, true_values)
        mase = self.calculate_mase(predicted_values, true_values)
        rmse = self.calculate_rmse(predicted_values, true_values)
        return {'RMSE': rmse, 'MASE': mase, 'MAPE': mape, 'sMAPE': smape}