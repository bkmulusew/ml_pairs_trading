from matplotlib import pyplot as plt
import pandas as pd
from darts import TimeSeries
import numpy as np

class DataProcessor:
    def __init__(self, model_config):
        self.model_config = model_config
        self.data_df = None

    def load_and_prepare_data(self):
        """Loads the CSV file and calculates the ratio if not already done."""
        if self.data_df is None:
            self.data_df = pd.read_csv(self.model_config.DATA_FILE_PATH)
            numerator_column_name = self.data_df.columns[2]
            denominator_column_name = self.data_df.columns[1]
            self.data_df['ratio'] = self.data_df[numerator_column_name] / self.data_df[denominator_column_name]
            self.data_df['difference'] = self.data_df[numerator_column_name] - self.data_df[denominator_column_name]
        return self.data_df

    def get_ratio_time_series(self):
        """Converts the ratio column to a time series object."""
        df = self.load_and_prepare_data()
        return TimeSeries.from_dataframe(df, value_cols='ratio')

    def get_test_columns(self, test_size):
        """Retrieves the test dataset's numerator and denominator columns."""
        self.load_and_prepare_data()
        numerator_column_name = self.data_df.columns[2]
        denominator_column_name = self.data_df.columns[1]
        return (self.data_df[numerator_column_name].values)[-test_size:], (self.data_df[denominator_column_name].values)[-test_size:]

    def plot_ratio(self):
        """Plots the ratio time series."""
        df = self.load_and_calculate_ratio()
        numerator = df.columns[2]
        denominator = df.columns[1]
        plt.title(f'Ratio of {numerator} to {denominator}', color='red')
        plt.plot(df['ratio'], color='blue', label="Ratio")
        plt.xlabel('Observation')
        plt.ylabel('Ratio Value')
        plt.legend()
        plt.show()

    def calculate_spread(self):
        """Calculate the spread for each element in the difference list using a window of size w."""
        spread = []  # Initialize the spread list
        price = []
        df = self.load_and_prepare_data()
        difference = df['difference'].values.tolist()
        w = self.model_config.INPUT_CHUNK_LENGTH

        # Loop through the difference list
        for i in range(w, len(difference)):
            # Extract the window elements
            window_elements = difference[i - w : i]
            # Calculate the mean and std of the window
            window_mean = np.mean(window_elements)
            window_std = np.std(window_elements)   # Paper is not clear on how to calculate decentralized sequence
            # Calculate the spread for the current element i
            if window_std > 0:
                current_spread = (difference[i - 1] - window_mean) / window_std
            else:
                # Handle the division by zero case as needed
                current_spread = 0
            # Append the spread calculation to the spread list
            price.append(difference[i - 1])
            spread.append(current_spread)

        return spread, price

    def compute_states(self, train=True):
        """Computes states for RL agents."""
        # Initialize empty lists to hold the state[t] and next_state[t] values
        spread, price = self.calculate_spread()
        split_index = int(len(spread) * self.model_config.TRAIN_RATIO)

        if train:
            spread, price = spread[: split_index], price[: split_index]
        else:
            spread, price = spread[split_index :], price[split_index :]

        w = self.model_config.INPUT_CHUNK_LENGTH
        states = []
        next_states = []
        new_spread = []
        new_price = []

        # Iterate over the spread array to calculate each state and next_state
        for t in range(w, len(spread) - 1):
            # Calculate state[t]
            state = np.array([spread[i] / spread[i - 1] if spread[i - 1] != 0 else 0 for i in range(t - w + 1, t + 1)])
            new_spread.append(spread[t])
            new_price.append(price[t])
            states.append(state)

            # Calculate next_state[t]
            # For next_state, we need to ensure that t+1 does not exceed the array bounds
            next_state = np.array([spread[i] / spread[i - 1] if spread[i - 1] != 0 else 0 for i in range(t - w + 2, t + 2)])
            next_states.append(next_state)

        return states, next_states, new_spread, new_price