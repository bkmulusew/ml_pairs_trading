from models import FinancialForecastingModel
from utils import ModelConfig
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MultiHeadAttention, Dense, Dropout, Bidirectional, LSTM, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate

class TfFinancialForecastingModel(FinancialForecastingModel):
    """A financial forecasting model based on the Tensorflow library."""
    def __init__(self, mode_name, data_processor, model_config):
        self.data_processor = data_processor
        self.scaler = None
        self.model_config = model_config
        self.model = self.initalize_model(mode_name)

    def initalize_model(self, model_name):
        """Creates the model."""
        if model_name == "bilstm":
            inp = Input(shape = (self.model_config.INPUT_CHUNK_LENGTH, 1))

            x = Bidirectional(LSTM(128, return_sequences=True))(inp)
            x = Bidirectional(LSTM(128, return_sequences=True))(x)

            x = MultiHeadAttention(num_heads=4, key_dim=128, dropout=0.1)(x, x, x)

            avg_pool = GlobalAveragePooling1D()(x)
            max_pool = GlobalMaxPooling1D()(x)
            conc = concatenate([avg_pool, max_pool])
            conc = Dense(512, activation="relu")(conc)
            x = Dense(1, activation="sigmoid")(conc)

            model = Model(inputs = inp, outputs = x)
            model.compile(
                loss = "mean_squared_error",
                optimizer = Adam(learning_rate=0.001))

            return model

        else:
            raise ValueError("Invalid model name.")

    def split_and_scale_data(self, train_ratio=0.5, validation_ratio=0.1):
        """Splits the data into training, validation, and test sets and applies scaling."""
        series = self.data_processor.load_and_prepare_data()
        series = series['ratio'].values
        series = series.reshape(-1, 1)

        num_observations = len(series)
        train_end_index = int(num_observations * train_ratio)
        validation_end_index = int(num_observations * (train_ratio + validation_ratio))

        train_data = series[:train_end_index]
        valid_data = series[train_end_index:validation_end_index]
        test_data = series[validation_end_index:]

        self.scaler = MinMaxScaler((0, 1))
        train_data = self.scaler.fit_transform(train_data)
        valid_data = self.scaler.transform(valid_data)
        test_data = self.scaler.transform(test_data)

        X_train = []
        y_train = []
        X_valid = []
        y_valid = []
        X_test = []
        y_test = []

        for i in range(self.model_config.INPUT_CHUNK_LENGTH, len(train_data)):
            X_train.append(train_data[i - self.model_config.INPUT_CHUNK_LENGTH : i])
            y_train.append(train_data[i + (self.model_config.OUTPUT_CHUNK_LENGTH - 1)])

        for i in range(self.model_config.INPUT_CHUNK_LENGTH, len(valid_data)):
            X_valid.append(valid_data[i - self.model_config.INPUT_CHUNK_LENGTH : i])
            y_valid.append(valid_data[i + (self.model_config.OUTPUT_CHUNK_LENGTH - 1)])

        for i in range(self.model_config.INPUT_CHUNK_LENGTH, len(test_data)):
            X_test.append(test_data[i - self.model_config.INPUT_CHUNK_LENGTH : i])
            y_test.append(test_data[i + (self.model_config.OUTPUT_CHUNK_LENGTH - 1)])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_train, y_train = shuffle(X_train, y_train)

        X_valid, y_valid = np.array(X_valid), np.array(y_valid)
        X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))

        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        return {'x_train': X_train, 'y_train': y_train, 'x_valid': X_valid, 'y_valid': y_valid, 'x_test': X_test, 'y_test': y_test}

    def train(self, x_train, y_train, x_valid, y_valid):
        """Trains the model."""
        self.model.fit(x_train, y_train, batch_size=self.model_config.BATCH_SIZE, epochs=self.model_config.N_EPOCHS, validation_data=(x_valid, y_valid))

    def predict_future_values(self, x_test):
        """Makes future value predictions based on the test series."""
        return self.model.predict(x_test)

    def generate_predictions(self, x_test, y_test):
        """Generates predictions for each window of the test series."""
        pred_series = self.predict_future_values(x_test)
        pred_series = self.scaler.inverse_transform(pred_series)

        # Storing the predictions
        predicted_values = pred_series.tolist()
        true_values = self.scaler.inverse_transform(y_test).tolist()
        return {'predicted_values': predicted_values, 'true_values': true_values}