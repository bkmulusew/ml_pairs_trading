from models import FinancialForecastingModel
from utils import ModelConfig
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NHiTSModel, NBEATSModel, TCNModel, TransformerModel

class DartsFinancialForecastingModel(FinancialForecastingModel):
    """A financial forecasting model based on the Darts library."""
    def __init__(self, model_name, data_processor, model_config):
        self.data_processor = data_processor
        self.scaler = None
        self.model_config = model_config
        self.model = self.initalize_model(model_name)

    def initalize_model(self, model_name):
        """Creates the model."""
        if model_name == "nbeats":
            return NBEATSModel(
                input_chunk_length=self.model_config.INPUT_CHUNK_LENGTH,
                output_chunk_length=self.model_config.OUTPUT_CHUNK_LENGTH,
                num_layers=5,
                num_blocks=1,
                num_stacks=2,
                layer_widths=512,
                dropout=0.2,
                n_epochs=self.model_config.N_EPOCHS,
                batch_size=self.model_config.BATCH_SIZE,
                model_name="nbeats",
                optimizer_kwargs={"lr": 0.0001},
                pl_trainer_kwargs={
                "accelerator": "gpu",
                "devices": [0]
                },
            )

        elif model_name == "nhits":
            return NHiTSModel(
                input_chunk_length=self.model_config.INPUT_CHUNK_LENGTH,
                output_chunk_length=self.model_config.OUTPUT_CHUNK_LENGTH,
                num_layers=5,
                num_blocks=1,
                num_stacks=2,
                layer_widths=512,
                dropout=0.2,
                n_epochs=self.model_config.N_EPOCHS,
                batch_size=self.model_config.BATCH_SIZE,
                model_name="nhits",
                optimizer_kwargs={"lr": 0.0001},
                pl_trainer_kwargs={
                "accelerator": "gpu",
                "devices": [0]
                },
            )

        elif model_name == "transformer":
            return TransformerModel(
                input_chunk_length=self.model_config.INPUT_CHUNK_LENGTH,
                output_chunk_length=self.model_config.OUTPUT_CHUNK_LENGTH,
                n_epochs=self.model_config.N_EPOCHS,
                batch_size=self.model_config.BATCH_SIZE,
                nhead=4,
                d_model=256,
                num_encoder_layers = 3,
                num_decoder_layers = 3,
                dim_feedforward = 16,
                norm_type = "LayerNormNoBias",
                dropout=0.2,
                model_name="transformer",
                optimizer_kwargs={"lr": 0.0001},
                pl_trainer_kwargs={
                "accelerator": "gpu",
                "devices": [0]
                },
            )

        elif model_name == "tcn":
            return TCNModel(
                input_chunk_length=self.model_config.INPUT_CHUNK_LENGTH,
                output_chunk_length=self.model_config.OUTPUT_CHUNK_LENGTH,
                kernel_size = 3,
                num_filters = 64,
                num_layers = 8,
                dilation_base = 2,
                n_epochs=self.model_config.N_EPOCHS,
                batch_size=self.model_config.BATCH_SIZE,
                weight_norm = True,
                model_name="tcn",
                dropout = 0.2,
                optimizer_kwargs={"lr": 0.0001},
                pl_trainer_kwargs = {
                    "accelerator": "gpu",
                    "devices": [0]
                }
            )

        else:
            raise ValueError("Invalid model name.")


    def split_and_scale_data(self, train_ratio=0.5, validation_ratio=0.1):
        """Splits the data into training, validation, and test sets and applies scaling."""
        series = self.data_processor.get_ratio_time_series()

        num_observations = len(series)
        train_end_index = int(num_observations * train_ratio)
        validation_end_index = int(num_observations * (train_ratio + validation_ratio))

        train_series = series[:train_end_index]
        val_series = series[train_end_index:validation_end_index]
        test_series = series[validation_end_index:]

        # Scaling the data
        self.scaler = Scaler()
        train_series_scaled = self.scaler.fit_transform(train_series)
        valid_series_scaled = self.scaler.transform(val_series)
        test_series_scaled = self.scaler.transform(test_series)

        return train_series_scaled, valid_series_scaled, test_series_scaled

    def train(self, train_series, validation_series):
        """Trains the model."""
        self.model.fit(train_series, val_series=validation_series, verbose=True)

    def predict_future_values(self, test_series):
        """Makes future value predictions based on the test series."""
        return self.model.predict(self.model_config.OUTPUT_CHUNK_LENGTH, series=test_series)

    def generate_predictions(self, test_series):
        """Generates predictions for each window of the test series."""
        transformed_series = []

        for i in range(len(test_series) - self.model_config.INPUT_CHUNK_LENGTH):
            transformed_series.append(test_series[i: i + self.model_config.INPUT_CHUNK_LENGTH])

        pred_series = self.predict_future_values(transformed_series)

        predicted_values = []
        for pred in pred_series:
            predicted_values.append(pred.values()[0][0])

        predicted_df = pd.DataFrame(predicted_values, columns=['predicted'])
        tseries_predicted = TimeSeries.from_dataframe(predicted_df, value_cols='predicted')
        tseries_predicted = self.scaler.inverse_transform(tseries_predicted).pd_series(copy=True)
        predicted_values = tseries_predicted.values.tolist()

        return predicted_values

    def get_true_values(self, test_series):
        """Retrieves true values from the test series after scaling back."""
        test_series_inverse = self.scaler.inverse_transform(test_series)
        true_df = test_series_inverse.pd_series(copy=True)
        true_values = true_df[self.model_config.INPUT_CHUNK_LENGTH:].values.tolist()
        return true_values