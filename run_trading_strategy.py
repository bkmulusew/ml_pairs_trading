from utils import ModelConfig
from data_processing import DataProcessor
from models import DartsFinancialForecastingModel
from models import TfFinancialForecastingModel
from simulator import TradingSimulator
from metrics import ModelEvaluationMetrics
from matplotlib import pyplot as plt
import random
import numpy as np
from models import MultiAgentReplayBuffer
from models import MADDPG
from strategies import RLTradingStrategy
import argparse

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

def float_list(strings):
    return [float(item) for item in strings.split(',')]

def run_sl_based_trading_strategy(model_name, model_config, trade_thresholds):
    eval_metrics = ModelEvaluationMetrics()

    # Initialize a DataProcessor instance to preprocess and manage the dataset.
    dataProcessor = DataProcessor(model_config)

    # Initialize a trading simulator.
    trading_simulator = TradingSimulator()

    # Instantiate a financial forecasting model, train, evaluate, and predict.
    if model_name == 'bilstm':
        predictor = TfFinancialForecastingModel(model_name, dataProcessor, model_config)
        processed_data = predictor.split_and_scale_data()
        predictor.train(processed_data['x_train'], processed_data['y_train'], processed_data['x_valid'], processed_data['y_valid'])
        generated_values = predictor.generate_predictions(processed_data['x_test'], processed_data['y_test'])
        predicted_values = generated_values['predicted_values']
        true_values = generated_values['true_values']
    else:
        predictor = DartsFinancialForecastingModel(model_name, dataProcessor, model_config)
        train_series, valid_series, test_series = predictor.split_and_scale_data()
        predictor.train(train_series, valid_series)
        predicted_values = predictor.generate_predictions(test_series)
        true_values = predictor.get_true_values(test_series)

    # Plot the actual and predicted values using matplotlib to visualize the model's performance.
    plt.plot(true_values, color = 'blue', label = 'True')
    plt.plot(predicted_values, color = 'red', label = 'Prediction')
    plt.title('True and Predicted Values')
    plt.xlabel('Observations')
    plt.ylabel('Ratio')
    plt.legend()
    plt.show()

    # Determine the size of the test dataset to simulate trading strategies.
    test_size = len(true_values)

    # Calculate the prediction error using the actual and predicted values.
    prediction_error = eval_metrics.calculate_prediction_error(predicted_values, true_values)
    print(f"Prediction Error: {prediction_error}")
    print (f"\n")

    # Retrieve the numerator and denominator prices for the test dataset from the DataProcessor.
    numerator_prices, denominator_prices = dataProcessor.get_test_columns(test_size)

    # Simulate trading strategies using the actual and predicted values, along with the numerator and
    # denominator prices, to assess the financial performance of the forecasting model.
    trading_simulator.simulate_trading_with_strategies(true_values, predicted_values, numerator_prices, denominator_prices, trade_thresholds)

def run_rl_based_trading_strategy(model_config):

    # Initialize a DataProcessor instance to preprocess and manage the dataset.
    dataProcessor = DataProcessor(model_config)
    train_states_space, train_next_states_space, train_new_spread, train_new_price = dataProcessor.compute_states()
    test_states_space, _, test_new_spread, test_new_price = dataProcessor.compute_states(train=False)

    n_actors = 3
    actor_dims = []
    for i in range(n_actors):
        actor_dims.append(train_states_space[0].shape[0])
    critic_dims = sum(actor_dims)
    action_dims = [6, 6, 2]

    # action space is a list of arrays, assume each agent has same action space
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_actors, action_dims,
                            alpha=0.0001, beta=0.0001, chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(20000, critic_dims, actor_dims,
                        action_dims, n_actors, batch_size=256)

    PRINT_INTERVAL = 1
    EPISODES = 50
    MAX_STEPS = 15000
    total_steps = 0

    print("---Train---")
    for i in range(EPISODES):
        trading_strategy = RLTradingStrategy(transaction_cost=0)
        state_space_index = random.randint(0, len(train_states_space) - 1)
        obs = [train_states_space[state_space_index]] * n_actors
        done = [False] * n_actors
        episode_step = 0
        while not any(done):
            actions = maddpg_agents.choose_action(obs)
            obs_ = [train_next_states_space[state_space_index]] * n_actors
            reward = trading_strategy.reward(train_new_spread, actions, train_new_price, state_space_index)
            reward = [reward] * n_actors
            done = [False] * n_actors

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS:
                done = [True] * n_actors

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            maddpg_agents.learn(memory)

            obs = obs_

            total_steps += 1
            episode_step += 1
            state_space_index = (state_space_index + 1) % len(train_states_space)

        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'total profit {:.1f}'.format(trading_strategy.total_profit_or_loss))

    print("---Test---")
    trading_strategy = RLTradingStrategy(transaction_cost=0)
    state_space_index = 0
    obs = [test_states_space[state_space_index]] * n_actors
    while state_space_index < len(test_states_space):
        actions = maddpg_agents.choose_action(obs)
        reward = trading_strategy.reward(test_new_spread, actions, test_new_price, state_space_index)
        obs = [test_states_space[(state_space_index + 1) % len(test_states_space)]] * n_actors
        state_space_index += 1

    print('total profit {:.1f}'.format(trading_strategy.total_profit_or_loss))

def run(args):

    model_config = ModelConfig()
    model_config.INPUT_CHUNK_LENGTH = args.input_chunk_length
    model_config.OUTPUT_CHUNK_LENGTH = args.output_chunk_length
    model_config.N_EPOCHS = args.n_epochs
    model_config.BATCH_SIZE = args.batch_size
    model_config.TRAIN_RATIO = args.train_ratio
    model_config.DATA_FILE_PATH = args.data_path

    model_name = args.sl_model
    thresholds = [float(threshold) for threshold in args.thresholds.split(',')]

    if args.rl:
        run_rl_based_trading_strategy(model_config)
    if args.sl:
        run_sl_based_trading_strategy(model_name, model_config, thresholds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl", action="store_true", default=False, help="Enable the reinforcement learning (RL) based model.")
    parser.add_argument("--sl", action="store_true", default=False, help="Enable the supervised learning (SL) based model.")
    parser.add_argument(
        "--sl_model",
        type=str,
        default="tcn",
        help="Specify the supervised learning model to use. Supported models include 'bilstm' for Bidirectional LSTM, \
            'nbeats' for NBEATS, 'nhits' for NHiTS, 'transformer' for Transformer, and 'tcn' for Temporal Convolutional Network. \
            Default is 'tcn'."
    )
    parser.add_argument("--input_chunk_length", type=int, default=50, help="Length of the input sequences.")
    parser.add_argument("--output_chunk_length", type=int, default=1, help="Length of the output sequences.")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training.")
    parser.add_argument("--train_ratio", type=float, default=0.5, help="Ratio of training data used in the train/test split.")
    parser.add_argument("--data_path", type=str, default="", help="Path to the training data.", required=True)
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0,0.00025,0.0005,0.001",
        help="Specify a list of threshold values for trading. Provide the values as a comma-separated list of size 4. \
            For example, use '--threshold 0,0.00025,0.0005,0.001' to set thresholds at 0, 0.00025, 0.0005, and 0.001."
    )

    args = parser.parse_args()
    run(args)