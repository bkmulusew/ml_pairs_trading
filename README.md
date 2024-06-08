# Machine Learning Enhanced Pairs Trading

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Example Usage](#example-usage)
- [Contributors](#contributors)

## Introduction
In this project, we explore and apply various deep learning models to predict changes in the price ratio of closely related stock pairs. Our approach includes the utilization of the following deep learning models:
- Bidirectional Long Short-Term Memory (BiLSTM) with Attention
- Transformers
- N-BEATS (Neural Basis Expansion Analysis)
- N-HiTS (Neural Hierarchical Interpolation for Time Series)
- Temporal Convolutional Networks (TCNs)
- MADDPG (Multi-Agent Deep Deterministic Policy Gradient) with Three Actors and One Critic -- This implementation follows the TS-MADDPG framework as described in the paper "Improved pairs trading strategy using two-level reinforcement learning framework" by Zhizhao Xu and Chao Luo (https://www.sciencedirect.com/science/article/abs/pii/S0952197623013325).

Once the changes are predicted, different pairs trading strategies are used to yield profit. Our approach includes the utilization of the following pairs trading strategies:
- Reversion Strategy :
  - Case 1: If the ratio at time t is greater than the ratio at time t − 1, then sell the stock in the numerator and buy the stock in the denominator.
  - Case 2: If the ratio at time t is less than the ratio at time t − 1, then buy the stock in the numerator and sell the stock in the denominator.
  - Case 3: If the ratio at time t is equal to the ratio at time t − 1, then don't trade.
- Pure Forecasting Strategy:
  - Case 1: If the predicted ratio at time t + 1 is greater than the ratio at time t, then buy the stock in the numerator and sell the stock in the denominator.
  - Case 2: If the predicted ratio at time t + 1 is less than the ratio at time t, then sell the stock in the numerator and buy the stock in the denominator.
  - Case 3: If the predicted ratio at time t + 1 is equal to the ratio at time t, then don't trade.
- Hybrid Strategy:
  - Case 1: If the predicted ratio at time t + 1 is less than the ratio at time t and the ratio at time t is greater than the ratio at time t − 1, then sell the stock in the numerator and buy the stock in the denominator.
  - Case 2: If the predicted ratio at time t + 1 is greater than the ratio at time t and the ratio at time t is less than the ratio at time t − 1, then buy the stock in the numerator and sell the stock in the denominator.
  - Case 3: In other cases, don't trade

## Dataset
Before training or evaluating the model and running the trading simulation with different pairs trading strategies, it is necessary to have a dataset. The dataset should be in the following format:
| Time	                  | A	    | B     |
| :---                    | :---: | :---: |
| 2023 01 03 10 09 00.000	| 13.15	| 14.75 |
| 2023 01 03 10 10 00.000	| 13.1	| 14.68 |
| 2023 01 03 10 11 00.000	| 13.1	| 14.65 |
| 2023 01 03 10 12 00.000	| 13.1	| 14.63 |
| 2023 01 03 10 13 00.000	| 13.05	| 14.61 |
| 2023 01 03 10 14 00.000	| 13.05	| 14.6  |

In this dataset:
- Time represents the timestamp of each data point.
- A and B represent the prices of two related stocks.

The pair ratio p is calculated by dividing the price of stock A by the price of stock B:
  - p = A / B

This ratio will be used as the target variable in the model training and trading strategies.

The dataset used to train, validate, and test the models are found in the ml_pairs_trading/dataset directory.

## Example Usage
```bash
python run_trading_strategy.py --sl  --sl_model tcn --data_path /path/to/data --n_epochs 3
```

The full list of flags and options for the python script is as follows:
```
--rl: Enable the reinforcement learning (RL) based model.
--sl: Enable the supervised learning (SL) based model.
--sl_model: Specify the supervised learning model to use. Supported models include 'bilstm' for Bidirectional LSTM,
            'nbeats' for NBEATS, 'nhits' for NHiTS, 'transformer' for Transformer, and 'tcn' for Temporal Convolutional Network.
--input_chunk_length: Length of the input sequences.
--output_chunk_length: Length of the output sequences.
--n_epochs: Number of training epochs.
--batch_size: Batch size for training.
--train_ratio: Ratio of training data used in the train/test split. 1% of the data is used for validation.
--data_path: Path to the dataset.
--thresholds: Specify a list of threshold values for trading. Provide the values as a comma-separated list of size 4.
            For example, use '--threshold 0,0.00025,0.0005,0.001' to set thresholds at 0, 0.00025, 0.0005, and 0.001.
```

## Contributors:

1.   Sohail Hodarkar: sph8686@nyu.edu
2.   Beakal Lemeneh: beakalmulusew@gmail.com
