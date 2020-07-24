import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import alpha_vantage as av 
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators

api_key = 'XFDSGKBCYAWM4BX3'

def get_data(symbl):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data_ts, meta_data_ts = ts.get_daily_adjusted(symbol=symbl, outputsize='full') # pylint: disable=unbalanced-tuple-unpacking
    print("Fetched.")
    data_ts = pd.DataFrame(data_ts)
    as_numpy = data_ts['4. close'].to_numpy()
    x = torch.flip(torch.from_numpy(as_numpy), dims=(0,)) \
            .unsqueeze(0) \
            .T
    return x

def get_sma(symbl):
    ts = TechIndicators(key=api_key, output_format='pandas')
    data_ts, meta_data_ts = ts.get_sma(symbol='SPY', interval='daily', time_period=20, series_type='close') # pylint: disable=unbalanced-tuple-unpacking
    print("Fetched.")
    data_ts = pd.DataFrame(data_ts)
    as_numpy = data_ts['SMA'].to_numpy()
    x = torch.flip(torch.from_numpy(as_numpy), dims=(0,)) \
            .unsqueeze(0) \
            .T
    return x

class ExperienceReplay:
    def __init__(self, input_dims, capacity):
        self.input_dims = input_dims
        self.capacity = capacity

        self.states = np.zeros((capacity, *input_dims), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.states_ = np.zeros((capacity, *input_dims), dtype=np.float32)
        #self.terminals = np.zeros(capacity, dtype=np.bool)
        self.store_counter = 0

    def store(self, s, a, r, s_):
        i = self.store_counter % self.capacity
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.states_[i] = s_
        #self.terminals[i] = done
        self.store_counter += 1

    def sample(self, batch_size=1):
        population_size = min(self.store_counter, self.capacity)
        sample = np.random.choice(population_size, size=batch_size, replace=False)
        states = self.states[sample]
        actions = self.actions[sample]
        rewards = self.rewards[sample]
        states_ = self.states_[sample]
        #terminals = self.terminals[sample]
        return (states, actions, rewards, states_)

def store_transition(replay_buffer, s, a, r, s_):
    replay_buffer.store(s, a, r, s_)    
    
def sample_memory(replay_buffer, batch_size=1, device=torch.device('cpu')):
    sample = replay_buffer.sample(batch_size)
    return tuple(map(lambda x: torch.tensor(x).to(device), sample))

def makeNN(in_size, out_size, device=torch.device('cpu')):
    return nn.Sequential(nn.LayerNorm(in_size),
                     nn.Linear(in_size,150),
                     nn.ReLU(),
                     nn.Linear(150,150),
                     #nn.LayerNorm(150),
                     nn.BatchNorm1d(150),
                     nn.ReLU(),
                     nn.Linear(150, 80),
                     nn.BatchNorm1d(80),
                     nn.ReLU(),
                     nn.Linear(80,20),
                     nn.BatchNorm1d(20), 
                     nn.ReLU(), 
                     nn.Linear(20,out_size)).to(device)