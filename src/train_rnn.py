import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import pretty_midi
from encoding import encoding_to_LSTM
from etude_rnn import EtudeRNN
import os
import time

absolute_path = "C:\\Users\\Andreas\\Documents\\CS397Pardo\\Project\\EtudeRNN\\"
model_path = "src\\LSTM_model"
npdata_filepath = "data\\classical_guitar_npdata\\"
try:
    list_songs = os.listdir(npdata_filepath)
except FileNotFoundError:
    npdata_filepath = absolute_path + npdata_filepath
    model_path = absolute_path + model_path
    list_songs = os.listdir(npdata_filepath)

class MidiDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __getitem__(self, item):
        return (self.x[item], self.y[item])

    def __len__(self):
        return len(self.x)

def train_LSTM(model, midi_dataset, training_set: str, lr=1e-4, batch_size=300):
    """
    Trains LSTM

    :param model: LSTM model to train
    :param training_dataset: training dataset
    :param lr: Learning Rate for optimizer
    :param batch_size: Batch Size for data loader
    :return:
    """
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    training_dataloader = DataLoader(midi_dataset, batch_size=batch_size)

    for epoch in range(model.n_steps):
        begin = time.time()
        print("Epoch = %i" % epoch)
        for features, targets in training_dataloader:
            if model.device == "cuda:0":
                features = features.cuda()
                targets = targets.cuda()
            model.zero_grad()
            out, _, _ = model(features)
            batch_loss = loss(out, targets)
            print(batch_loss)
            batch_loss.backward()
            optimizer.step()
        print("Epoch Duration: %f seconds"%(time.time() - begin))
        if epoch % 1 == 0:
            torch.save(model.state_dict(), model_path + training_set +"_epoch%i"%(epoch))
    return model

encoded_matrices = []
for path in list_songs:
    encoded_matrices.append(torch.from_numpy(np.load(npdata_filepath + path)[np.newaxis, :, :]).float())

sample_beats = 16
sample_time_steps = 24 * sample_beats
long_x = []
long_y = []
for matrix in encoded_matrices:
    for i in range(0, matrix.size(2) - sample_time_steps - 1):
        long_x.append(matrix[0, :, i:i+sample_time_steps])
        long_y.append(matrix[0, :, i+1: i+sample_time_steps+1])
long_midi_dataset = MidiDataset(long_x, long_y)

sample_beats = 1
sample_time_steps = 24 * sample_beats
small_x = []
small_y = []
for matrix in encoded_matrices:
    for i in range(0, matrix.size(2) - sample_time_steps - 1):
        small_x.append(matrix[0, :, i:i+sample_time_steps])
        small_y.append(matrix[0, :, i+1: i+sample_time_steps+1])
small_midi_dataset = MidiDataset(small_x, small_y)

LSTMmodel = EtudeRNN(50, n_steps=1)
if torch.cuda.is_available():
    LSTMmodel.set_device('cuda:0')

for i in range(50):
    LSTMmodel = train_LSTM(LSTMmodel, long_midi_dataset, "coarse")
    LSTMmodel = train_LSTM(LSTMmodel, small_midi_dataset, "fine")

torch.save(LSTMmodel.state_dict(), model_path + _final)

