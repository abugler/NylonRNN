import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import pretty_midi
import configurations.configuration as config
from src.encoding import encoding_to_LSTM
from src.etude_rnn import EtudeRNN

absolute_path = "C:\\Users\\Andreas\\Documents\\CS397Pardo\\Project\\EtudeRNN\\"
data_path = "data\\Classical_Guitar_classicalguitarmidi.com_MIDIRip\\"
model_path = "src\\LSTM_model"
song_list_filepath = "data\\classical_guitar_training_set"
try:
    with open(song_list_filepath, 'r') as song_list_file:
        song_list = song_list_file.readlines()
except FileNotFoundError:
    data_path = absolute_path + data_path
    song_list_filepath = absolute_path + song_list_filepath
    model_path = absolute_path + model_path
    with open(song_list_filepath, 'r') as song_list_file:
        song_list = ''.join(song_list_file.read()).split('\n')

class MidiDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __getitem__(self, item):
        return (self.x[item], self.y[item])

    def __len__(self):
        return len(self.x)

def train_LSTM(model, encoded_matrices, lr=1e-4, batch_size=50):
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
    # validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)
    for epoch in range(model.n_steps):
        print("Epoch = %i" % epoch)
        batch = np.random.choice(len(encoded_matrices), batch_size, replace=False)
        batch_loss = None
        for idx in batch:
            matrix = encoded_matrices[idx]
            matrix.to(device)
            optimizer.zero_grad()
            pred = model(matrix[:, :, 0:1].float(), matrix.size(2))
            if batch_loss is not None:
                batch_loss += loss(pred, matrix.float())
            else:
                batch_loss = loss(pred, matrix.float())
        batch_loss.backward()
        optimizer.step()

    return model

encoded_matrices = []
for path in song_list:
    midi_data = pretty_midi.PrettyMIDI(data_path + path)
    for matrix in encoding_to_LSTM(midi_data):
        encoded_matrices.append(torch.from_numpy(matrix[np.newaxis, :, :]))

LSTMmodel = EtudeRNN(50)
LSTMmodel = train_LSTM(LSTMmodel, encoded_matrices)
torch.save(LSTMmodel.state_dict(), model_path)

