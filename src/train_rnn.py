import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import pretty_midi
from encoding import encoding_to_LSTM
import matplotlib.pyplot as plt
from nylon_rnn import NylonRNN
import os
import time

absolute_path = "C:\\Users\\Andreas\\Documents\\CS397Pardo\\Project\\NylonRNN\\"
model_path = "models\\"
npdata_filepath = "data\\classical_guitar_npdata\\"
state_dict = "models\\3800318"
try:
    list_songs = os.listdir(npdata_filepath)
except FileNotFoundError:
    npdata_filepath = absolute_path + npdata_filepath
    model_path = absolute_path + model_path
    list_songs = os.listdir(npdata_filepath)
    state_dict = absolute_path + state_dict

np.random.seed(int(time.time()))

class MidiDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __getitem__(self, item):
        return (self.x[item], self.y[item])

    def __len__(self):
        return len(self.x)

def train_LSTM(model, encoded_matrices, batch_size=20, regular_param=1e-7):
    """
    Trains LSTM

    :param model: LSTM model to train
    :param training_dataset: training dataset
    :param lr: Learning Rate for optimizer
    :param batch_size: Batch Size for data loader
    :return:
    """
    loss = nn.BCELoss()
    regularization = nn.MSELoss()
    training_loss = np.empty((model.n_steps))
    saved_loss = 1e20
    model_name = str(np.random.randint(0, 1e7))
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    for epoch in range(model.n_steps):
        begin = time.time()
        indices = np.random.choice(len(encoded_matrices), batch_size)
        model.zero_grad()
        optimizer.zero_grad()
        batch_loss = None
        for idx in indices:
            matrix = encoded_matrices[idx]
            if model.device == "cuda:0":
                matrix = matrix.cuda()

            # init hidden state
            hn = torch.zeros(model.n_layers, matrix.size(0), model.n_hidden)
            # init cell state
            cn = torch.zeros(model.n_layers, matrix.size(0), model.n_hidden)

            for i in range(matrix.size(1) - 1):
                out, hn, cn = model(matrix[:, :, i:i+1], hn, cn)
                if i == 0:
                    batch_loss = loss(out, matrix[:, :, i+1:i+2])
                else:
                    batch_loss += loss(out, matrix[:, :, i+1:i+2])

            # out, _, _ = model(matrix)
            # if batch_loss is None:
            #     batch_loss = loss(out[:, :, :-1], matrix[:, :, 1:])
            # else:
            #     batch_loss += loss(out[:, :, :-1], matrix[:, :, 1:])

            for list in model.lstm.all_weights:
                for param in list:
                    batch_loss += regular_param * regularization(param.data.float(), torch.zeros_like(param.data).float())

        if epoch % 10 == 0:
            print("Epoch = %i \nEpoch Duration: %f seconds \n Loss: %f" %
              (epoch, time.time() - begin, batch_loss.item()))
        batch_loss.backward()
        optimizer.step()
        training_loss[epoch] = batch_loss.item()
        if epoch % 1000 == 0:
            if batch_loss.item() < saved_loss:
                print("Writing model")
                torch.save(model.state_dict(), model_path +model_name)
    return model, training_loss

encoded_matrices = []
for path in list_songs:
    encoded_matrices.append(torch.from_numpy(np.load(npdata_filepath + path)[np.newaxis, :, :]).float())
    break

model = NylonRNN(50, n_steps=50000, learning_rate=1e-4)
# model.load_state_dict(torch.load(state_dict, map_location=model.device))
if torch.cuda.is_available():
    model.set_device('cuda:0')

model, training_loss = train_LSTM(model, encoded_matrices, batch_size=1)

plt.plot(np.arange(0, model.n_steps), training_loss)
plt.title("BCELoss over epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

