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
try:
    list_songs = os.listdir(npdata_filepath)
except FileNotFoundError:
    npdata_filepath = absolute_path + npdata_filepath
    model_path = absolute_path + model_path
    list_songs = os.listdir(npdata_filepath)

np.random.seed(int(time.time()))

class MidiDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __getitem__(self, item):
        return (self.x[item], self.y[item])

    def __len__(self):
        return len(self.x)

def train_LSTM(model, encoded_matrices, batch_size=20, regular_param=1e-6):
    """
    Trains LSTM

    :param model: LSTM model to train
    :param training_dataset: training dataset
    :param lr: Learning Rate for optimizer
    :param batch_size: Batch Size for data loader
    :return:
    """
    loss = nn.BCEWithLogitsLoss()
    regularization = nn.MSELoss()
    # training_dataloader = DataLoader(midi_dataset, batch_size=batch_size)
    training_loss = np.empty((model.n_steps))
    saved_loss = 1e20
    model_name = str(np.random.randint(0, 1e7))
    for epoch in range(model.n_steps):
        optimizer = torch.optim.SGD(model.parameters(), lr=model.learning_rate * (epoch / model.n_steps), momentum=1e-3)
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

            # for i in range(matrix.size(1) - 1):
            #     out, hn, cn = model(matrix[:, :, i:i+1], hn, cn)
            #     if i == 0:
            #         batch_loss = loss(out, matrix[:, :, i+1:i+2])
            #     else:
            #         batch_loss += loss(out, matrix[:, :, i+1:i+2])

            out, _, _ = model(matrix)
            if batch_loss is None:
                batch_loss = loss(out[:, :, :-1], matrix[:, :, 1:])
            else:
                batch_loss += loss(out[:, :, :-1], matrix[:, :, 1:])

            for list in model.lstm.all_weights:
                for param in list:
                    batch_loss += regular_param * regularization(param.data.float(), torch.zeros_like(param.data).float())

        batch_loss.backward()
        optimizer.step()
        print("Epoch = %i \nEpoch Duration: %f seconds \n Loss: %i" %
              (epoch, time.time() - begin, batch_loss.item()))
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


# sample_beats = 16
# sample_time_steps = 24 * sample_beats
# long_x = []
# long_y = []
# for matrix in encoded_matrices:
#     for i in range(0, matrix.size(2) - sample_time_steps - 1):
#         long_x.append(matrix[0, :, i:i+sample_time_steps])
#         long_y.append(matrix[0, :, i+1: i+sample_time_steps+1])
# long_midi_dataset = MidiDataset(long_x, long_y)

# sample_beats = 1
# sample_time_steps = 24 * sample_beats
# small_x = []
# small_y = []
# for matrix in encoded_matrices:
#     for i in range(0, matrix.size(2) - sample_time_steps - 1):
#         small_x.append(matrix[0, :, i:i+sample_time_steps])
#         small_y.append(matrix[0, :, i+1: i+sample_time_steps+1])
# small_midi_dataset = MidiDataset(small_x, small_y)
# small_midi_dataset.x = small_midi_dataset.x[:20]
# small_midi_dataset.y = small_midi_dataset.y[:20]

model = NylonRNN(50, n_steps=50000, learning_rate=5e-2)
if torch.cuda.is_available():
    model.set_device('cuda:0')

# model = train_LSTM(model, long_midi_dataset, "coarse")
model, training_loss = train_LSTM(model, encoded_matrices, batch_size=1)
plt.plot(np.arange(0, model.n_steps), training_loss)
plt.title("BCELoss over epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
torch.save(model.state_dict(), model_path + "_final")

