import numpy as np
import torch.nn as nn
import torch
from encoding import encoding_to_LSTM, find_beat_matrix
import matplotlib.pyplot as plt
from nylon_rnn import NylonRNN
import os
import time

absolute_path = "C:\\Users\\Andreas\\Documents\\CS397Pardo\\Project\\NylonRNN\\"
model_path = "models\\"
npdata_filepath = "data\\classical_guitar_npdata\\"
state_dict = "models\\3800318"
timestep_path = "data\\timestep.npy"
try:
    list_songs = os.listdir(npdata_filepath)
except FileNotFoundError:
    npdata_filepath = absolute_path + npdata_filepath
    model_path = absolute_path + model_path
    list_songs = os.listdir(npdata_filepath)
    state_dict = absolute_path + state_dict
    timestep_path = absolute_path + timestep_path

list_songs = sorted(list_songs, key=lambda name: int(name[:name.find(".")]))
np.random.seed(int(time.time()))

def train_LSTM(model, features, targets, batch_size=20, regular_param=1e-8, learning_rate=.003,
               loss=nn.BCELoss(), regularization=nn.MSELoss(), model_name=str(np.random.randint(0, 1e7)),
               optimizer_algorithm=torch.optim.Adam):
    """
    Trains LSTM given the following parameters

    :param model: NylonRNN model to be trained
    :param training_dataset: A list of numpy array objects representing the midi files, segmented by tempo and
    time signature. All numpy arrays in this list should have the beat matrix already appended to it on axis 0.
    :param lr: Learning Rate for optimizer
    :param batch_size: Batch Size for data loader
    :param regular_param: Coefficient to multiply the output of the regularization by
    :param loss: Loss function to pass in. Loss function ideally be in the same format as a PyTorch loss function
    :param regularization: Loss function for regularization.  Ideally should be in the same format as a PyTorch loss function
    :param model_name: Filename for the trained model
    :param optimizer_algorithm: A torch.optim.* object.
    :return model: Trained model
    :return training_loss: Loss of the training set by epoch
    """
    training_loss = np.empty((model.n_steps))
    saved_loss = 1e20
    optimizer = optimizer_algorithm(model.parameters(), lr=learning_rate)
    for epoch in range(model.n_steps):
        if epoch % 10 == 0:
            begin = time.time()
        indices = np.random.choice(len(features), batch_size)
        model.zero_grad()
        optimizer.zero_grad()
        batch_loss = None
        for idx in indices:
            feature = features[idx]
            target = targets[idx]
            if model.device == "cuda:0":
                feature = feature.cuda()
                target = target.cuda()

            out, _, _ = model(feature)
            if batch_loss is None:
                batch_loss = loss(out, target)
            else:
                batch_loss += loss(out, target)

        for list in model.lstm.all_weights:
            for param in list:
                batch_loss += regular_param * regularization(param.data.float(), torch.zeros_like(param.data).float())

        batch_loss.backward()
        optimizer.step()
        training_loss[epoch] = batch_loss.item()
        if epoch % 10 == 0:
            print("Epoch = %i \nEpoch Duration: %f seconds \n Loss: %f" %
              (epoch, time.time() - begin, batch_loss.item()))
        if batch_loss.item() < saved_loss:
            print("Writing model")
            saved_loss = batch_loss.item()
            torch.save(model.state_dict(), model_path + model_name)
    return model, training_loss

training_features = []
training_targets = []
beginning_ts = np.load(timestep_path)
for i in range(len(list_songs)):
    next_array = np.load(npdata_filepath + list_songs[i])
    next_beat = find_beat_matrix(next_array, beginning_ts[i])
    feature = torch.from_numpy(np.append(next_array, next_beat, axis=0)[np.newaxis, :, :-1]).float()
    target = torch.from_numpy(next_array[np.newaxis, :, 1:]).float()
    training_features.append(feature)
    training_targets.append(target)


model = NylonRNN(60, 50, n_steps=10000)
# model.load_state_dict(torch.load(state_dict, map_location=model.device))
if torch.cuda.is_available():
    model.set_device('cuda:0')
else:
    model.set_device('cpu')

model, training_loss = train_LSTM(model, training_features, training_targets,
                                  batch_size=1, learning_rate=1e-3, regular_param=1e-15)

plt.plot(np.arange(0, model.n_steps), training_loss)
plt.title("BCELoss over epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

