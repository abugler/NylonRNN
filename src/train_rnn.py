import sys
import os
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pretty_midi
# import filter_songs

"""
Resources: 
https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
"""

# if len(sys.argv) < 3:
#     sys.exit("Usage: train_rnn.py <configuration_name> <train data filename>")

# config_name = "configuration.py"
# data_path = "data\\lmd_matched"
# config = importlib.import_module('configurations.%s' % config_name)

### Building the model

# learning_rate = np.float32(config.learning_rate)

## Let us define our RNN class

class EtudeRNN(torch.nn.Module):
    def __init__(self, input_dimensions, n_steps=100, n_hidden=128, n_layers=3, dropout=.5, learning_rate=.003):
        super().__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.input_dimensions = input_dimensions

        self.layers = []

        # Define LSTM
        self.lstm = nn.LSTM(input_size=self.input_dimensions, hidden_size=n_hidden,
                    num_layers=n_layers, batch_first=True, dropout=dropout)
        # Readout Layer
        self.fc = nn.Linear(n_hidden, self.input_dimensions)
        self.activation = F.sigmoid


    def forward(self, x, timesteps):
        """
        Predicts the next timesteps - x.shape[0] notes
        :param x: A tensor of shape (sequence, batch_size, self.input_dimensions)
        :param timesteps: Number of total timesteps that should be outputed
        :return out: A tensor of shape (timesteps, batch_size, self.input_dimensions).
        Can be coded to a PrettyMIDI object with the decoding_to_midi
        """

        if x.size[0] >= timesteps:
            raise ValueError("Timesteps must be greater than the x.size[0]. "
                             "(What would we predict then if timesteps <= x.size(0)?)")

        # init hidden state
        hn = torch.randn(self.n_layers, self.n_hidden)
        # init cell state
        cn = torch.randn(self.n_layers, self.n_hidden)

        # init out vector
        out = torch.empty(timesteps, x.size[1], self.input_dimensions)
        out[0:x.size(0), :, :] = x

        network_in = x[0, :, :]
        for step in range(timesteps - 1):
            lstm_output, (hn, cn) = self.lstm(network_in, (hn.detach(), cn.detach()))
            if step + 1 >= x.size[0]:
                network_in = self.activation(self.fc(lstm_output))
                out[step + 1, :, :] = network_in
            else:
                network_in = x[step + 1, :, :]
        return out

def train_LSTM(model, training_dataset, validation_dataset, n_epochs=1000, lr=1e-4, batch_size=50):
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    for epoch in range(n_epochs):
        for features, targets in training_dataloader:
            optimizer.zero_grad()
            # init hidden state
            hn = torch.randn(model.n_layers, model.n_hidden)
            # init cell state
            cn = torch.randn(model.n_layers, model.n_hidden)
            pred = model(features)
            single_loss = loss(pred, targets)
            single_loss.backward()
            optimizer.step()



# E2 is the lowest note on a Standard classical Guitar
E2 = 40
# B5 is the highest fretted note on a standard classical guitar
B5 = 83

def encoding_to_LSTM(midi_data: pretty_midi.PrettyMIDI):
    """
    The encoding for this data is specific to solo classical guitar pieces with no pinch harmonics nor percussive elements.

    The encoding for LSTM will be a np.ndarray of 50 rows and d columns, where there are d time steps.
    The first 44 rows will be for marking whether or not the corresponding pitch will be played. Row 0 will correspond with E2,
    the lowest note on classical guitar, row 1 will correspond with F2, row 2 will correspond with F#2, and so on,
    until row 43, which corresponds with B5, and is the highest non-harmonic note on classical guitar.

    The last 6 rows correspond with whether or not a specific note if plucked or held from the previous timestep.
    For example, if a 1 exists in row 44, then the lowest note found in above 44 rows is to be plucked in this timestep.
    If it is 0, then the lowest note found in the above 44 rows is held from the previous timestep. This is the same for rows
    45-49, where each row corresponds with the 2nd-6th lowest note, respectively. The rationale for this part of the encoding is
    to differentiate between many of the same note being played at the same time and a not being held.

    Each timestep is 1/24 of a beat.  This is to account for both notes that last 1/8 of a beat, and notes that last 1/3 of a beat.
    As most songs' shortest notes are roughly either 1/3 or 1/8 of a beat, this will account for both.

    Midi_data will be segmented by tempo. Sections less than 20 beats of constant tempo will be ignored.

    :param midi_data: A pretty_midi.PrettyMidi object to be encoded
    :param tempo: Tempo of pretty_midi object.  Default is 100
    :return: encoded_matrices: A list of encoded matrices
    """
    beats_min = 20
    tempo_change_times, tempi = midi_data.get_tempo_changes()
    if tempo_change_times is None:
        tempo = midi_data.estimate_tempo()
        range_vectors = [np.arange(0, midi_data.get_end_time(), 1 / (24 * tempo))]
        range_tempi = [tempo]
    else:
        range_vectors = []
        range_tempi = []
        for i in range(len(tempi)):
            start_time = tempo_change_times[i]
            end_time = tempo_change_times[i + 1] if i < len(tempi) - 1 else midi_data.get_end_time()
            vector = np.arange(start_time, end_time, 1 / (tempi[i] / 60 * 24))
            if not range_vectors:
                vector[0] -= 1e5
            if vector.shape[0] > beats_min * 24:
                range_vectors.append(vector)
                range_tempi.append(tempi[i])

    # This will only work with midi data with a single instrument
    def find_pluck_matrix(midi_data: pretty_midi.PrettyMIDI, vector: np.ndarray, tempo: int):
        pluck_matrix = np.zeros((6, vector.shape[0]))
        section_notes = lambda _note:_note.start >= vector[0] and _note.end <= vector[-1] + tempo / 60 / 24
        notes = sorted(filter(section_notes, midi_data.instruments[0].notes), key=lambda x: x.pitch)
        section_start = vector[0]
        for note in notes:
            timestep = int((note.start - section_start) / 60 * tempo * 24)
            simultaneous_notes = 0
            while pluck_matrix[simultaneous_notes, timestep] == 1:
                simultaneous_notes += 1
            pluck_matrix[simultaneous_notes, timestep] = 1
        return pluck_matrix

    encoded_matrices = []
    instrument = midi_data.instruments[0]
    for vector, tempo in zip(range_vectors, range_tempi):
        midi_matrix = instrument.get_piano_roll(times=vector)[E2:B5 + 1, :]
        # Right now, midi_matrix is a matrix of velocities.
        # Let's change this so midi matrix is a matrix of whether the note is played or not
        one_hot = np.vectorize(lambda x: np.int(x != 0))
        midi_matrix = one_hot(midi_matrix)
        pluck_matrix = find_pluck_matrix(midi_data, vector, tempo)
        encoded_matrices.append(
            np.append(midi_matrix, pluck_matrix, axis=0)
        )

    return encoded_matrices

def decoding_to_midi(encoded_matrix, tempo=100, time_signature="4/4"):
    """
    Decodes a matrix encoded for LSTM back to a PrettyMIDI file
    :param encoded_matrix:
    :param tempo:
    :return:
    """
    raise NotImplementedError("bitch")





