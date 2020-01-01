import numpy as np
import torch.nn as nn
import torch
from encoding import encoding_to_LSTM, find_beat_matrix
import matplotlib.pyplot as plt
from nylon_rnn import NylonRNN
import os
import time
from train_rnn import train_LSTM
import pretty_midi

"""
This experiment tests the necessity of the following configurations:
1. Segmenting by Time Signature
2. Segmenting by Tempo
3. Continuous Beat Matrices

See here for more info: 
https://www.notion.so/andreasbugler/Intuitions-and-Experiments-with-Time-Signatures-and-Tempo-Changes-77baf23cf99b4fe7b2e70f5d0f9bbfc4
"""
song_path = "data\\Classical_Guitar_classicalguitarmidi.com_MIDIRip\\Aguado_Ocho_Pequenas_Piezas_Op3_No8_Minueto.mid"
configs = [
(True, True),
(True, False),
(False, True),
(False, False)
]
try:
    midi_data = pretty_midi.PrettyMIDI(song_path)
except FileNotFoundError:
    absolute_path = "C:\\Users\\Andreas\\Documents\\CS397Pardo\\Project\\NylonRNN\\"
    midi_data = pretty_midi.PrettyMIDI(absolute_path + song_path)

for ts_segmenting, contiguous_beat in configs:
    training_features = []
    training_targets = []
    for matrix, ts in encoding_to_LSTM(midi_data, time_segments=ts_segmenting):
        beat_matrix = find_beat_matrix(matrix, ts if contiguous_beat else 0)
        feature = torch.from_numpy(np.append(
            matrix, beat_matrix, axis=0)[np.newaxis, :, :-1]).float()
        target = torch.from_numpy(matrix[np.newaxis, : ,1:]).float()
        training_features.append(feature)
        training_targets.append(target)

    model = NylonRNN(60, 50, n_steps=1000)
    if torch.cuda.is_available():
        model.set_device('cuda:0')
    else:
        model.set_device('cpu')

    model, training_loss, model_name = train_LSTM(model, training_features, training_targets,
                                   batch_size=1, learning_rate=1e-3, regular_param=1e-15)
    plt.plot(np.arange(0, model.n_steps), training_loss)
    plt.title("Loss over epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(model_name + ".png")