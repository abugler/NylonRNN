from src.encoding import decoding_to_midi, encoding_to_LSTM
from etude_rnn import EtudeRNN
import numpy as np
import torch
import datetime

absolute_path = "C:\\Users\\Andreas\\Documents\\CS397Pardo\\Project\\EtudeRNN\\"
primer_path = "src\\primer.npy"
lstm_path = "src\\LSTM_model"

model = EtudeRNN(50)

try:
    primer_matrix = np.load(primer_path)
    model.load_state_dict(torch.load(lstm_path))
except FileNotFoundError:
    primer_path = absolute_path + primer_path
    lstm_path = absolute_path + lstm_path
    primer_matrix = np.load(primer_path)
    model.load_state_dict(torch.load(lstm_path))

model.eval()

now = datetime.datetime.now()
np.random.seed(int(now.microsecond * now.second ** 2))
random_row = np.random.randint(0, primer_matrix.shape[1])
primer = [np.newaxis, primer_matrix[:, random_row:random_row + 1]]

