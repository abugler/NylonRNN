from encoding import decoding_to_midi, encoding_to_LSTM
from etude_rnn import EtudeRNN
import numpy as np
import torch
import datetime
import pretty_midi

absolute_path = "C:\\Users\\Andreas\\Documents\\CS397Pardo\\Project\\EtudeRNN\\"
primer_path = "src\\primer.npy"
lstm_path = "src\\LSTM_model"
output_path = "midi_output\\"

model = EtudeRNN(50)
beats_to_generate = 64

try:
    primer_matrix = np.load(primer_path)
    model.load_state_dict(torch.load(lstm_path))
except FileNotFoundError:
    primer_path = absolute_path + primer_path
    lstm_path = absolute_path + lstm_path
    output_path = absolute_path + output_path
    primer_matrix = np.load(primer_path)
    model.load_state_dict(torch.load(lstm_path))

primer_matrix = primer_matrix.astype(float)
model.eval()
now = datetime.datetime.now()
np.random.seed(int(now.microsecond * now.second ** 2))
random_row = np.random.randint(0, primer_matrix.shape[1] / 24)
primer_np = primer_matrix[np.newaxis, :, random_row:random_row+24]
primer = torch.from_numpy(primer_np).float()

generated_matrix = np.empty((50, beats_to_generate * 24))
generated_matrix[:, 0:24] = primer_np[0, :, :]

# init hidden state
hn = torch.zeros(model.n_layers, 1, model.n_hidden)
# init cell state
cn = torch.zeros(model.n_layers, 1, model.n_hidden)

for i in range(1, (beats_to_generate -1) * 24):
    out, hn, cn = model(primer, hn, cn)
    generated_matrix[:, i + 23]= out[0, :, -1].detach().numpy()
    primer = torch.from_numpy(generated_matrix[np.newaxis, :, i:i+24]).float()

midi_data = decoding_to_midi(generated_matrix)
midi_data.write(output_path + str(now.date()) + '_' + str(np.random.randint(0, 1e8)) + ".mid")

