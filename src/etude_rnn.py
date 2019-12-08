import torch.nn as nn
import torch

class EtudeRNN(torch.nn.Module):
    def __init__(self, input_dimensions, n_steps=100, n_hidden=128, n_layers=3, dropout=.5, learning_rate=.003):
        super().__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.input_dimensions = input_dimensions

        self.device = None

        self.layers = []

        # Define LSTM
        self.lstm = nn.LSTM(input_size=self.input_dimensions, hidden_size=n_hidden,
                    num_layers=n_layers, batch_first=False, dropout=dropout)
        # Readout Layer
        self.fc = nn.Linear(n_hidden, self.input_dimensions)
        self.activation = torch.sigmoid

    def forward(self, x, hn = None, cn = None):
        """
        Predicts the next timesteps - x.shape[0] notes
        :param x: A tensor of shape (batch_size, self.input_dimensions, timesteps)
        :param timesteps: Number of total timesteps that should be outputed
        :return out: A tensor of shape (batch_size, self.input_dimensions, timesteps).
        Can be coded to a PrettyMIDI object with the decoding_to_midi
        """

        # reorder x

        x = x.permute(2, 0, 1)
        if hn is None or cn is None:
            # init hidden state
            hn = torch.zeros(self.n_layers, x.size(1), self.n_hidden)
            # init cell state
            cn = torch.zeros(self.n_layers, x.size(1), self.n_hidden)

        hn.to(device)
        cn.to(device)

        lstm_output, (hn, cn) = self.lstm(x, (hn.detach(), cn.detach()))
        out = self.activation(self.fc(lstm_output))

        out = out.permute(1, 2, 0)
        return out, hn, cn

