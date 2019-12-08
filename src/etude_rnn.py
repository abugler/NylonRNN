import torch.nn as nn
import torch

class EtudeRNN(torch.nn.Module):
    def __init__(self, input_dimensions, n_steps=400, n_hidden=128, n_layers=3, dropout=.5, learning_rate=.003):
        super().__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.input_dimensions = input_dimensions

        self.layers = []

        # Define LSTM
        self.lstm = nn.LSTM(input_size=self.input_dimensions, hidden_size=n_hidden,
                    num_layers=n_layers, batch_first=True, dropout=dropout)
        # Readout Layer
        self.fc = nn.Linear(n_hidden, self.input_dimensions)
        self.activation = torch.sigmoid

    def forward(self, x, timesteps=None):
        """
        Predicts the next timesteps - x.shape[0] notes
        :param x: A tensor of shape (batch_size, self.input_dimensions, timesteps)
        :param timesteps: Number of total timesteps that should be outputed
        :return out: A tensor of shape (batch_size, self.input_dimensions, timesteps).
        Can be coded to a PrettyMIDI object with the decoding_to_midi
        """

        # reorder x
        x = x.permute(2, 0, 1)

        if x.size(0) >= timesteps:
            raise ValueError("Timesteps must be greater than the x.size[0]. "
                             "(What would we predict then if timesteps <= x.size(0)?)")

        # init hidden state
        hn = torch.randn(self.n_layers, x.size(1), self.n_hidden)
        # init cell state
        cn = torch.randn(self.n_layers, x.size(1), self.n_hidden)

        # init out vector
        out = torch.empty(timesteps, x.size(1), self.input_dimensions)
        out[0:x.size(0), :, :] = x

        network_in = x[0:1, :, :]
        for step in range(timesteps - 1):
            lstm_output, (hn, cn) = self.lstm(network_in, (hn.detach(), cn.detach()))
            if step + 1 >= x.size(0):
                network_in = self.activation(self.fc(lstm_output.detach().reshape(self.n_hidden))).reshape(1, x.size(1), self.input_dimensions)
                out[step + 1, :, :] = network_in[0, :, :]
            else:
                network_in = x[step + 1:step + 2, :, :]
        return out.permute((1, 2, 0))