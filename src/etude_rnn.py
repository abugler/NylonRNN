import torch.nn as nn
import torch

class EtudeRNN(torch.nn.Module):
    def __init__(self, input_dimensions, n_steps=40, n_hidden=16, n_layers=2, dropout=.2, learning_rate=.003):
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

    def set_device(self, device):
        self.device = device
        if device == "cuda:0":
            self.lstm = self.lstm.cuda()
            self.fc = self.fc.cuda()
            print("The device is cuda")

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

        hn = hn.detach()
        cn = cn.detach()

        if self.device == 'cuda:0':
            hn = hn.cuda()
            cn = cn.cuda()
            x = x.cuda()

        lstm_output, (hn, cn) = self.lstm(x, (hn, cn))
        out = self.activation(self.fc(lstm_output))

        out = out.permute(1, 2, 0)
        return out, hn, cn

