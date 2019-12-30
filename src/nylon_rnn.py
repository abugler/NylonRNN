import torch.nn as nn
import torch

class NylonRNN(torch.nn.Module):
    """
    NylonRNN is the LSTM network used to generate songs to for classical guitar.
    Default architecture includes 60 input dimensions, 44 dimensions to represent a piano roll,
    6 dimensions to represent the attack matrix, and 10 for beat signaling, 64 hidden LSTM layers consisting of
    50 hidden units per layer, with a dropout of .2, and a fully connected output layer with sigmoid activation
    """
    def __init__(self, input_dimensions, output_dimensions, n_steps=30000, n_hidden=50, n_layers=64, dropout=.2):
        super().__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_steps = n_steps
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions

        self.device = None

        self.layers = []

        # Define LSTM
        self.lstm = nn.LSTM(input_size=self.input_dimensions, hidden_size=n_hidden,
                    num_layers=n_layers, batch_first=False, dropout=dropout)

        # Readout Layer
        self.fc = nn.Linear(n_hidden, self.output_dimensions)

        self.activation = torch.sigmoid

    # Switches network components to cuda
    def set_device(self, device):
        self.device = device
        if device != "cpu":
            self.lstm = self.lstm.cuda()
            self.fc = self.fc.cuda()
            print("The device is cuda")

    def forward(self, x, hn = None, cn = None):
        """
        Predicts the next timesteps - x.shape[0] notes
        :param x: A tensor of shape (batch_size, self.input_dimensions, timesteps)
        :param timesteps: Number of total timesteps that should be outputed
        :return out: A tensor of shape (batch_size, self.input_dimensions, timesteps).
        Can be coded to a PrettyMIDI object with the encoding.decoding_to_midi function
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

        # transfers hn, cn, and x to cuda, if device is cuda
        if self.device != 'cpu':
            hn = hn.cuda()
            cn = cn.cuda()
            x = x.cuda()

        lstm_output, (hn, cn) = self.lstm(x, (hn, cn))
        out = self.activation(self.fc(lstm_output))

        out = out.permute(1, 2, 0)
        return out, hn, cn
