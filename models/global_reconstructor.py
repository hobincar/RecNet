import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalReconstructor(nn.Module):
    def __init__(self, rnn_type, num_layers, num_directions, decoder_size, hidden_size, rnn_dropout):
        super(GlobalReconstructor, self).__init__()
        self._type = 'global'
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.decoder_size = decoder_size
        self.hidden_size = hidden_size
        self.rnn_dropout_p = rnn_dropout

        RNN = nn.LSTM if self.rnn_type == 'LSTM' else nn.GRU
        self.rnn = RNN(
            input_size=self.decoder_size * 2,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.rnn_dropout_p,
            bidirectional=True if self.num_directions == 2 else False)

    def forward(self, decoder_hidden, decoder_hiddens_mean_pooled, hidden):
        input_combined = torch.cat([
            decoder_hidden,
            decoder_hiddens_mean_pooled ], dim=1)
        input_combined = input_combined.unsqueeze(0)

        output, hidden = self.rnn(input_combined, hidden)
        return output, hidden

