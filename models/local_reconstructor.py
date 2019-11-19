import torch
import torch.nn as nn
import torch.nn.functional as F

from models.temporal_attention import TemporalAttention

class LocalReconstructor(nn.Module):
    def __init__(self, rnn_type, num_layers, num_directions, decoder_size, hidden_size, attn_size, rnn_dropout):
        super(LocalReconstructor, self).__init__()
        self._type = 'local'
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.decoder_size = decoder_size
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.rnn_dropout_p = rnn_dropout

        RNN = nn.LSTM if self.rnn_type == 'LSTM' else nn.GRU
        self.rnn = RNN(
            input_size=self.decoder_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.rnn_dropout_p,
            bidirectional=True if self.num_directions == 2 else False)

        self.attention = TemporalAttention(
            hidden_size=self.num_layers * self.num_directions * self.hidden_size,
            feat_size=self.decoder_size,
            bottleneck_size=self.attn_size)

    def get_last_hidden(self, hidden):
        last_hidden = hidden[0] if isinstance(hidden, tuple) else hidden
        last_hidden = last_hidden.view(self.num_layers, self.num_directions, last_hidden.size(1), last_hidden.size(2))
        last_hidden = last_hidden.transpose(2, 1).contiguous()
        last_hidden = last_hidden.view(self.num_layers, last_hidden.size(1), self.num_directions * last_hidden.size(3))
        last_hidden = last_hidden[-1]
        return last_hidden

    def forward(self, decoder_hiddens, hidden, caption_masks):
        last_hidden = self.get_last_hidden(hidden)
        attention_masks = caption_masks.transpose(0, 1)
        decoder_hidden, attn_weights = self.attention(last_hidden, decoder_hiddens, attention_masks)

        decoder_hidden = decoder_hidden.unsqueeze(0)
        output, hidden = self.rnn(decoder_hidden, hidden)
        return output, hidden

