import torch
import torch.nn as nn
import torch.nn.functional as F

from models.temporal_attention import TemporalAttention


class Decoder(nn.Module):
    def __init__(self, rnn_type, num_layers, num_directions, feat_size, feat_len, embedding_size,
                 hidden_size, attn_size, output_size, rnn_dropout):
        super(Decoder, self).__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.feat_size = feat_size
        self.feat_len = feat_len
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.output_size = output_size
        self.rnn_dropout_p = rnn_dropout

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)

        self.attention = TemporalAttention(
            hidden_size=self.num_layers * self.num_directions * self.hidden_size,
            feat_size=self.feat_size,
            bottleneck_size=self.attn_size)

        RNN = nn.LSTM if self.rnn_type == 'LSTM' else nn.GRU
        self.rnn = RNN(
            input_size=self.embedding_size + self.feat_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.rnn_dropout_p,
            bidirectional=True if self.num_directions == 2 else False)

        self.out = nn.Linear(self.num_layers * self.num_directions * self.hidden_size, self.output_size)

    def forward(self, input, hidden, feats):
        embedded = self.embedding(input)

        feats, attn_weights = self.attention(hidden, feats)

        input_combined = torch.cat((
            embedded,
            feats.unsqueeze(0)), dim=2)
        _, hidden = self.rnn(input_combined, hidden)

        output = hidden[0] if self.rnn_type == 'LSTM' else hidden
        output = torch.cat([ o for o in output ], dim=1)
        output = self.out(output)
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights

