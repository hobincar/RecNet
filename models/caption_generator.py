import random

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class CaptionGenerator(nn.Module):
    def __init__(self, decoder, reconstructor, max_caption_len, vocab):
        super(CaptionGenerator, self).__init__()
        self.decoder = decoder
        self.reconstructor = reconstructor
        self.max_caption_len = max_caption_len
        self.vocab = vocab

    def get_rnn_init_hidden(self, batch_size, num_layers, num_directions, hidden_size):
        if self.decoder.rnn_type == 'LSTM':
            hidden = (
                torch.zeros(num_layers * num_directions, batch_size, hidden_size).cuda(),
                torch.zeros(num_layers * num_directions, batch_size, hidden_size).cuda())
        else:
            hidden = torch.zeros(num_layers * num_directions, batch_size, hidden_size)
            hidden = hidden.cuda()
        return hidden

    def forward_decoder(self, batch_size, vocab_size, hidden, feats, captions, teacher_forcing_ratio):
        outputs = Variable(torch.zeros(self.max_caption_len + 2, batch_size, vocab_size)).cuda()
        hiddens = Variable(
            torch.zeros_like(hidden[0] if self.decoder.rnn_type == 'LSTM' else hidden)
                .unsqueeze(0)
                .repeat(self.max_caption_len + 2, 1, 1, 1))

        output = Variable(torch.cuda.LongTensor(1, batch_size).fill_(self.vocab.word2idx['<SOS>']))
        for t in range(1, self.max_caption_len + 2):
            output, hidden, attn_weights = self.decoder(output.view(1, -1), hidden, feats)
            outputs[t] = output
            hiddens[t] = hidden[0] if self.decoder.rnn_type == 'LSTM' else hidden
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(captions.data[t] if is_teacher else top1).cuda()
        return outputs, hiddens

    @property
    def forward_reconstructor(self):
        if self.reconstructor._type == 'global':
            return self.forward_global_reconstructor
        else:
            return self.forward_local_reconstructor

    def forward_global_reconstructor(self, batch_size, decoder_hiddens, caption_masks):
        def build_hiddens(_decoder_hiddens, _caption_masks):
            caption_lens = _caption_masks.sum(dim=0).type(torch.cuda.FloatTensor)
            _decoder_hiddens = _decoder_hiddens.transpose(1, 2)
            _decoder_hiddens = _decoder_hiddens.view(
                _decoder_hiddens.size(0),
                _decoder_hiddens.size(1),
                _decoder_hiddens.size(2) * _decoder_hiddens.size(3)) # T, B, H
            _caption_masks = _caption_masks.unsqueeze(2).expand_as(_decoder_hiddens).type_as(_decoder_hiddens)
            _decoder_hiddens = _caption_masks * _decoder_hiddens
            _decoder_hiddens_mean_pooled = _decoder_hiddens.sum(dim=0) / \
                caption_lens.unsqueeze(1).expand(caption_lens.size(0), _decoder_hiddens.size(2)) 
            return _decoder_hiddens, _decoder_hiddens_mean_pooled

        decoder_hiddens, decoder_hiddens_mean_pooled = build_hiddens(decoder_hiddens, caption_masks)

        feats_recons = Variable(torch.zeros(self.max_caption_len + 2, batch_size, self.reconstructor.hidden_size))
        feats_recons = feats_recons.cuda()
        hidden = self.get_rnn_init_hidden(batch_size, self.reconstructor.num_layers, self.reconstructor.num_directions,
                                          self.reconstructor.hidden_size)
        for t in range(self.max_caption_len + 2):
            decoder_hidden = decoder_hiddens[t]
            _, hidden = self.reconstructor(decoder_hidden, decoder_hiddens_mean_pooled, hidden)
            feats_recons[t] = hidden[0] if self.decoder.rnn_type == 'LSTM' else hidden
        feats_recons = feats_recons.transpose(0, 1)
        return feats_recons

    def forward_local_reconstructor(self, batch_size, decoder_hiddens, caption_masks):
        def build_hiddens(_decoder_hiddens, _caption_masks):
            _decoder_hiddens = _decoder_hiddens.permute(2, 0, 1, 3)
            _decoder_hiddens = _decoder_hiddens.view(
                _decoder_hiddens.size(0),
                _decoder_hiddens.size(1),
                _decoder_hiddens.size(2) * _decoder_hiddens.size(3)) # B, T, H
            _caption_masks = _caption_masks.transpose(0, 1).unsqueeze(2).expand_as(_decoder_hiddens).type_as(_decoder_hiddens)
            _decoder_hiddens = _caption_masks * _decoder_hiddens
            return _decoder_hiddens

        decoder_hiddens = build_hiddens(decoder_hiddens, caption_masks)

        feats_recons = Variable(torch.zeros(self.decoder.feat_len, batch_size, self.reconstructor.hidden_size))
        feats_recons = feats_recons.cuda()
        hidden = self.get_rnn_init_hidden(batch_size, self.reconstructor.num_layers, self.reconstructor.num_directions,
                                          self.reconstructor.hidden_size)
        for t in range(self.decoder.feat_len):
            _, hidden = self.reconstructor(decoder_hiddens, hidden)
            feats_recons[t] = hidden[0] if self.decoder.rnn_type == 'LSTM' else hidden
        feats_recons = feats_recons.transpose(0, 1)
        return feats_recons

    def forward(self, feats, captions=None, teacher_forcing_ratio=0.):
        batch_size = feats.size(0)
        vocab_size = self.decoder.output_size
        caption_masks = captions != self.vocab.word2idx['<PAD>']
        caption_masks = caption_masks.cuda()

        hidden = self.get_rnn_init_hidden(batch_size, self.decoder.num_layers, self.decoder.num_directions,
                                          self.decoder.hidden_size)
        outputs, decoder_hiddens = self.forward_decoder(batch_size, vocab_size, hidden, feats, captions,
                                                        teacher_forcing_ratio)
        feats_recon = self.forward_reconstructor(batch_size, decoder_hiddens, caption_masks)
        return outputs, feats_recon

    def describe(self, feats, beam_width, beam_alpha):
        batch_size = feats.size(0)
        vocab_size = self.decoder.output_size

        captions = self.beam_search(batch_size, vocab_size, feats, beam_width, beam_alpha)
        return captions

    def beam_search(self, batch_size, vocab_size,  feats, width, alpha):
        hidden = self.get_rnn_init_hidden(batch_size, self.decoder.num_layers, self.decoder.num_directions,
                                          self.decoder.hidden_size)

        input_list = [ torch.cuda.LongTensor(1, batch_size).fill_(self.vocab.word2idx['<SOS>']) ]
        hidden_list = [ hidden ]
        cum_prob_list = [ torch.ones(batch_size).cuda() ]
        cum_prob_list = [ torch.log(cum_prob) for cum_prob in cum_prob_list ]
        EOS_idx = self.vocab.word2idx['<EOS>']

        output_list = [ [[]] for _ in range(batch_size) ]
        for t in range(self.max_caption_len + 1):
            beam_output_list = [] # width x ( 1, 100 )
            normalized_beam_output_list = [] # width x ( 1, 100 )
            if self.decoder.rnn_type == "LSTM":
                beam_hidden_list = ( [], [] ) # 2 * width x ( 1, 100, 512 )
            else:
                beam_hidden_list = [] # width x ( 1, 100, 512 )
            next_output_list = [ [] for _ in range(batch_size) ]

            assert len(input_list) == len(hidden_list) == len(cum_prob_list)
            for i, (input, hidden, cum_prob) in enumerate(zip(input_list, hidden_list, cum_prob_list)):
                output, next_hidden, _ = self.decoder(input, hidden, feats)

                caption_list = [ output_list[b][i] for b in range(batch_size)]
                EOS_mask = [ 0. if EOS_idx in [ idx.item() for idx in caption ] else 1. for caption in caption_list ]
                EOS_mask = torch.cuda.FloatTensor(EOS_mask)
                EOS_mask = EOS_mask.unsqueeze(1).expand_as(output)
                output = EOS_mask * output

                output += cum_prob.unsqueeze(1)
                beam_output_list.append(output)

                caption_lens = [ [ idx.item() for idx in caption ].index(EOS_idx) + 1 if EOS_idx in [ idx.item() for idx in caption ] else t + 1 for caption in caption_list ]
                caption_lens = torch.cuda.FloatTensor(caption_lens)
                normalizing_factor = ((5 + caption_lens) ** alpha) / (6 ** alpha)
                normalizing_factor = normalizing_factor.unsqueeze(1).expand_as(output)
                normalized_output = output / normalizing_factor
                normalized_beam_output_list.append(normalized_output)
                if self.decoder.rnn_type == "LSTM":
                    beam_hidden_list[0].append(next_hidden[0])
                    beam_hidden_list[1].append(next_hidden[1])
                else:
                    beam_hidden_list.append(next_hidden)
            beam_output_list = torch.cat(beam_output_list, dim=1) # ( 100, n_vocabs * width )
            normalized_beam_output_list = torch.cat(normalized_beam_output_list, dim=1)
            beam_topk_output_index_list = normalized_beam_output_list.argsort(dim=1, descending=True)[:, :width] # ( 100, width )
            topk_beam_index = beam_topk_output_index_list // vocab_size # ( 100, width )
            topk_output_index = beam_topk_output_index_list % vocab_size # ( 100, width )

            topk_output_list = [ topk_output_index[:, i] for i in range(width) ] # width * ( 100, )
            if self.decoder.rnn_type == "LSTM":
                topk_hidden_list = (
                    [ [] for _ in range(width) ],
                    [ [] for _ in range(width) ]) # 2 * width * (1, 100, 512)
            else:
                topk_hidden_list = [ [] for _ in range(width) ] # width * ( 1, 100, 512 )
            topk_cum_prob_list = [ [] for _ in range(width) ] # width * ( 100, )
            for i, (beam_index, output_index) in enumerate(zip(topk_beam_index, topk_output_index)):
                for k, (bi, oi) in enumerate(zip(beam_index, output_index)):
                    if self.decoder.rnn_type == "LSTM":
                        topk_hidden_list[0][k].append(beam_hidden_list[0][bi][:, i, :])
                        topk_hidden_list[1][k].append(beam_hidden_list[1][bi][:, i, :])
                    else:
                        topk_hidden_list[k].append(beam_hidden_list[bi][:, i, :])
                    topk_cum_prob_list[k].append(beam_output_list[i][vocab_size * bi + oi])
                    next_output_list[i].append(output_list[i][bi] + [ oi ])
            output_list = next_output_list
    
            input_list = [ topk_output.unsqueeze(0) for topk_output in topk_output_list ] # width * ( 1, 100 )
            if self.decoder.rnn_type == "LSTM":
                hidden_list = (
                    [ torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list[0] ],
                    [ torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list[1] ]) # 2 * width * ( 1, 100, 512 )
                hidden_list = [ ( hidden, context ) for hidden, context in zip(*hidden_list) ]
            else:
                hidden_list = [ torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list ] # width * ( 1, 100, 512 )
            cum_prob_list = [ torch.cuda.FloatTensor(topk_cum_prob) for topk_cum_prob in topk_cum_prob_list ] # width * ( 100, )

        SOS_idx = self.vocab.word2idx['<SOS>']
        outputs = [ [ SOS_idx ] + o[0] for o in output_list ]
        return outputs

