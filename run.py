from __future__ import print_function
import os

import torch

from utils import dict_to_cls, get_predicted_captions, get_groundtruth_captions, save_result, score
from configs.run import RunConfig as C
from loader.MSVD import MSVD
from loader.MSRVTT import MSRVTT
from models.decoder import Decoder
from models.global_reconstructor import GlobalReconstructor
from models.local_reconstructor import LocalReconstructor
from models.caption_generator import CaptionGenerator


def run(ckpt_fpath):
    checkpoint = torch.load(ckpt_fpath)

    """ Load Config """
    config = dict_to_cls(checkpoint['config'])

    """ Build Data Loader """
    if config.corpus == "MSVD":
        corpus = MSVD(config)
    elif config.corpus == "MSR-VTT":
        corpus = MSRVTT(config)
    train_iter, val_iter, test_iter, vocab = \
        corpus.train_data_loader, corpus.val_data_loader, corpus.test_data_loader, corpus.vocab
    print('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.'.format(
        vocab.n_vocabs, vocab.n_vocabs_untrimmed, vocab.n_words, vocab.n_words_untrimmed, config.loader.min_count))

    """ Build Models """
    decoder = Decoder(
        rnn_type=config.decoder.rnn_type,
        num_layers=config.decoder.rnn_num_layers,
        num_directions=config.decoder.rnn_num_directions,
        feat_size=config.feat.size,
        feat_len=config.loader.frame_sample_len,
        embedding_size=config.vocab.embedding_size,
        hidden_size=config.decoder.rnn_hidden_size,
        attn_size=config.decoder.rnn_attn_size,
        output_size=vocab.n_vocabs,
        rnn_dropout=config.decoder.rnn_dropout)
    decoder.load_state_dict(checkpoint['decoder'])

    if config.reconstructor is None:
        reconstructor = None
    else:
        if config.reconstructor.type == 'global':
            reconstructor = GlobalReconstructor(
                rnn_type=config.reconstructor.rnn_type,
                num_layers=config.reconstructor.rnn_num_layers,
                num_directions=config.reconstructor.rnn_num_directions,
                decoder_size=config.decoder.rnn_hidden_size,
                hidden_size=config.reconstructor.rnn_hidden_size,
                rnn_dropout=config.reconstructor.rnn_dropout)
        else:
            reconstructor = LocalReconstructor(
                rnn_type=config.reconstructor.rnn_type,
                num_layers=config.reconstructor.rnn_num_layers,
                num_directions=config.reconstructor.rnn_num_directions,
                decoder_size=config.decoder.rnn_hidden_size,
                hidden_size=config.reconstructor.rnn_hidden_size,
                attn_size=config.reconstructor.rnn_attn_size,
                rnn_dropout=config.reconstructor.rnn_dropout)
        reconstructor.load_state_dict(checkpoint['reconstructor'])

    model = CaptionGenerator(decoder, reconstructor, config.loader.max_caption_len, vocab)
    model = model.cuda()

    '''
    """ Train Set """
    train_vid2pred = get_predicted_captions(train_iter, model, model.vocab, beam_width=5, beam_alpha=0.)
    train_vid2GTs = get_groundtruth_captions(train_iter, model.vocab)
    train_scores = score(train_vid2pred, train_vid2GTs)
    print("[TRAIN] {}".format(train_scores))

    """ Validation Set """
    val_vid2pred = get_predicted_captions(val_iter, model, model.vocab, beam_width=5, beam_alpha=0.)
    val_vid2GTs = get_groundtruth_captions(val_iter, model.vocab)
    val_scores = score(val_vid2pred, val_vid2GTs)
    print("[VAL] scores: {}".format(val_scores))
    '''

    """ Test Set """
    test_vid2pred = get_predicted_captions(test_iter, model, model.vocab, beam_width=5, beam_alpha=0.)
    test_vid2GTs = get_groundtruth_captions(test_iter, model.vocab)
    test_scores = score(test_vid2pred, test_vid2GTs)
    print("[TEST] {}".format(test_scores))

    test_save_fpath = os.path.join(C.result_dpath, "{}_{}.csv".format(config.corpus, 'test'))
    save_result(test_vid2pred, test_vid2GTs, test_save_fpath)

if __name__ == "__main__":
    run(C.ckpt_fpath)

