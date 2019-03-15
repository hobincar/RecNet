from __future__ import print_function

import torch

from utils import score, dict_to_cls, save_result
from config import EvalConfig as C
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
    train_scores, train_refs, train_hypos = score(model, train_iter, vocab)
    save_result(train_refs, train_hypos, C.result_dpath, config.corpus, 'train')
    print("[TRAIN] {}".format(train_scores))

    """ Validation Set """
    val_scores, val_refs, val_hypos = score(model, val_iter, vocab)
    save_result(val_refs, val_hypos, C.result_dpath, config.corpus, 'val')
    print("[VAL] scores: {}".format(val_scores))
    '''

    """ Test Set """
    test_scores, test_refs, test_hypos = score(model, test_iter, vocab)
    save_result(test_refs, test_hypos, C.result_dpath, config.corpus, 'test')
    print("[TEST] {}".format(test_scores))

if __name__ == "__main__":
    run(C.ckpt_fpath)

