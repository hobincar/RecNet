from __future__ import print_function

from tensorboardX import SummaryWriter
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import train, evaluate, score, get_lr, save_checkpoint, load_checkpoint
from config import TrainConfig as C
from loader.MSVD import MSVD
from loader.MSRVTT import MSRVTT
from models.decoder import Decoder
from models.global_reconstructor import GlobalReconstructor
from models.local_reconstructor import LocalReconstructor
from models.caption_generator import CaptionGenerator


def build_loaders():
    if C.corpus == "MSVD":
        corpus = MSVD(C)
    elif C.corpus == "MSR-VTT":
        corpus = MSRVTT(C)
    print('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.'.format(
        corpus.vocab.n_vocabs, corpus.vocab.n_vocabs_untrimmed, corpus.vocab.n_words,
        corpus.vocab.n_words_untrimmed, C.loader.min_count))
    return corpus.train_data_loader, corpus.val_data_loader, corpus.test_data_loader, corpus.vocab


def build_model(vocab):
    decoder = Decoder(
        rnn_type=C.decoder.rnn_type,
        num_layers=C.decoder.rnn_num_layers,
        num_directions=C.decoder.rnn_num_directions,
        feat_size=C.feat.size,
        feat_len=C.loader.frame_sample_len,
        embedding_size=C.vocab.embedding_size,
        hidden_size=C.decoder.rnn_hidden_size,
        attn_size=C.decoder.rnn_attn_size,
        output_size=vocab.n_vocabs,
        rnn_dropout=C.decoder.rnn_dropout)
    if C.pretrained_decoder_fpath is not None:
        decoder.load_state_dict(torch.load(C.pretrained_decoder_fpath)['decoder'])
        print("Pretrained decoder is loaded from {}".format(C.pretrained_decoder_fpath))
    '''
    for param in decoder.parameters():
        param.requires_grad = False
    print("The parameters of decoder is frozen.")
    '''

    if C.reconstructor.type == 'global':
        reconstructor = GlobalReconstructor(
            rnn_type=C.reconstructor.rnn_type,
            num_layers=C.reconstructor.rnn_num_layers,
            num_directions=C.reconstructor.rnn_num_directions,
            decoder_size=C.decoder.rnn_hidden_size,
            hidden_size=C.reconstructor.rnn_hidden_size,
            rnn_dropout=C.reconstructor.rnn_dropout)
    else:
        reconstructor = LocalReconstructor(
            rnn_type=C.reconstructor.rnn_type,
            num_layers=C.reconstructor.rnn_num_layers,
            num_directions=C.reconstructor.rnn_num_directions,
            decoder_size=C.decoder.rnn_hidden_size,
            hidden_size=C.reconstructor.rnn_hidden_size,
            attn_size=C.reconstructor.rnn_attn_size,
            rnn_dropout=C.reconstructor.rnn_dropout)
    if C.pretrained_reconstructor_fpath is not None:
        reconstructor.load_state_dict(torch.load(C.pretrained_reconstructor_fpath)['reconstructor'])
        print("Pretrained reconstructor is loaded from {}".format(C.pretrained_reconstructor_fpath))

    model = CaptionGenerator(decoder, reconstructor, C.loader.max_caption_len, vocab)
    model.cuda()
    return model


def log_train(summary_writer, e, loss, lr, scores=None):
    summary_writer.add_scalar(C.tx_train_loss, loss['total'], e)
    summary_writer.add_scalar(C.tx_train_cross_entropy_loss, loss['cross_entropy'], e)
    summary_writer.add_scalar(C.tx_train_entropy_loss, loss['entropy'], e)
    summary_writer.add_scalar(C.tx_train_reconstruction_loss, loss['reconstruction'], e)
    summary_writer.add_scalar(C.tx_lr, lr, e)
    print("loss: {} (CE {} + E {} + REC {})".format(loss['total'], loss['cross_entropy'], loss['entropy'],
                                                    loss['reconstruction']))
    if scores is not None:
      for metric in C.metrics:
          summary_writer.add_scalar("TRAIN SCORE/{}".format(metric), scores[metric], e)
      print("scores: {}".format(scores))


def log_val(summary_writer, e, loss, scores=None):
    summary_writer.add_scalar(C.tx_val_loss, loss['total'], e)
    summary_writer.add_scalar(C.tx_val_cross_entropy_loss, loss['cross_entropy'], e)
    summary_writer.add_scalar(C.tx_val_entropy_loss, loss['entropy'], e)
    summary_writer.add_scalar(C.tx_val_reconstruction_loss, loss['reconstruction'], e)
    print("loss: {} (CE {} + E {} + REC {})".format(loss['total'], loss['cross_entropy'], loss['entropy'],
                                                    loss['reconstruction']))
    if scores is not None:
        for metric in C.metrics:
            summary_writer.add_scalar("VAL SCORE/{}".format(metric), scores[metric], e)
        print("scores: {}".format(scores))


def log_test(summary_writer, e, test_scores):
    for metric in C.metrics:
        summary_writer.add_scalar("TEST SCORE/{}".format(metric), test_scores[metric], e)
    print("scores: {}".format(test_scores))


def main():
    print("MODEL ID: {}".format(C.model_id))

    summary_writer = SummaryWriter(C.log_dpath)

    train_iter, val_iter, test_iter, vocab = build_loaders()

    model = build_model(vocab)

    optimizer = torch.optim.Adam(model.parameters(), lr=C.lr, weight_decay=C.weight_decay,
                                 amsgrad=True)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=C.lr_decay_gamma,
                                     patience=C.lr_decay_patience, verbose=True)

    """ Train """
    try:
        best_val_scores = { 'CIDEr': 0. }
        best_epoch = 0
        best_ckpt_fpath = None
        test_scores_at_best_val_score = {}
        for e in range(1, C.epochs + 1):
            print("\n\n\nEpoch {:d}".format(e))

            ckpt_fpath = C.ckpt_fpath_tpl.format(e)

            """ Train """
            print("\n[TRAIN]")
            train_loss = train(e, model, optimizer, train_iter, vocab, C.decoder.rnn_teacher_forcing_ratio,
                               C.reg_lambda, C.recon_lambda, C.gradient_clip)
            log_train(summary_writer, e, train_loss, get_lr(optimizer))

            """ Validation """
            print("\n[VAL]")
            val_loss = evaluate(
                model, val_iter, vocab, C.reg_lambda, C.recon_lambda)
            val_scores, _, _ = score(model, val_iter, vocab)
            log_val(summary_writer, e, val_loss, val_scores)

            if e % C.save_every == 0:
                print("Saving checkpoint at epoch={} to {}".format(e, ckpt_fpath))
                save_checkpoint(model, ckpt_fpath, C)

            if e >= C.lr_decay_start_from:
                lr_scheduler.step(val_loss)
            if val_scores['CIDEr'] > best_val_scores['CIDEr']:
                best_epoch = e
                best_val_scores = val_scores
                best_ckpt_fpath = ckpt_fpath

                test_scores_at_best_val_score, _, _ = score(model, test_iter, vocab)

            for metric in C.metrics:
                summary_writer.add_scalar("TEST SCORE AT BEST VAL SCORE/{}".format(metric),
                                          test_scores_at_best_val_score[metric], e)
    except KeyboardInterrupt:
        if e >= C.save_from:
            print("Saving checkpoint at epoch={}".format(e))
            save_checkpoint(model, ckpt_fpath, C)
        else:
            print("Do not save checkpoint at epoch={}".format(e))
    finally:
        """ Test with Best Model """
        print("\n\n\n[BEST]")
        best_model = load_checkpoint(model, best_ckpt_fpath)
        for metric in C.metrics:
            summary_writer.add_scalar("BEST SCORE/{}".format(metric), test_scores_at_best_val_score[metric], best_epoch)
        save_checkpoint(best_model, C.ckpt_fpath_tpl.format("best"), C)


if __name__ == "__main__":
    main()

