import inspect
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import losses
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor


class LossChecker:
    def __init__(self, num_losses):
        self.num_losses = num_losses

        self.losses = [ [] for _ in range(self.num_losses) ]

    def update(self, *loss_vals):
        assert len(loss_vals) == self.num_losses

        for i, loss_val in enumerate(loss_vals):
            self.losses[i].append(loss_val)

    def mean(self, last=0):
        mean_losses = [ 0. for _ in range(self.num_losses) ]
        for i, loss in enumerate(self.losses):
            _loss = loss[-last:]
            mean_losses[i] = sum(_loss) / len(_loss)
        return mean_losses


def parse_batch(batch):
    vids, feats, captions = batch
    feats = [ feat.cuda() for feat in feats ]
    feats = torch.cat(feats, dim=2)
    captions = captions.long().cuda()
    return vids, feats, captions


def train(e, model, optimizer, train_iter, vocab, teacher_forcing_ratio, reg_lambda, recon_lambda, gradient_clip):
    model.train()

    loss_checker = LossChecker(4)
    PAD_idx = vocab.word2idx['<PAD>']
    for b, batch in enumerate(train_iter, 1):
        _, feats, captions = parse_batch(batch)
        optimizer.zero_grad()
        output, feats_recon = model(feats, captions, teacher_forcing_ratio)
        cross_entropy_loss = F.nll_loss(output[1:].view(-1, vocab.n_vocabs),
                                        captions[1:].contiguous().view(-1),
                                        ignore_index=PAD_idx)
        entropy_loss = reg_lambda * losses.entropy_loss(output[1:], ignore_mask=(captions[1:] == PAD_idx))
        if model.reconstructor._type == 'global':
            reconstruction_loss = recon_lambda * \
                losses.global_reconstruction_loss(feats, feats_recon, keep_mask=(captions != PAD_idx))
        else:
            reconstruction_loss = recon_lambda * \
                losses.local_reconstruction_loss(feats, feats_recon)

        loss = cross_entropy_loss + entropy_loss + reconstruction_loss
        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        loss_checker.update(loss.item(), cross_entropy_loss.item(), entropy_loss.item(), reconstruction_loss.item())
        if len(train_iter) < 10 or b % (len(train_iter) // 10) == 0:
            inter_loss, inter_cross_entropy_loss, inter_entropy_loss, inter_reconstruction_loss = \
                loss_checker.mean(last=10)
            print("\t[{:d}/{:d}] loss: {:.4f} = CE {:.4f} + E {:.4f} + REC {:.4f}".format(
                b, len(train_iter), inter_loss, inter_cross_entropy_loss, inter_entropy_loss,
                inter_reconstruction_loss))

    total_loss, cross_entropy_loss, entropy_loss, reconstruction_loss = loss_checker.mean()
    loss = {
        'total': total_loss,
        'cross_entropy': cross_entropy_loss,
        'entropy': entropy_loss,
        'reconstruction': reconstruction_loss,
    }
    return loss


def evaluate(model, val_iter, vocab, reg_lambda, recon_lambda):
    model.eval()

    loss_checker = LossChecker(4)
    PAD_idx = vocab.word2idx['<PAD>']
    for b, batch in enumerate(val_iter, 1):
        _, feats, captions = parse_batch(batch)
        output, feats_recon = model(feats, captions)
        cross_entropy_loss = F.nll_loss(output[1:].view(-1, vocab.n_vocabs),
                                        captions[1:].contiguous().view(-1),
                                        ignore_index=PAD_idx)
        if model.reconstructor._type == 'global':
            reconstruction_loss = recon_lambda * \
                losses.global_reconstruction_loss(feats, feats_recon, keep_mask=(captions != PAD_idx))
        else:
            reconstruction_loss = recon_lambda * \
                losses.local_reconstruction_loss(feats, feats_recon)
        entropy_loss = reg_lambda * losses.entropy_loss(output[1:], ignore_mask=(captions[1:] == PAD_idx))
        loss = cross_entropy_loss + entropy_loss + reconstruction_loss
        loss_checker.update(loss.item(), cross_entropy_loss.item(), entropy_loss.item(), reconstruction_loss.item())

    total_loss, cross_entropy_loss, entropy_loss, reconstruction_loss = loss_checker.mean()
    loss = {
        'total': total_loss,
        'cross_entropy': cross_entropy_loss,
        'entropy': entropy_loss,
        'reconstruction': reconstruction_loss,
    }
    return loss


def score(model, data_iter, vocab):
    def build_score_iter(data_iter):
        score_dataset = {}
        for batch in iter(data_iter):
            vids, feats, _ = parse_batch(batch)
            for vid, feat in zip(vids, feats):
                if vid not in score_dataset:
                    score_dataset[vid] = feat
        score_iter = []
        vids = score_dataset.keys()
        feats = score_dataset.values()
        batch_size = 100
        while len(vids) > 0:
            score_iter.append(( vids[:batch_size], torch.stack(feats[:batch_size]) ))
            vids = vids[batch_size:]
            feats = feats[batch_size:]
        return score_iter

    def build_refs(data_iter):
        vid_idx = 0
        vid2idx = {}
        refs = {}
        for batch in iter(data_iter):
            vids, _, captions = parse_batch(batch)
            captions = captions.transpose(0, 1)
            for vid, caption in zip(vids, captions):
                if vid not in vid2idx:
                    vid2idx[vid] = vid_idx
                    refs[vid2idx[vid]] = []
                    vid_idx += 1
                caption = idxs_to_sentence(caption, vocab.idx2word, vocab.word2idx['<EOS>'])
                refs[vid2idx[vid]].append(caption)
        return refs, vid2idx

    model.eval()

    PAD_idx = vocab.word2idx['<PAD>']
    score_iter = build_score_iter(data_iter)
    refs, vid2idx = build_refs(data_iter)

    hypos = {}
    for vids, feats in score_iter:
        captions = model.describe(feats, beam_width=5, beam_alpha=0.)
        captions = [ idxs_to_sentence(caption, vocab.idx2word, vocab.word2idx['<EOS>']) for caption in captions ]
        for vid, caption in zip(vids, captions):
            hypos[vid2idx[vid]] = [ caption ]
    scores = calc_scores(refs, hypos)
    return scores, refs, hypos


# refers: https://github.com/zhegan27/SCN_for_video_captioning/blob/master/SCN_evaluation.py
def calc_scores(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


# refers: https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def idxs_to_sentence(idxs, idx2word, EOS_idx):
    words = []
    for idx in idxs[1:]:
        idx = idx.item()
        if idx == EOS_idx:
            break
        word = idx2word[idx]
        words.append(word)
    sentence = ' '.join(words)
    return sentence


def cls_to_dict(cls):
    properties = dir(cls)
    properties = [ p for p in properties if not p.startswith("__") ]
    d = {}
    for p in properties:
        v = getattr(cls, p)
        if inspect.isclass(v):
            v = cls_to_dict(v)
            v['was_class'] = True
        d[p] = v
    return d


# refers https://stackoverflow.com/questions/1305532/convert-nested-python-dict-to-object
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def dict_to_cls(d):
    cls = Struct(**d)
    properties = dir(cls)
    properties = [ p for p in properties if not p.startswith("__") ]
    for p in properties:
        v = getattr(cls, p)
        if isinstance(v, dict) and 'was_class' in v and v['was_class']:
            v = dict_to_cls(v)
        setattr(cls, p, v)
    return cls


def load_checkpoint(model, ckpt_fpath):
    checkpoint = torch.load(ckpt_fpath)
    model.decoder.load_state_dict(checkpoint['decoder'])
    return model


def save_checkpoint(e, model, ckpt_fpath, config):
    ckpt_dpath = os.path.dirname(ckpt_fpath)
    if not os.path.exists(ckpt_dpath):
        os.makedirs(ckpt_dpath)

    torch.save({
        'epoch': e,
        'decoder': model.decoder.state_dict(),
        'reconstructor': model.reconstructor.state_dict(),
        'config': cls_to_dict(config),
    }, ckpt_fpath)


def save_result(refs, hypos, save_dpath_root, corpus, phase):
    save_dpath = os.path.join(save_dpath_root, corpus)
    if not os.path.exists(save_dpath):
        os.makedirs(save_dpath)

    save_fpath = os.path.join(save_dpath, "{}.tsv".format(phase))
    with open(save_fpath, 'w') as fout:
        for vid in refs:
            ref = ', '.join(refs[vid])
            hypo = hypos[vid][0]
            line = '\t'.join([ str(vid), hypo, ref ])
            fout.write("{}\n".format(line))

