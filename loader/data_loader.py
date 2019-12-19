from __future__ import print_function, division

from collections import defaultdict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms

from loader.transform import UniformSample, RandomSample, ToTensor, TrimExceptAscii, Lowercase, \
                             RemovePunctuation, SplitWithWhiteSpace, Truncate, PadFirst, PadLast, PadToLength, \
                             ToIndex


class CustomVocab(object):
    def __init__(self, caption_fpath, init_word2idx, min_count=1, transform=str.split):
        self.caption_fpath = caption_fpath
        self.min_count = min_count
        self.transform = transform

        self.word2idx = defaultdict(lambda: init_word2idx['<UNK>'])
        self.word2idx.update(init_word2idx)
        self.idx2word = { v: k for k, v in self.word2idx.items() }
        self.word_freq_dict = defaultdict(lambda: 0)
        self.n_vocabs = len(self.word2idx)
        self.n_words = self.n_vocabs
        self.max_sentence_len = -1

        self.build()

    def load_captions(self):
        raise NotImplementedError("You should implement this function.")

    def build(self):
        captions = self.load_captions()
        for caption in captions:
            words = self.transform(caption)
            self.max_sentence_len = max(self.max_sentence_len, len(words))
            for word in words:
                self.word_freq_dict[word] += 1
        self.n_vocabs_untrimmed = len(self.word_freq_dict)
        self.n_words_untrimmed = sum(list(self.word_freq_dict.values()))

        keep_words = [ word for word, freq in self.word_freq_dict.items() if freq >= self.min_count ]

        for idx, word in enumerate(keep_words, len(self.word2idx)):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.n_vocabs = len(self.word2idx)
        self.n_words = sum([ self.word_freq_dict[word] for word in keep_words ])


class CustomDataset(Dataset):
    """ Dataset """

    def __init__(self, C, phase, caption_fpath, transform_frame=None, transform_caption=None):
        self.C = C
        self.phase = phase
        self.caption_fpath = caption_fpath
        self.transform_frame = transform_frame
        self.transform_caption = transform_caption

        self.video_feats = defaultdict(lambda: [])
        self.captions = defaultdict(lambda: [])
        self.data = []

        self.build_video_caption_pairs()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vid, video_feats, caption = self.data[idx]

        if self.transform_frame:
            video_feats = [ self.transform_frame(feat) for feat in video_feats ]
        if self.transform_caption:
            caption = self.transform_caption(caption)

        return vid, video_feats, caption

    def load_video_feats(self):
        for model in self.C.feat.models:
            fpath = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus, model, self.phase)

            fin = h5py.File(fpath, 'r')
            for vid in fin.keys():
                feats = fin[vid].value

                # Fix the number of frames for each video
                if len(feats) < self.C.loader.frame_max_len:
                    num_paddings = self.C.loader.frame_max_len - len(feats)
                    feats = feats.tolist() + [ np.zeros_like(feats[0]) for _ in range(num_paddings) ]
                    feats = np.asarray(feats)
                else:
                    feats = feats[:self.C.loader.frame_max_len]
                assert len(feats) == self.C.loader.frame_max_len

                # Sample fixed number of frames
                sampled_idxs = np.linspace(0, len(feats) - 1, self.C.loader.frame_sample_len, dtype=int)
                feats = feats[sampled_idxs]

                self.video_feats[vid].append(feats)
            fin.close()

    def load_captions(self):
        raise NotImplementedError("You should implement this function.")

    def build_video_caption_pairs(self):
        self.load_video_feats()
        self.load_captions()

        for vid in self.video_feats.keys():
            video_feats = self.video_feats[vid]
            for caption in self.captions[vid]:
                self.data.append(( vid, video_feats, caption ))


class Corpus(object):
    """ Data Loader """

    def __init__(self, C, vocab_cls=CustomVocab, dataset_cls=CustomDataset):
        self.C = C
        self.vocab = None
        self.train_dataset = None
        self.train_data_loader = None
        self.val_dataset = None
        self.val_data_loader = None
        self.test_dataset = None
        self.test_data_loader = None

        self.CustomVocab = vocab_cls
        self.CustomDataset = dataset_cls

        self.transform_sentence = transforms.Compose([
            TrimExceptAscii(self.C.corpus),
            Lowercase(),
            RemovePunctuation(),
            SplitWithWhiteSpace(),
            Truncate(self.C.loader.max_caption_len),
        ])

        self.build()

    def build(self):
        self.build_vocab()
        self.build_data_loaders()

    def build_vocab(self):
        self.vocab = self.CustomVocab(
            self.C.loader.train_caption_fpath,
            self.C.vocab.init_word2idx,
            self.C.loader.min_count,
            transform=self.transform_sentence)

    def build_data_loaders(self):
        """ Transformation """
        if self.C.loader.frame_sampling_method == "uniform":
            Sample = UniformSample
        elif self.C.loader.frame_sampling_method == "random":
            Sample = RandomSample
        else:
            raise NotImplementedError("Unknown frame sampling method: {}".format(self.C.loader.frame_sampling_method))

        self.transform_frame = transforms.Compose([
            Sample(self.C.loader.frame_sample_len),
            ToTensor(torch.float),
        ])
        self.transform_caption = transforms.Compose([
            self.transform_sentence,
            ToIndex(self.vocab.word2idx),
            PadFirst(self.vocab.word2idx['<SOS>']),
            PadLast(self.vocab.word2idx['<EOS>']),
            PadToLength(self.vocab.word2idx['<PAD>'], self.vocab.max_sentence_len + 2), # +2 for <SOS> and <EOS>
            ToTensor(torch.long),
        ])

        self.train_dataset = self.build_dataset("train", self.C.loader.train_caption_fpath)
        self.val_dataset = self.build_dataset("val", self.C.loader.val_caption_fpath)
        self.test_dataset = self.build_dataset("test", self.C.loader.test_caption_fpath)

        self.train_data_loader = self.build_data_loader(self.train_dataset)
        self.val_data_loader = self.build_data_loader(self.val_dataset)
        self.test_data_loader = self.build_data_loader(self.test_dataset)

    def build_dataset(self, phase, caption_fpath):
         dataset = self.CustomDataset(
            self.C,
            phase,
            caption_fpath,
            transform_frame=self.transform_frame,
            transform_caption=self.transform_caption)
         return dataset


    def collate_fn(self, batch):
        vids, video_feats, captions = zip(*batch)
        video_feats_list = zip(*video_feats)

        video_feats_list = [ torch.stack(video_feats) for video_feats in video_feats_list ]
        captions = torch.stack(captions)

        video_feats_list = [ video_feats.float() for video_feats in video_feats_list ]
        captions = captions.float()

        """ (batch, seq, feat) -> (seq, batch, feat) """
        captions = captions.transpose(0, 1)

        return vids, video_feats_list, captions

    def build_data_loader(self, dataset):
        data_loader = DataLoader(
            dataset,
            batch_size=self.C.batch_size,
            shuffle=False, # If sampler is specified, shuffle must be False.
            sampler=RandomSampler(dataset, replacement=False),
            num_workers=self.C.loader.num_workers,
            collate_fn=self.collate_fn)
        return data_loader

