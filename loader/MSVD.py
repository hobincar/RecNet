import pandas as pd

from loader.data_loader import CustomVocab, CustomDataset, Corpus


class MSVDVocab(CustomVocab):
    """ MSVD Vocaburary """

    def load_captions(self):
        df = pd.read_csv(self.caption_fpath)
        df = df[df['Language'] == 'English']
        df = df[pd.notnull(df['Description'])]
        captions = df['Description'].values
        return captions

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


class MSVDDataset(CustomDataset):
    """ MSVD Dataset """

    def load_captions(self):
        df = pd.read_csv(self.caption_fpath)
        df = df[df['Language'] == 'English']
        df = df[[ 'VideoID', 'Start', 'End', 'Description' ]]
        df = df[pd.notnull(df['Description'])]

        for video_id, start, end, caption in df.values:
            vid = "{}_{}_{}".format(video_id, start, end)
            self.captions[vid].append(caption)


class MSVD(Corpus):
    """ MSVD Corpus """

    def __init__(self, C):
        super(MSVD, self).__init__(C, MSVDVocab, MSVDDataset)

