from torch.utils.data import Dataset

from utils import encode_sentence
from vocabulary import Vocabulary


class AutoencoderDataset(Dataset):
    def __init__(self, filename, voc=None, max_len=None):
        super(AutoencoderDataset, self).__init__()

        # .txt file with separate sentences on separate lines
        with open(filename) as f:
            self.data = f.readlines()

        # For padding to the same length
        if max_len is not None:
            self.max_len = max_len
        else:
            self.max_len = self.get_max_example_len()

        # For converting words to their ids
        if voc is None:
            voc = Vocabulary(' '.join(self.data))

        self.w2idx = voc.w2idx
        self.idx2w = voc.idx2w

    def get_max_example_len(self):
        from nltk.tokenize import RegexpTokenizer
        tokenizer = RegexpTokenizer('\w+')
        max_len = max(map(len, map(tokenizer.tokenize, self.data)))
        return max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        ehr = encode_sentence(sentence, self.w2idx, self.max_len)
        return ehr
