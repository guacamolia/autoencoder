import itertools
import warnings
import pickle


class Vocabulary:
    """
    Create a vocabulary of a training dataset.
    Args:
        text: Input text.
        If str is given, tokenization is done by RegexpTokenizer('\w+').

    Attributes:
        tokens: A set of all tokens present in the dataset
        w2idx (dict): A dictionary matching tokens to their ids
        idx2w (dict): A reversed dictionary matching ids to tokens

    """
    def __init__(self, text):
        if isinstance(text, str):
            from nltk.tokenize import RegexpTokenizer
            tokenizer = RegexpTokenizer('\w+')
            self.tokens = set(tokenizer.tokenize(text.lower()))

        elif isinstance(text, list):
            self.tokens = set(itertools.chain.from_iterable(text))
        elif isinstance(text, set):
            self.tokens = text
        else:
            raise TypeError('Invalid type of input data. Acceptable types are string, list and set.')

        self.w2idx = {word: idx+1 for idx, word in enumerate(self.tokens)}
        self.idx2w = {idx: word for word, idx in self.w2idx.items()}
        self.w2idx['<pad>'] = 0
        self.idx2w[0] = '<pad>'
        self.add_token('<unk>')
        self.add_token('<start>')
        self.add_token('<end>')
        self.add_token('<num>')
        assert len(self.w2idx) == len(self.idx2w)

    def add_token(self, token):
        """

        Args:
            token: a token to be added

        Returns: updated vocabulary

        """
        if token in self.w2idx.keys():
            print('Token "{}" is already present in the vocabulary'.format(token))
            return self
        cur_len = self.get_length()
        self.w2idx[token] = cur_len
        self.idx2w[cur_len] = token

        assert len(self.w2idx) == len(self.idx2w)
        return self

    def get_length(self):
        """
        Get the current vocabulary size.
        Returns (int): Vocabulary size

        """
        return len(self.w2idx)

    def save(self, loc):
        with open(loc, 'wb') as f:
            pickle.dump(self, f)
        print('Vocabulary saved to {}'.format(loc))


if __name__ == "__main__":
    with open("data/train.txt") as f:
        notes = f.read()
    voc = Vocabulary(notes)
    print("The size of the vocabulary is: {}".format(voc.get_length()))
    print(voc.tokens)