import numpy as np
import torch

from nltk.tokenize import RegexpTokenizer
from torch.autograd import Variable
from tqdm import tqdm


def get_sequences_lengths(sequences, pad_idx=0, dim=1):
    """
    Helper function for torch.pack_padded_sequence
    Args:
        sequences: tensor of [BxL] word ids
        pad_idx: index of the padding token in the vocabulary. 0 by default
        dim: dimension along which sequences lengths are calculated

    Returns (torch.Tensor):  [Bx1] tensor with length of every sequence in a batch

    """
    if len(sequences.size()) > 2:
        sequences = sequences.sum(dim=2)

    masks = torch.ne(sequences, pad_idx)
    lengths = masks.sum(dim=dim)
    return lengths


def save_weights(model, filename):
    """
    Saves trained model weights to a specified location
    Args:
        model: nn.Module to be saved
        filename (str): location where to save

    """
    if not isinstance(filename, str):
        filename = str(filename)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    torch.save(model.state_dict(), filename)


# Handling CUDA
def cuda(obj):
    """

    Args:
        obj: object to be transferred to CUDA

    Returns:  .cuda() object if CUDA is available

    """
    if torch.cuda.is_available():
        obj = obj.cuda()
    return obj


def variable(obj):
    """
    Takes a tensor or an iterable and converts every item to .cuda() object if CUDA is available.
    Wraps tensors into Variable.
    Args:
        obj: object to be transferred to CUDA

    """
    if isinstance(obj, (list, tuple)):
        return [variable(o) for o in obj]

    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)

    obj = cuda(obj)
    obj = Variable(obj)
    return obj


# Text processing functions
def text2idx(words, w2idx):
    """
    Converts a string or an iterable of tokens into token ids according to the vocabulary.
    Args:
        words (str or iterable): Input string or an iterable of tokens.
        If a string is used, the default tokenizer is RegexpTokenizer('w+')

        w2idx (dict):  A dictionary matching words to their ids

    Returns: a list of token ids

    """
    if isinstance(words, str):
        words = words.lower()
        tokenizer = RegexpTokenizer('\w+')
        words = tokenizer.tokenize(words)
    ids = []
    for word in words:
        if word in w2idx.keys():
            ids.append(w2idx[word])
        else:
            ids.append(w2idx['<unk>'])
    return ids


def idx2text(token_ids, idx2w):
    """
    Converts a list of token ids into a text string.
    Args:
        token_ids (list): Input list of token ids
        idx2w (dict): A reversed dictionary matching indices to tokens

    Returns (str): Text string

    """
    sent = ""
    for id in token_ids:
        if idx2w[id] == '<end>':
            break
        if idx2w[id] != '<pad>':
            sent += idx2w[id] + ' '
    return sent


def encode_sentence(sentence, w2idx, max_len, pad_token='<pad>', end_token='<end>'):
    """
    Processes the sequence of ids by padding it to a specified length and adding an ending token
    Args:
        sentence (list): A list of tokens
        w2idx (dict): A dictionary matching words to their ids
        max_len: Maximum length for padding. Longer sentences are cropped
        pad_token: Padding token
        end_token: Ending token

    Returns (numpy array): Transformed sentence

    """
    enc_sentence = text2idx(sentence, w2idx)
    if len(enc_sentence) > max_len:
        enc_sentence = enc_sentence[:max_len]
    enc_sentence = enc_sentence + [w2idx[end_token]]
    enc_sentence = enc_sentence + [w2idx[pad_token]]*(max_len + 1 - len(enc_sentence))
    enc_sentence = np.array(enc_sentence)
    return enc_sentence


def match_embeddings(idx2w, w2vec, dim):
    embeddings = []
    oov = 0
    voc_size = len(idx2w)
    for idx in tqdm(range(voc_size)):
        word = idx2w[idx]
        if word not in w2vec:
            oov += 1
            print('OOV: "{}"'.format(word))
            embeddings.append(np.random.uniform(low=-1.0, high=1.0, size=(dim, )))
        else:
            embeddings.append(w2vec[word])
    print('{} words are out of pretrained vectors voc'.format(oov))
    embeddings = np.stack(embeddings)
    return embeddings
