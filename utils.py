import numpy as np
import torch

from nltk.tokenize import RegexpTokenizer
from torch.autograd import Variable
from tqdm import tqdm


def get_sequences_lengths(sequences, masking=0, dim=1):
    if len(sequences.size()) > 2:
        sequences = sequences.sum(dim=2)

    masks = torch.ne(sequences, masking)
    lengths = masks.sum(dim=dim)
    return lengths


def cuda(obj):
    if torch.cuda.is_available():
        obj = obj.cuda()
    return obj


def variable(obj, volatile=False):
    if isinstance(obj, (list, tuple)):
        return [variable(o, volatile=volatile) for o in obj]

    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)

    obj = cuda(obj)
    obj = Variable(obj, volatile=volatile)
    return obj


def idx2text(token_ids, idx2w):
    sent = ""
    for id in token_ids:
        if idx2w[id] == '<end>':
            break
        if idx2w[id] != '<pad>':
            sent += idx2w[id] + ' '
    return sent


def save_weights(model, filename):
    if not isinstance(filename, str):
        filename = str(filename)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    torch.save(model.state_dict(), filename)


def text2idx(words, voc):
    if isinstance(words, str):
        words = words.lower()
        tokenizer = RegexpTokenizer('\w+')
        words = tokenizer.tokenize(words)
    idx = []
    for word in words:
        if word in voc.keys():
            idx.append(voc[word])
        else:
            idx.append(voc['<unk>'])
    return idx


def encode_sentence(sentence, w2idx, max_len, pad_token='<pad>', end_token='<end>'):
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
