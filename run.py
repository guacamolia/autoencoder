""" Vanilla autoencoder implementation in PyTorch. Requires pytorch >= v0.4"""
import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from autoencoder import Autoencoder
from dataset import AutoencoderDataset
from train_model import train
from utils import cuda, match_embeddings
from vocabulary import Vocabulary


def main():

    # PARAMETERS #
    max_len = 20
    hidden_size = 128
    embedding_dim = 200
    batch_size = 75
    n_epochs = 1000
    lr = 0.0001

    # DATA FILES #
    voc_file = 'data/vocabulary.pkl'
    w2vec_file = 'data/w2vec.pkl'
    train_file = 'data/train.txt'
    dev_file = 'data/test.txt'

    # VOCABULARY #
    with open(train_file) as f:
        voc = Vocabulary(f.read())
    w2idx = voc.w2idx
    idx2w = voc.idx2w
    init_idx = w2idx['<start>']
    end_idx = w2idx['<end>']
    padding_idx = w2idx['<pad>']
    voc_size = voc.get_length()
    voc.save(voc_file)

    # PRETRAINED EMBEDDINGS #
    with open(w2vec_file, 'rb') as f:
        w2vec = pickle.load(f)
    embedding_dim = len(random.choice(list(w2vec.values())))
    embeddings = torch.Tensor(match_embeddings(idx2w, w2vec, dim=embedding_dim))

    # DATASET #
    dataset_train = AutoencoderDataset(train_file, voc, max_len)
    dataset_dev = AutoencoderDataset(dev_file, voc, max_len)

    # DATALOADER #
    dataloader_train = DataLoader(dataset_train, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dataloader_dev = DataLoader(dataset_dev, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dataloaders = {'train': dataloader_train, 'dev': dataloader_dev}

    # MODEL #
    model = cuda(nn.DataParallel(Autoencoder(hidden_size, voc_size, padding_idx, init_idx, max_len, embeddings)))
    model_loc = "saved_models/autoencoder_trained.pt"

    # MODEL PARAMS #
    optimizer = optim.Adam([parameter for parameter in list(model.parameters()) if parameter.requires_grad], lr)
    criterion = nn.CrossEntropyLoss()

    # TRAIN #
    train(model, model_loc, criterion, optimizer, dataloaders, n_epochs, idx2w)


if __name__ == "__main__":
    main()
