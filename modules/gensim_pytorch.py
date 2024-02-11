#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import gensim
import smart_open
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class MyDataset(Dataset):
    def __init__(self, fname, tokens_only=False):
        self.data = list(self.read_corpus_gzip(fname, tokens_only))

        print(f"self.data {self.data}")

        # self.data is a list of tuples (tokens, tags), find all unique tokens
        # Find all unique tokens
        self.vocab = set()
        for data_itm in self.data:
            print(f"data_itm {data_itm}")
            tokens, _ = data_itm
            self.vocab.update(tokens)

        # Create a mapping from tokens to indices
        self.vocab = list(self.vocab)
        self.vocab_size = len(self.vocab)
        self.vocab_to_int = {word: idx for idx, word in enumerate(self.vocab)}

        # Index the tokens in the data
        self.data_indexed = []
        for data_itm in self.data:
            tokens, tags = data_itm
            tokens_indexed = [self.vocab_to_int[token] for token in tokens]
            self.data_indexed.append((tokens_indexed, tags))

    def __len__(self):
        return len(self.data_indexed)

    def __getitem__(self, idx):
        return self.data_indexed[idx]

    def read_corpus_gzip(self, fname, tokens_only=False):
        """This functions reads the corpus and returns the tokens

        Args:
            fname (file): File to be read
            tokens_only (bool, optional): _description_. Defaults to False.

        Yields:
            _type_: _description_
        """
        with gzip.open(fname, 'rb') as f:
            with smart_open.open(fname, encoding="iso-8859-1") as f:
                for i, line in enumerate(f):
                    tokens = gensim.utils.simple_preprocess(line, deacc=False)
                    if tokens_only:
                        yield tokens
                    else:
                        # For training data, add tags
                        # yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
                        yield (tokens, [i])


# Create Training and Test DataLoaders
train_file = 'path/to/data/test.gz'
train_dataset = MyDataset(train_file)
test_dataset = MyDataset(train_file, tokens_only=False)
batch_size = 1


def collate_batch(batch):
    words, tags = zip(*batch)
    # Convert list of lists to list of tensors for words
    words = [torch.tensor(seq) for seq in words]
    # Pad sequences to the maximum length in the batch
    words = pad_sequence(words, batch_first=True, padding_value=0)
    tags = torch.tensor(tags)
    return words, tags


# Create DataLoaders with the collate function
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch)

# Define a custom Dataset for reading the corpus


class Doc2VecModel(nn.Module):
    def __init__(self, vector_size, vocab_size):
        super(Doc2VecModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, vector_size)
        self.fc = nn.Linear(vector_size, vector_size)

    def forward(self, x):
        x = self.embeddings(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x

    # def forward(self, x):
    #     x = self.embeddings(x)
    #     x = torch.mean(x, dim=1)
    #     x = self.fc(x)
    #     # Apply softmax activation
    #     return F.softmax(x, dim=1)
