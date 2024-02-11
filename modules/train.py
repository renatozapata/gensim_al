#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

import gensim


def train_model(train_corpus: list) -> gensim.models.doc2vec.Doc2Vec:
    # Training the Model
    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    return model
