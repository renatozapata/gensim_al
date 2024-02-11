#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

import gensim
import smart_open
import logging
import gzip


def read_corpus(fname, tokens_only=False):
    """This functions reads the corpus and returns the tokens

    Args:
        fname (file): File to be read
        tokens_only (bool, optional): _description_. Defaults to False.

    Yields:
        _type_: _description_
    """
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


def read_input(input_file):
    """This method reads the input file which is in gzip format"""

    logging.info("reading file {0}...this may take a while".format(input_file))
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):

            if (i % 100000 == 0):
                logging.info("read {i} lines of log".format(i))
            # do some pre-processing and return list of words for each review
            # text
            yield gensim.utils.simple_preprocess(line)


def read_corpus_gzip(fname, tokens_only=False):
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
                    yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
