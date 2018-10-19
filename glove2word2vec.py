#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import gensim
import logging
import smart_open
import os

try:
    from gensim.models import KeyedVectors
except ImportError:
    from gensim.models import Word2Vec as KeyedVectors

import six
from nlgeval.word2vec.glove2word2vec import glove2word2vec
program=sys.argv[0]
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

def glove2word2vec(glove_vector_file, output_model_file):
    """Convert GloVe vectors into word2vec C format"""

    def get_info(glove_file_name):
        """Return the number of vectors and dimensions in a file in GloVe format."""
        with smart_open.smart_open(glove_file_name,encoding='utf-8') as f:
            num_lines = sum(1 for line in f)
        with smart_open.smart_open(glove_file_name,encoding='utf-8') as f:
            num_dims = len(f.readline().split()) - 1
        return num_lines, num_dims

    def prepend_line(infile, outfile, line):
        """
        Function to prepend lines using smart_open
        """
        with smart_open.smart_open(infile, 'r',encoding='utf-8') as old:
            with smart_open.smart_open(outfile, 'w',encoding='utf-8') as new:
                new.write(str(line.strip()) + "\n")
                for line in old:
                    new.write(line)
        return outfile

    num_lines, dims = get_info(glove_vector_file)

    logger.info('%d lines with %s dimensions' % (num_lines, dims))

    gensim_first_line = "{} {}".format(num_lines, dims)
    model_file = prepend_line(glove_vector_file, output_model_file, gensim_first_line)

    logger.info('Model %s successfully created !!'%output_model_file)

    # Demo: Loads the newly created glove_model.txt into gensim API.
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False) #GloVe Model

    logger.info('Most similar to king are: %s' % model.most_similar(positive=['king'], topn=10))
    logger.info('Similarity score between woman and man is %s ' % model.similarity('woman', 'man'))

    logger.info("Finished running %s", program)

    return model_file

def txt2bin(filename):
    m = KeyedVectors.load_word2vec_format(filename)
    m.vocab[next(six.iterkeys(m.vocab))].sample_int = 1
    m.save(filename.replace('txt', 'bin'), separately=None)
    KeyedVectors.load(filename.replace('txt', 'bin'), mmap='r')


def generate():
    path = '../model'
    glove_vector_file = os.path.join(path, 'deps.words.txt')
    output_model_file = os.path.join(path, 'deps.words.300.model.txt')
    txt2bin(glove2word2vec(glove_vector_file, output_model_file))
    
if __name__ == "__main__":
    generate()
