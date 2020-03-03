# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import embeddings
from cupy_utils import *

import argparse
import collections
import numpy as np
import sys


BATCH_SIZE = 500


def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate embeddings of two languages in a shared space in word translation induction')
    parser.add_argument('src_embeddings', help='the source language embeddings')
    parser.add_argument('trg_embeddings', help='the target language embeddings')
    parser.add_argument('-d', '--dictionary', default=sys.stdin.fileno(), help='the test dictionary file (defaults to stdin)')
    parser.add_argument('--retrieval', default='nn', choices=['nn', 'invnn', 'invsoftmax', 'csls', 'fcsls'], help='the retrieval method (nn: standard nearest neighbor; invnn: inverted nearest neighbor; invsoftmax: inverted softmax; csls: cross-domain similarity local scaling)')
    parser.add_argument('--inv_temperature', default=1, type=float, help='the inverse temperature (only compatible with inverted softmax)')
    parser.add_argument('--inv_sample', default=None, type=int, help='use a random subset of the source vocabulary for the inverse computations (only compatible with inverted softmax)')
    parser.add_argument('--neighborhood', default=10, type=int, help='the neighborhood size (only compatible with csls)')
    parser.add_argument('--nbest', default=3, type=int, help='number of candidates to get')
    parser.add_argument('--dot', action='store_true', help='use the dot product in the similarity computations instead of the cosine')
    parser.add_argument('--verbose', action='store_true', help='verbose, print more information')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--seed', type=int, default=0, help='the random seed')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
    parser.add_argument('--threshold', default=0, type=int, help='vocab limit for reading the embedding')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    args = parser.parse_args()

    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'

    # Read input embeddings
    srcfile = open(args.src_embeddings, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_embeddings, encoding=args.encoding, errors='surrogateescape')
    src_words, x = embeddings.read(srcfile, dtype=dtype, threshold=args.threshold)
    trg_words, z = embeddings.read(trgfile, dtype=dtype, threshold=args.threshold)

    # NumPy/CuPy management
    if args.cuda:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = get_cupy()
        x = xp.asarray(x)
        z = xp.asarray(z)
    else:
        xp = np
    xp.random.seed(args.seed)

    # Length normalize embeddings so their dot product effectively computes the cosine similarity
    if not args.dot:
        embeddings.length_normalize(x)
        embeddings.length_normalize(z)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_ind2word = {i: word for i, word in enumerate(trg_words)}
    src_ind2word = {i: word for i, word in enumerate(src_words)}

    # Read dictionary and compute coverage
    f = open(args.dictionary, encoding=args.encoding, errors='surrogateescape')
    oov = set()
    vocab = set()
    src = [] 
    for line in f:
        if '\t' in line:
            w, _ = line.split()
        else:
            w = line.strip()
        if w in vocab:
            continue
        try:
            src.append(src_word2ind[w])
            vocab.add(w)
        except KeyError:
            oov.add(w)
    
    # if args.verbose:
        # print(f'{len(oov)} oovs: ' + '|'.join(list(oov)), file=sys.stderr)

    if args.retrieval == 'nn': # Standard nearest neighbor
        queries = x[src]
        topvals, topinds = embeddings.faiss_knn(queries, z, k=args.nbest)
        for i, wind in enumerate(src):
            w = src_ind2word[wind]
            for k, tind in enumerate(topinds[i]):
                wt = trg_ind2word[tind]
                st = topvals[i, k]
                print(f'{w}\t{wt}\t{st:.3f}')
    elif args.retrieval == 'fcsls':  # Cross-domain similarity local scaling
        sim_bwd, _ = embeddings.faiss_knn(z, x, k=args.neighborhood)
        knn_sim_bwd = sim_bwd.mean(axis=1)
        queries = x[src]
        topvals, topinds = embeddings.faiss_knn(queries, z, k=20)
        for i, wind in enumerate(src):
            w = src_ind2word[wind]
            for k, tind in enumerate(topinds[i]):
                wt = trg_ind2word[tind]
                st = 2 * topvals[i, k] - knn_sim_bwd[topinds[i, k]]
                print(f'{w}\t{wt}\t{st:.3f}')
    elif args.retrieval == 'csls':  # Cross-domain similarity local scaling
        sim_bwd, _ = embeddings.faiss_knn(z, x, k=args.neighborhood)
        knn_sim_bwd = sim_bwd.mean(axis=1)
        queries = x[src]

        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            similarities = 2*x[src[i:j]].dot(z.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN
            nn = (-similarities).argpartition(args.nbest, axis=1)
            for k in range(j-i):
                w = src_ind2word[src[i+k]]
                for tind in nn[k, :args.nbest]:
                    wt = trg_ind2word[tind]
                    st = similarities[k, tind]
                    print(f'{w}\t{wt}\t{st:.3f}')


if __name__ == '__main__':
    main()
