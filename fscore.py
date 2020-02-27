import embeddings
from cupy_utils import *

import argparse
import collections
import numpy as np
import sys

def readdict(f):
    d = collections.defaultdict(set)
    for line in f:
        s, t = line.split()[:2]
        d[s].add(t)
    return d

def pairs(setd):
    for k in setd:
        for v in setd[k]:
            yield k, v

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate embeddings of two languages in a shared space in word translation induction')
    parser.add_argument('translations', default=sys.stdin.fileno(), help='file containing nbest of the tranlsations')
    parser.add_argument('-d', '--dictionary', help='the test dictionary file (defaults to stdin)')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    args = parser.parse_args()

    with open(args.dictionary, encoding=args.encoding, errors='surrogateescape') as f:
        ref = readdict(f)
    with open(args.translations, encoding=args.encoding, errors='surrogateescape') as f:
        trans = readdict(f)

    def precision(testd, refd):
        N = 0
        TP = 0
        NF = 0
        for s, t in pairs(testd):
            N += 1
            if s not in refd:
                NF += 1
            elif t in refd[s]:
                TP += 1
        return TP / N, NF

    prec, shouldbe0 = precision(trans, ref)
    recall, oov = precision(ref, trans)
    assert shouldbe0 == 0
    eps = 1e-10
    F = 0 if prec < eps or recall < eps else 2 / (1/prec + 1/recall)
    print(f'reference: {args.dictionary}')
    print(f'predictions: {args.translations}')
    print(f'P / R / F: {prec:.2%}\t{recall:.2%}\t{F:.2%}')
    print(f'num oov {oov}')


if __name__ == '__main__':
    main()
