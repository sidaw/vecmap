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
        assert(s == s.strip())
        assert(t == t.strip())
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

    def precision(testd, refd, tag='FP'):
        stats = []
        for s, t in pairs(testd):
            status = tag
            if s not in refd:
                status = 'missing'
            elif t in refd[s]:
                status = 'TP'
            stats.append([(s, t), status])
        return stats

    statsp = precision(trans, ref, tag='FP')
    # everything being predicted should be in the dict
    assert 0 == np.sum([1 if f[1] == 'missing' else 0 for f in statsp])
    statsr = precision(ref, trans, tag='FN')
    oov = np.sum([1 if f[1] == 'missing' else 0 for f in statsr])

    allstats = {}
    for s in statsp + statsr:
        if s[0] in allstats:
            assert allstats[s[0]] == 'TP'

        allstats[s[0]] = s[1]

    for st in allstats:
        print(f'{st[0]}\t{st[1]}\t{allstats[st]}')

    prec = np.mean([1 if f[1] == 'TP' else 0 for f in statsp])
    recall = np.mean([1 if f[1] == 'TP' else 0 for f in statsr])
    eps = 1e-10
    F = 0 if prec < eps or recall < eps else 2 / (1/prec + 1/recall)
    print(f'reference: {args.dictionary}')
    print(f'predictions: {args.translations}')
    print(f'P/R/F: {prec:.2%}\t{recall:.2%}\t{F:.2%}')
    print(f'num oov {oov} / {len(ref)}')
    print(f'len(ref):\t{len(ref)}')
    print(f'len(trans):\t{len(trans)}')

if __name__ == '__main__':
    main()
