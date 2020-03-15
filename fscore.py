import embeddings
from cupy_utils import *

import argparse
import collections
import numpy as np
import sys

def readdict(f):
    d = collections.OrderedDict()
    for line in f:
        l = line.split()[:2]
        if len(l) == 1:
            s = l[0].strip()
            d[s] = set()
        else:
            s, t = line.split()[:2]
            assert(s == s.strip())
            assert(t == t.strip())
            if s not in d:
                d[s] = set()
            d[s].add(t)
    return d

def readpreds(f, k=3, thres=0.3, maxgap=0.3):
    d = collections.defaultdict(lambda: collections.Counter())
    maxval = collections.Counter()
    for line in f:
        s, t, score = line.split('\t')[:3]
        score = float(score)
        if score > maxval[s]:
            maxval[s] = score
        # for BUCC, always add the first prediction
        # assuming sorted
        if score < thres:
            continue
        if maxval[s] - score > maxgap:
            continue
        d[s][t] = score
    pred = collections.defaultdict(set)
    for s in d:
        for t in d[s].most_common(k):
            pred[s].add(t[0])
    return pred, d

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
    parser.add_argument('--k', default=2, type=int, help='the max amount of predictions per word')
    parser.add_argument('--thres', default=0.3, type=float, help='the threshold')
    parser.add_argument('--maxgap', default=10, type=float, help='the maximum gap with the biggest')
    parser.add_argument('--predmode', action='store_true', help='output in source order instead of confidence order')
    args = parser.parse_args()

    with open(args.dictionary, encoding=args.encoding, errors='surrogateescape') as f:
        ref = readdict(f)
    with open(args.translations, encoding=args.encoding, errors='surrogateescape') as f:
        trans, scores = readpreds(f, args.k, args.thres, args.maxgap)

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
    # assert 0 == np.sum([1 if f[1] == 'missing' else 0 for f in statsp])
    statsr = precision(ref, trans, tag='FN')
    oov = np.sum([1 if f[1] == 'missing' else 0 for f in statsr])

    allstats = collections.defaultdict(dict)
    for l in statsp + statsr:
        st, status = l
        s, t = st
        if s not in ref:
            # filter out random predictions
            continue
        if t in allstats[s]:
            assert allstats[s][t] == 'TP'
        allstats[s][t] = status

    if args.predmode:
        for s in ref:
            for tscore in scores[s].most_common():
                t, score = tscore
                print(f'{s}\t{t}\t{score}')
        return

    for s in allstats:
        for t in allstats[s]:
            stscore = scores[s][t]
            print(f'{s}\t{t}\t{allstats[s][t]}\t{stscore}')

    prec = np.mean([1 if f[1] == 'TP' else 0 for f in statsp if f[1] != 'missing'])
    recall = np.mean([1 if f[1] == 'TP' else 0 for f in statsr])
    eps = 1e-10
    F = 0 if prec < eps or recall < eps else 2 / (1/prec + 1/recall)
    printe = lambda x: print(x, file=sys.stderr)
    printe(f'reference: {args.dictionary}')
    printe(f'predictions: {args.translations}')
    printe(f'P/R/F: {prec:.2%}\t{recall:.2%}\t{F:.2%}')
    printe(f'num oov {oov} / {len(ref)}')
    printe(f'len(ref):\t{len(ref)}')
    printe(f'len(trans):\t{len(trans)}')
    inter = trans.keys() & ref.keys()
    printe(f'covered:\t{len(inter)}')
    numpreds = 0
    for k in inter:
        numpreds += len(trans[k])
    printe(f'numpreds:\t{numpreds}')
    total_preds = 0
    for k in trans:
        total_preds += len(trans[k])
    printe(f'total_preds:\t{total_preds}')

if __name__ == '__main__':
    main()
