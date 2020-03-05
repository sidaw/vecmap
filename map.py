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
import re
import sys
import time
from scipy.special import softmax
xp = np

def sample(xw, zw, T, kbest=10):
    pass

def sample_matches(matches, p=0.3):
    sampled = {}
    for k in matches:
        if np.random.rand() < p:
            sampled[k] = matches[k]
    return sampled

def _find_matches(xw, zw, T, kbest=30, threshold=0.3):
    # match = (sind, tind, weight?)
    # numcands = kbest
    topvals, topinds = embeddings.faiss_knn(xw, zw, k=kbest, dist='IP')
    # numcands = 30
    # topvals, topinds = embeddings.faiss_csls(xw, zw, k=numcands, dist='IP')
    objective = np.mean(topvals[:, 0])
    mean = np.mean(topvals[:, 0])
    std = np.std(topvals[:, 0])
    matches = {}
    print(f'mean\t{mean:.2%}')
    print(f'std\t{std:.2%}')
    for i in range(xw.shape[0]):
        topvali = topvals[i]
        j = 0
        valmax = topvali[j]
        if valmax < mean - 0.5 * std:
            continue
        # topprobs = softmax(topvali / 0.02)
        # j = xp.random.choice(range(numcands), p=topprobs)
        # hit = topinds[i, j]
        hit = topinds[i, j]
        if (i, hit) not in matches:
            matches[(i, hit)] = valmax
        # matches[(i, hit)] = (1-eta) * matches[(i, hit)] + eta * topvali[j]
    return matches, objective


def find_matches(matches, xw, zw, excluded, T, kbest=30):
    matches_fwd, obj_fwd = _find_matches(xw, zw, T, kbest=10)
    matches_rev, obj_rev = _find_matches(zw, xw, T, kbest=10)
    matches = {}
    # for m in matches_fwd:
    #     matches[m] = matches_fwd[m]
    for r in matches_rev:
        m = (r[1], r[0])
        if m in matches_fwd:
            if m in excluded:
                continue
            matches[m] = 0.5 * (matches_rev[r] + matches_fwd[m])
        # else:
        #     matches[m] = 1
    return matches, (obj_fwd + obj_rev) / 2


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Map word embeddings in two languages into a shared space')
    parser.add_argument('src_input', help='the input source embeddings')
    parser.add_argument('trg_input', help='the input target embeddings')
    parser.add_argument('src_output', help='the output source embeddings')
    parser.add_argument('trg_output', help='the output target embeddings')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    parser.add_argument('--batch_size', default=10000, type=int, help='batch size (defaults to 10000); does not affect results, larger is usually faster but uses more memory')
    parser.add_argument('--seed', type=int, default=0, help='the random seed (defaults to 0)')
    parser.add_argument('--maxiter', type=int, default=10, help='max number of iterations')

    recommended_group = parser.add_argument_group('recommended settings', 'Recommended settings for different scenarios')
    recommended_type = recommended_group.add_mutually_exclusive_group()
    recommended_type.add_argument('--supervised', metavar='DICTIONARY', help='recommended if you have a large training dictionary')
    recommended_type.add_argument('--semi_supervised', metavar='DICTIONARY', help='recommended if you have a small seed dictionary')
    recommended_type.add_argument('--identical', action='store_true', help='recommended if you have no seed dictionary but can rely on identical words')
    recommended_type.add_argument('--identical_custom', action='store_true', help='identical without self')
    recommended_type.add_argument('--fast', action='store_true', help='identical without self')
    recommended_type.add_argument('--unsupervised', action='store_true', help='recommended if you have no seed dictionary and do not want to rely on identical words')
    recommended_type.add_argument('--acl2018', action='store_true', help='reproduce our ACL 2018 system')
    recommended_type.add_argument('--aaai2018', metavar='DICTIONARY', help='reproduce our AAAI 2018 system')
    recommended_type.add_argument('--acl2017', action='store_true', help='reproduce our ACL 2017 system with numeral initialization')
    recommended_type.add_argument('--acl2017_seed', metavar='DICTIONARY', help='reproduce our ACL 2017 system with a seed dictionary')
    recommended_type.add_argument('--emnlp2016', metavar='DICTIONARY', help='reproduce our EMNLP 2016 system')

    init_group = parser.add_argument_group('advanced initialization arguments', 'Advanced initialization arguments')
    init_type = init_group.add_mutually_exclusive_group()
    init_type.add_argument('-d', '--init_dictionary', default=sys.stdin.fileno(), metavar='DICTIONARY', help='the training dictionary file (defaults to stdin)')
    init_type.add_argument('--init_identical', action='store_true', help='use identical words as the seed dictionary')
    init_type.add_argument('--init_numerals', action='store_true', help='use latin numerals (i.e. words matching [0-9]+) as the seed dictionary')
    init_type.add_argument('--init_unsupervised', action='store_true', help='use unsupervised initialization')
    init_group.add_argument('--unsupervised_vocab', type=int, default=0, help='restrict the vocabulary to the top k entries for unsupervised initialization')

    mapping_group = parser.add_argument_group('advanced mapping arguments', 'Advanced embedding mapping arguments')
    mapping_group.add_argument('--normalize', choices=['unit', 'center', 'unitdim', 'centeremb', 'none'], nargs='*', default=[], help='the normalization actions to perform in order')
    mapping_group.add_argument('--whiten', action='store_true', help='whiten the embeddings')
    mapping_group.add_argument('--src_reweight', type=float, default=0, nargs='?', const=1, help='re-weight the source language embeddings')
    mapping_group.add_argument('--trg_reweight', type=float, default=0, nargs='?', const=1, help='re-weight the target language embeddings')
    mapping_group.add_argument('--src_dewhiten', choices=['src', 'trg'], help='de-whiten the source language embeddings')
    mapping_group.add_argument('--trg_dewhiten', choices=['src', 'trg'], help='de-whiten the target language embeddings')
    mapping_group.add_argument('--dim_reduction', type=int, default=0, help='apply dimensionality reduction')
    mapping_type = mapping_group.add_mutually_exclusive_group()
    mapping_type.add_argument('-c', '--orthogonal', action='store_true', help='use orthogonal constrained mapping')
    mapping_type.add_argument('-u', '--unconstrained', action='store_true', help='use unconstrained mapping')

    self_learning_group = parser.add_argument_group('advanced self-learning arguments', 'Advanced arguments for self-learning')
    self_learning_group.add_argument('--self_learning', action='store_true', help='enable self-learning')
    self_learning_group.add_argument('--vocabulary_cutoff', type=int, default=0, help='restrict the vocabulary to the top k entries')
    self_learning_group.add_argument('--direction', choices=['forward', 'backward', 'union'], default='union', help='the direction for dictionary induction (defaults to union)')
    self_learning_group.add_argument('--csls', type=int, nargs='?', default=0, const=10, metavar='NEIGHBORHOOD_SIZE', dest='csls_neighborhood', help='use CSLS for dictionary induction')
    self_learning_group.add_argument('--threshold', default=0.000001, type=float, help='the convergence threshold (defaults to 0.000001)')
    self_learning_group.add_argument('--validation', default=None, metavar='DICTIONARY', help='a dictionary file for validation at each iteration')
    self_learning_group.add_argument('--stochastic_initial', default=0.1, type=float, help='initial keep probability stochastic dictionary induction (defaults to 0.1)')
    self_learning_group.add_argument('--stochastic_multiplier', default=2.0, type=float, help='stochastic dictionary induction multiplier (defaults to 2.0)')
    self_learning_group.add_argument('--stochastic_interval', default=50, type=int, help='stochastic dictionary induction interval (defaults to 50)')
    self_learning_group.add_argument('--log', help='write to a log file in tsv format at each iteration')
    self_learning_group.add_argument('-v', '--verbose', action='store_true', help='write log information to stderr at each iteration')
    args = parser.parse_args()

    if args.fast:
        parser.set_defaults(init_identical=True, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', self_learning=True, vocabulary_cutoff=2000, csls_neighborhood=0)
    if args.supervised is not None:
        parser.set_defaults(init_dictionary=args.supervised, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', batch_size=1000)
    if args.semi_supervised is not None:
        parser.set_defaults(init_dictionary=args.semi_supervised, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', self_learning=True, vocabulary_cutoff=20000, csls_neighborhood=10)
    if args.identical:
        parser.set_defaults(init_identical=True, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', self_learning=True, vocabulary_cutoff=20000, csls_neighborhood=10)
    if args.unsupervised or args.acl2018:
        parser.set_defaults(init_unsupervised=True, unsupervised_vocab=4000, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', self_learning=True, vocabulary_cutoff=20000, csls_neighborhood=10)
    if args.aaai2018:
        parser.set_defaults(init_dictionary=args.aaai2018, normalize=['unit', 'center'], whiten=True, trg_reweight=1, src_dewhiten='src', trg_dewhiten='trg', batch_size=1000)
    if args.acl2017:
        parser.set_defaults(init_numerals=True, orthogonal=True, normalize=['unit', 'center'], self_learning=True, direction='forward', stochastic_initial=1.0, stochastic_interval=1, batch_size=1000)
    if args.acl2017_seed:
        parser.set_defaults(init_dictionary=args.acl2017_seed, orthogonal=True, normalize=['unit', 'center'], self_learning=True, direction='forward', stochastic_initial=1.0, stochastic_interval=1, batch_size=1000)
    if args.emnlp2016:
        parser.set_defaults(init_dictionary=args.emnlp2016, orthogonal=True, normalize=['unit', 'center'], batch_size=1000)
    args = parser.parse_args()

    # Check command line arguments
    if (args.src_dewhiten is not None or args.trg_dewhiten is not None) and not args.whiten:
        print('ERROR: De-whitening requires whitening first', file=sys.stderr)
        sys.exit(-1)

    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'

    # Read input embeddings
    with open(args.src_input, encoding=args.encoding, errors='surrogateescape') as srcfile, \
            open(args.trg_input, encoding=args.encoding, errors='surrogateescape') as trgfile:
        src_words, x = embeddings.read(srcfile, dtype=dtype, threshold=args.vocabulary_cutoff)
        trg_words, z = embeddings.read(trgfile, dtype=dtype, threshold=args.vocabulary_cutoff)

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

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    # STEP 0: Normalization
    embeddings.normalize(x, args.normalize)
    embeddings.normalize(z, args.normalize)

    # Build the seed dictionary
    src_indices = []
    trg_indices = []
    if args.init_numerals:
        numeral_regex = re.compile('^[0-9]+$')
        src_numerals = {word for word in src_words if numeral_regex.match(word) is not None}
        trg_numerals = {word for word in trg_words if numeral_regex.match(word) is not None}
        numerals = src_numerals.intersection(trg_numerals)
        for word in numerals:
            src_indices.append(src_word2ind[word])
            trg_indices.append(trg_word2ind[word])
    elif args.init_identical:
        identical = set(src_words).intersection(set(trg_words))
        for word in identical:
            src_indices.append(src_word2ind[word])
            trg_indices.append(trg_word2ind[word])
        print(f'count identical {len(identical)}')
    else:
        f = open(args.init_dictionary, encoding=args.encoding, errors='surrogateescape')
        for line in f:
            try:
                src, trg = line.split()[:2]
            except ValueError:
                continue
            try:
                src_ind = src_word2ind[src]
                trg_ind = trg_word2ind[trg]
                src_indices.append(src_ind)
                trg_indices.append(trg_ind)
            except KeyError:
                print('WARNING: OOV dictionary entry ({0} - {1})'.format(src, trg), file=sys.stderr)

    # Read validation dictionary
    if args.validation is not None:
        f = open(args.validation, encoding=args.encoding, errors='surrogateescape')
        validation = collections.defaultdict(set)
        oov = set()
        vocab = set()
        for line in f:
            try:
                src, trg = line.split()
            except ValueError:
                continue
            try:
                src_ind = src_word2ind[src]
                trg_ind = trg_word2ind[trg]
                validation[src_ind].add(trg_ind)
                vocab.add(src)
            except KeyError:
                oov.add(src)
        oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
        validation_coverage = len(validation) / (len(validation) + len(oov))

    # Create log file
    if args.log:
        log = open(args.log, mode='w', encoding=args.encoding, errors='surrogateescape')

    # Allocate memory
    xw = xp.empty_like(x)
    zw = xp.empty_like(z)

    matches = {}
    for p in zip(src_indices, trg_indices):
        matches[p] = 1
    identical = set(src_words).intersection(set(trg_words))
    for word in identical:
        p = (src_word2ind[word], trg_word2ind[word])
        matches[p] = 1

    if args.validation is not None:
        simval = xp.empty((len(validation.keys()), z.shape[0]), dtype=dtype)

    # Training loop
    it = 1
    keep_prob = args.stochastic_initial
    t = time.time()
    wprev = 0
    decided = collections.Counter()
    excluded_src = set()
    while True:
        def flatten_match(matches):
            indices, weights = [list(a) for a in zip(*matches.items())]
            weights = xp.array(weights, dtype=dtype)[:, None]
            src_indices, trg_indices = [list(a) for a in zip(*indices)]
            return src_indices, trg_indices, weights

        # samp_m = sample_matches(matches, p=1)
        src_indices, trg_indices, weights = flatten_match(matches)
        if args.unconstrained:
            w = np.linalg.lstsq(np.sqrt(weights) * x[src_indices], np.sqrt(weights) * z[trg_indices], rcond=None)[0]
            # w = np.linalg.lstsq(x[src_indices], z[trg_indices], rcond=None)[0]
            x.dot(w, out=xw)
            zw = z[:]
        else:
            u, s, vt = xp.linalg.svd((weights * z[trg_indices]).T.dot(x[src_indices]))
            # u, s, vt = xp.linalg.svd(z[trg_indices].T.dot(x[src_indices]))
            w = vt.T.dot(u.T)
            x.dot(w, out=xw)
            zw = z[:]

        if it % 10 == 0:
            print('updating decided ....')
            matchesp = collections.Counter(matches)
            tops = int(0.25 * len(matchesp))
            for k, v in matchesp.most_common()[:tops]:
                s, t = k
                decided[(s, t)] += v

        T = 1 * np.exp((it - 1) * np.log(1e-2) / (args.maxiter))
        # T = 1
        matches, objective = find_matches(matches, xw, zw, decided, T=T)
        
        # Accuracy and similarity evaluation in validation
        if args.validation is not None:
            src = list(validation.keys())
            xw[src].dot(zw.T, out=simval)
            nn = asnumpy(simval.argmax(axis=1))
            accuracy = np.mean([1 if nn[i] in validation[src[i]] else 0 for i in range(len(src))])
            similarity = np.mean([np.max([simval[i, j].tolist() for j in validation[src[i]]]) for i in range(len(src))])

        # Logging
        duration = time.time() - t
        if args.verbose:
            print(file=sys.stderr)
            print('ITERATION {0} ({1:.2f}s)'.format(it, duration), file=sys.stderr)
            print('\t- Objective:        {0:9.4f}%'.format(100 * objective), file=sys.stderr)
            print(f'\t- Temp:             {T:.3f}', file=sys.stderr)
            print(f'\t- #match/#decided:             {len(matches)}/{len(decided)}', file=sys.stderr)
            print(f'\t- DeltaW:             {np.linalg.norm(w - wprev):.3f}', file=sys.stderr)
            if args.validation is not None:
                print('\t- Val. similarity:  {0:9.4f}%'.format(100 * similarity), file=sys.stderr)
                print('\t- Val. accuracy:    {0:9.4f}%'.format(100 * accuracy), file=sys.stderr)
                print('\t- Val. coverage:    {0:9.4f}%'.format(100 * validation_coverage), file=sys.stderr)
            sys.stderr.flush()
        if args.log is not None:
            val = '{0:.6f}\t{1:.6f}\t{2:.6f}'.format(
                100 * similarity, 100 * accuracy, 100 * validation_coverage) if args.validation is not None else ''
            print('{0}\t{1:.6f}\t{2}\t{3:.6f}'.format(it, 100 * objective, val, duration), file=log)
            log.flush()

        if it >= args.maxiter:
            break

        t = time.time()
        wprev = w
        matchesprev = matches
        it += 1

    with open('dict.tmp', mode='w') as f:
        for p in decided:
            si = p[0]
            ti = p[1]
            print(f'{src_words[si]}\t{trg_words[ti]}\t{decided[p]:.3f}', file=f)

    # write mapped embeddings
    print('**** reading and writing final embeddings ****', file=sys.stderr)
    with open(args.src_input, encoding=args.encoding, errors='surrogateescape') as srcfile, \
            open(args.trg_input, encoding=args.encoding, errors='surrogateescape') as trgfile:
        src_words, x = embeddings.read(srcfile, dtype=dtype, threshold=100000)
        trg_words, z = embeddings.read(trgfile, dtype=dtype, threshold=100000)

    embeddings.normalize(x, args.normalize)
    embeddings.normalize(z, args.normalize)

    with open(args.src_output, mode='w', encoding=args.encoding, errors='surrogateescape') as srcfile, \
            open(args.trg_output, mode='w', encoding=args.encoding, errors='surrogateescape') as trgfile:
        embeddings.write(src_words, x.dot(w), srcfile)
        embeddings.write(trg_words, z, trgfile)

if __name__ == '__main__':
    main()
