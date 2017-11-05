#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jyao
"""
from __future__ import print_function
import codecs
import model.utils as utils
import argparse
import itertools

from model.baseline import Baseline
from model.data_utils import parse_pem, evaluate_baseline



parser = argparse.ArgumentParser(description='Evaluating Baseline')
parser.add_argument('--dev_file', default='../data/drugddi2011/val.ddi', help='path to development file')
parser.add_argument('--test_file', default='../data/drugddi2011/test.ddi', help='path to test file')
parser.add_argument('--pem_file', default='../data/pem.pkl', help='path to pem file')
parser.add_argument('--eva_matrix', choices=['a', 'fa'], default='fa', help='use f1 and accuracy or accuracy alone')
args = parser.parse_args()

data_path = '../data/crosswikis_wikipedia_p_e_m.txt'

# load corpus
with codecs.open(args.dev_file, 'r', 'utf-8') as f:
    dev_lines = f.readlines()

with codecs.open(args.test_file, 'r', 'utf-8') as f:
    test_lines = f.readlines()
    

# converting format
dev_features, dev_labels = utils.read_corpus(dev_lines)

# map label to int id
all_labels = list(set(itertools.chain.from_iterable(dev_labels)))
l_map = {k:v for v, k in enumerate(all_labels)}

test_features, test_labels = utils.read_corpus(test_lines)

#text = "In a placebo-controlled trial in normal volunteers, the administration of a single 1 mg dose of doxazosin doxazosin on day 1 of a four-day regimen of oral cimetidine (400 mg twice daily) resulted in a 10% increase in mean AUC of doxazosin (p=0.006), and a slight but not statistically significant increase in mean Cmax and mean half-life of doxazosin."
print('Preparing P(e|m)')
pem = parse_pem(data_path)
nums = range(6)
model = Baseline(pem, nums)

labels = model.decode(test_features)

test_f1, test_pre, test_rec, test_acc = evaluate_baseline(test_labels, labels, l_map)
print(' test_f1: %.4f test_rec: %.4f test_pre: %.4f test_acc: %.4f\n' % (test_f1, test_rec, test_pre, test_acc))

