#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import copy
from itertools import chain

import model.data_utils as data_utils

import pywikibot
from tqdm import tqdm

from utils.wikibot import Wikibot

def postprocessing(pred, pem, bot, keep=True):
    """
    for all mentions m found by your best NER system, check if they appear in the p(e|m) dictionary. 
    If they appear , check if the highest scored entity contains a DrugBank ID. If it does not contain, 
    then it is not a drug, so remove this mention from the annotations of your system. On the other side, 
    if the mention does not appear in the dictionary, you have two options: either to keep it 
    (maybe it is an uncommon drug acronym and you do not want to loose it), or remove it 
    
    Args:
        pred: [([str], [str])]
    Return:
        pred_copy = [([str], [str])]
    """
    
    pred_copy = copy.deepcopy(pred)
    for features, tgs in tqdm(pred_copy):
        for i in range(len(features)):
            if 'DRUG' in tgs[i]:
                if features[i] in pem:
                    top_entity, _ = pem[features[i]][0]
                    title = top_entity.replace('_', ' ')
                    bot.search_in_page('DrugBank_Ref', title)
                    if not bot.cache[title]:
                        tgs[i] = 'O'
                elif not keep:
                    tgs[i] = 'O'
    
    return pred_copy


data_path = '../data/drugddi2011'
pem_path = '../data/crosswikis_wikipedia_p_e_m.txt'
    
    
print('Preparing P(e|m)')
#pem = data_utils.parse_pem(pem_path)

# create wikibot
bot = Wikibot()

filename = os.path.join(data_path, 'pred_iobes.ddi')
with open(filename, 'r') as f:
    pred = data_utils.labelseq2conll(f.readlines(), iob=True)

filename = os.path.join(data_path, 'test.ddi')
with open(filename, 'r') as f:
    gold = data_utils.iob2etype(f.readlines(), iob=True)

data_utils.find_error(gold, pred, os.path.join(data_path, 'error.txt'), pem=pem)


# postprocessing
pred_pp_keep = postprocessing(pred, pem, bot, keep=True)
pred_pp_nonkeep = postprocessing(pred, pem, bot, keep=False)
data_utils.find_error(gold, pred_pp_keep, os.path.join(data_path, 'error_pp_keep.txt'), pem=pem)
data_utils.find_error(gold, pred_pp_nonkeep, os.path.join(data_path, 'error_pp_nonkeep.txt'), pem=pem)


# evaluation
_, pred_pp_keep_tgs = list(zip(*pred_pp_keep))
_, pred_pp_nonkeep_tgs = list(zip(*pred_pp_nonkeep))
_, gold_tgs = list(zip(*gold))
_, pred_tgs = list(zip(*pred))

tg_set = set(chain.from_iterable(gold_tgs))
l_map = {k:v for v, k in enumerate(tg_set)}

fmt = ' test_f1: {:.4f} test_pre: {:.4f} test_rec: {:.4f} test_acc: {:.4f}\n'
print(fmt.format(*data_utils.evaluate_baseline(gold_tgs, pred_tgs, l_map)))
print(fmt.format(*data_utils.evaluate_baseline(gold_tgs, pred_pp_keep_tgs, l_map)))
print(fmt.format(*data_utils.evaluate_baseline(gold_tgs, pred_pp_nonkeep_tgs, l_map)))




