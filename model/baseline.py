#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jyao
Go over all n-grams up to length k (say 5) in a text, check which of these n-grams appear in the p(e|m) dictionary. 
For each mention that appears, take the top scored entity e based on p(e|m) value, 
and check its wikipedia page to see if it contains a DrugBank id. 
If it does, link it, otherwise do not link it
"""

from collections import Counter
from operator import itemgetter

from .data_utils import generate_ngrams

import pywikibot

class Baseline:
    def __init__(self, pem, nums):
        """
        Args:
            pem: P(e|m)
            nums: consider ngrams with n given in nums
        """
        self.pem = pem
        self.nums = nums
        self.cache = {} # caching if entity is drug
        # configure
        self.site = pywikibot.Site('en', 'wikipedia')
        
    def decode(self, features):
        """
        Return labels of sequences given features
        """
        labels = []
        for feature in features:
            label = ['O' for i in range(len(feature))]
            all_ngrams = generate_ngrams(feature, nums=self.nums)
            for single_ngrams in all_ngrams:
                for i, ngram in enumerate(single_ngrams):
                    ngram_concat = '_'.join(ngram)
                    if ngram_concat in self.pem:
                        top_entity, _ = self.pem[ngram_concat][0]
                        title = top_entity.replace('_', ' ')
                        if top_entity not in self.cache:
                            page = pywikibot.Page(self.site, title)
                            if 'DrugBank' in page.text:
                                self.cache[top_entity] = True
                            else:
                                self.cache[top_entity] = False
                        if self.cache[top_entity]:
                            label[i:i+len(ngram)] = ['B-DRUG'] + ['I-DRUG']*(len(ngram) - 1)
            labels.append(label)
        return labels