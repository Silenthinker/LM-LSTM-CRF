#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jyao
Go over all n-grams up to length k (say 5) in a text, check which of these n-grams appear in the p(e|m) dictionary. 
For each mention that appears, take the top scored entity e based on p(e|m) value, 
and check its wikipedia page to see if it contains a DrugBank id. 
If it does, link it, otherwise do not link it
"""

from .data_utils import generate_ngrams

from utils.wikibot import Wikibot

class Baseline:
    def __init__(self, pem, nums, bot=None):
        """
        Args:
            pem: P(e|m)
            nums: consider ngrams with n given in nums
        """
        self.pem = pem
        self.nums = nums
        self.bot = bot if bot is not None else Wikibot()
        
    def decode(self, features):
        """
        Return labels of sequences given features
        Return:
            labels: [[str]]
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
                        if self.bot.search_in_page(['DrugBank_Ref', 'ATC_prefix'], title):
                            label[i:i+len(ngram)] = ['B-DRUG'] + ['I-DRUG']*(len(ngram) - 1)
            labels.append(label)
        return labels