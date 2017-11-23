#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm

from collections import Counter
from operator import itemgetter

import numpy as np
from sklearn.utils import shuffle

from model.utils import iobes_to_spans

from utils.wikibot import Wikibot


''' Example
<document id="DrugDDI.d89" origId="Aciclovir">
    <sentence id="DrugDDI.d89.s0" origId="s0" text="Co-administration of probenecid with acyclovir has been shown to increase the mean half-life and the area under the concentration-time curve.">
        <entity id="DrugDDI.d89.s0.e0" origId="s0.p1" charOffset="21-31" type="drug" text="probenecid"/>
        <entity id="DrugDDI.d89.s0.e1" origId="s0.p2" charOffset="37-47" type="drug" text="acyclovir"/>
        <pair id="DrugDDI.d89.s0.p0" e1="DrugDDI.d89.s0.e0" e2="DrugDDI.d89.s0.e1" interaction="true"/>
    </sentence><sentence id="DrugDDI.d89.s1" origId="s1" text="Urinary excretion and renal clearance were correspondingly reduced."/>
    <sentence id="DrugDDI.d89.s2" origId="s2" text="The clinical effects of this combination have not been studied."/>
</document>
'''

RANDOM_STATE = 5
## TODO: tokenization and remove punctuations which are not part of entity mentions
class Word:
    def __init__(self, index, text, etype):
        self.index = index
        self.text = text
        self.etype = etype
    
    def __repr__(self):
        return '[index: {}, text: {}, etype: {}]'.format(self.index, self.text, self.etype)
    
    __str__ = __repr__
        

def parse_charoffset(charoffset):
    """
    Parse charoffset to a tuple containing start and end indices.
    Example:
        charoffset = '3-7;8-9'
        
        [[3, 7], [8, 9]]
    """
    # try split by ';'
    charoffsets = charoffset.split(';')
    return [[int(x.strip()) for x in offset.split('-')] for offset in charoffsets]
    
def parse_sentence(sent):
    """
    Parse sentence to get a list of Word class
    Example:
        sent = 'it is it'
        print(parse_sentence(sent))
        
        [(0, 'it', 'O'), (3, 'is', 'O'), (6, 'it', 'O')]
    """
    sent = sent.strip()
    res = []
    if len(sent) == 0:
        return res
    i = j = 0
    while j <= len(sent):
        if j < len(sent) and sent[j] != ' ':
            j += 1
        else:
            if j > i: # in case where two spaces are adjacent
                res.append(Word(i, sent[i:j], 'O'))
            i = j + 1
            j = i
    return res

def tag_word(words, entity):
    """
    Args:
        entity: dict has keys charOffset, type, text
    Tag Word with entity type in-place
    Example:
        words = [(0, 'it(', O), (4, 'is', O), (7, 'it', O)]
        entity = {'charOffset': [0, 2], 'type': 'eng'} # [inclusive, exclusive]
        print(tag_word(words, entity))
        
        [(0, 'it', 'B-ENG'), (3, (, 'O'), (4, 'is', 'O'), (7, 'it', 'O')]
    """
    beg, end = entity['charOffset']
    count = 0
    origword = None
    orig_i = None
    for i, word in enumerate(words):
        if word.index >= beg and word.index < end: # coarse tagging
            if word.index + len(word.text) - 1 >= end:
                origword = word
                orig_i = i
            count += 1
            if count > 1:
                word.etype = 'I-' + entity['type'].upper()
            else:
                word.etype = 'B-' + entity['type'].upper()
    # fine tagging
    # if end index of word is larger than end index of charOffset, such as the case of example
    # split the word into two words, and tag the latter O
    if origword is None:
        return
    origtext = origword.text 
    origword.text = origtext[:end - origword.index] # update text
    nextindex = origword.index + len(origword.text)
    nextword = Word(nextindex, origtext[len(origword.text)], 'O')
    words.insert(orig_i + 1, nextword)
    
            
def generate_annotated_sentences(root):
    """
    Args:
        root: root Element of XML
    """
    for sent in root.findall('sentence'):
        words = parse_sentence(sent.get('text'))
        for entity in sent.findall('entity'):
            attributes = entity.attrib
            charoffset = attributes['charOffset']
            parsed_charoffsets = parse_charoffset(charoffset)
            for parsed_charoffset in parsed_charoffsets:
                attributes['charOffset'] = parsed_charoffset
                tag_word(words, attributes)
        yield words

def preprocess_ddi(data_path='../data/DrugDDI/DrugDDI_Unified/'):
    """
    Preprocess ddi data and write results to files
    Return:
        res: list of tuple (docname, list of parsed sentences)
    """
    res = []
    file_pattern = os.path.join(data_path, '*.xml')
    for f in glob.glob(file_pattern):
        res_perdoc = []
        print('Processing: {}...'.format(f))
        # import xml data into ElementTree
        tree = ET.parse(f)
        root = tree.getroot()
        for words in generate_annotated_sentences(root):
            res_perdoc.append(words)
        res.append((f, res_perdoc))
    print('Done')
    return res

def save_data(pre_data, output_path='../data/DrugDDI/ddi.txt'):
    with open(output_path, 'w') as output_file:
        output_file.write('-DOCSTART- -X- -X- O\n\n')
        for doc_tuple in pre_data:
            doc, sents = doc_tuple
            output_file.write('-DOCSTART- ' + doc + '\n')
            for words in sents:
                for word in words:
                    output_file.write('{} {}\n'.format(word.text, word.etype))
                output_file.write('\n') 

def labelseq2conll(labelseq, iob=False):
    """
    Change format of '<DRUG> sulindac </DRUG> ; ' to that of CoNll 2003
    also allow iobes tagging <B-DRUG>
    """
    # the first line starts with -DOCSTART-
    start_indicator = '-DOCSTART-'
    sp = re.compile(r'<([BI]?)(-?)(\w*)>')
    ep = re.compile(r'</([BI]?)(-?)(\w*)>')
    res = []
    sent = []
    labels = []
    for l in labelseq:
        l = l.strip()
        etype = None
        if len(l) > 0 and start_indicator not in l:
            terms = l.split()
            for term in terms:
                smatch = sp.match(term)
                ematch = ep.match(term)
                if smatch:
                    iob_prefix = smatch.group(1)
                    etype = smatch.group(3)
                    continue
                if ematch:
                    etype = None
                    continue
                sent.append(term)
                if etype:
                    if iob and iob_prefix:
                        labels.append(iob_prefix + '-' + etype)
                    else:
                        labels.append(etype)
                else:
                    labels.append('O')
        elif len(sent) > 0:
            res.append((sent, labels))
            sent = []
            labels = []
    if len(sent) > 0:
        res.append((sent, labels))
    return res
    
def iob2etype(lines, iob=False):
    """
    Change iob format to etype: for example, I-DRUG to DRUG
    """
    start_indicator = '-DOCSTART-'
    p = re.compile(r'([IB])-(\w+)')
    res = []
    sent = []
    labels = []
    for l in lines:
        l = l.strip()
        if len(l) > 0 and start_indicator not in l:
            w, label = l.split()
            sent.append(w)
            match = p.match(label)
            if match:
                if iob:
                    label = match.group(0)
                else:
                    label = match.group(2)
            labels.append(label)
        elif len(sent) > 0:
            res.append((sent, labels))
            sent = []
            labels = []
    if len(sent) > 0:
        res.append((sent, labels))
    return res
            
def find_error(gold, pred, fout, bot = None, pem=None):
    """
    Find error given gold reference and output to fout
    """
    bot = bot if bot is not None else Wikibot()
    fmt = '{:<20},{:<10},{:<10},{:<8},{:<20},{:<8}'
    with open(fout, 'w') as f:
        head = 'sentence,gold,pred'
        head += ',in_pem,top_entity,if_drugbank'
        head += '\n\n'
        f.write(head)
        for ref, p in tqdm(zip(gold, pred)):
            labels_ref = ref[1]
            labels_pred = p[1]
            terms = ref[0]
            if labels_ref != labels_pred: # find error!
                for i in range(len(terms)):
                    in_pem = 'None'
                    top_entity = 'None'
                    if_drugbank = 'None'
                    if pem:
                        in_pem = True if terms[i] in pem else False
                        if in_pem:
                            top_entity, _ = pem[terms[i]][0]
                            title = top_entity.replace('_', ' ')
                            if_drugbank = bot.search_in_page(['DrugBank_Ref', 'ATC_prefix'], title)
                    f.write(fmt.format(terms[i], labels_ref[i], labels_pred[i], in_pem, top_entity, if_drugbank))
                    if labels_ref[i] != labels_pred[i]:
                        f.write('-----------------------')
                    f.write('\n')
                f.write('\n')
                        
def train_val_test_split(data, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_state=None):
    """
    Shuffle a list of objects and split into train, val, and test dataset
    """
    indices = shuffle(np.arange(0, len(data)), random_state=RANDOM_STATE)
    
    
    # assert if ratios are compatible
    assert train_ratio + val_ratio + test_ratio == 1, 'train, val, and test ratio must sum to 1'
    
    
    ntrain = int(train_ratio*len(res))
    nval = int(val_ratio*len(res))
    ntest = len(res) - ntrain - nval
    train_indices = indices[0:ntrain]
    val_indices = indices[ntrain:ntrain+nval]
    test_indices = indices[-ntest:]
    
    def indexlistbylist(l, indices):
        return [l[indice] for indice in indices]
    train_data = indexlistbylist(data, train_indices)
    val_data = indexlistbylist(data, val_indices)
    test_data = indexlistbylist(data, test_indices)
    
    return train_data, val_data, test_data

def generate_ngrams(input_list, nums=2):
    """
    Generate n-grams from input_list
    Args:
        input_list: list of strings
        nums: if single int, only generate n-gram, by default 2-gram; if a list of int, generate grams specified by the list
    
    Return:
        list of list of ngrams as tuple
        
    Example:
        >>> a = ngrams('this is an exmaple'.split(), range(4))
        >>> a
        [[('this',), ('is',), ('an',), ('exmaple',)],
         [('this', 'is'), ('is', 'an'), ('an', 'exmaple')],
         [('this', 'is', 'an'), ('is', 'an', 'exmaple')]]
        
    """
    if type(nums) == int:
        nums = [nums]
    res = []
    for n in nums:
        if n > 0:
            res.append(list(zip(*(input_list[i:] for i in range(n)))))
    return res

def parse_pem(data_path='../data/crosswikis_wikipedia_p_e_m.txt', verbose=False):
    """
    Example:
        broncho	2	308453,1,Bronchiole	308449,1,Bronchus	
        Georgiana Molloy	9	2123300,9,Georgiana_Molloy
    Note:
        entities are sorted in descending order of frequency
    """
    pem = {} # p(e|m)
    e_map = {} # mapping entity to int id
    with open(data_path, 'r') as f:
        for i, line in enumerate(f, 1):
            if i % 500000 == 0:
                print('parsing {} lines'.format(i))
            mention, total_freq, *entities = line.strip().split('\t')
            unpacked_entities = (entity.split(',') for entity in entities)
            unpacked_entities = ((int(entity[0]), int(entity[1]), entity[2]) for entity in unpacked_entities)
            _counter = Counter()
            for idx, freq, name in unpacked_entities:
                # sanity check: if entity id is unique
                if verbose and name in e_map and e_map[name] != idx:
                     print('when dealing with {}, entity {} has non-unique idx: {} vs {}'.format(mention, name, e_map[name], idx))
                else:
                    e_map[name] = idx
                if name not in _counter:
                    _counter[name] = freq
                else:
                    _counter[name] += freq
            if mention not in pem:
                pem[mention] = _counter
            else:
                pem[mention] += _counter
    # cast Counter first to dict, then cast to list, then sort in descending order of frequency
    pem = {k:sorted(list(dict(v).items()), key=itemgetter(1), reverse=True) for k, v in pem.items()}
    return pem

def evaluate_baseline(ref_tgs, pred_tgs, l_map):
    """
    Args:
        ref_tgs, pred_tgs: [[str]]
    """
    r_l_map = {k:v for v, k in l_map.items()}
    correct_labels = 0
    total_labels = 0
    gold_count = 0
    guess_count = 0
    overlap_count = 0
    
    def f1_score():
        """
        calculate f1 score based on statics
        """
        print(correct_labels, total_labels, gold_count, guess_count, overlap_count)
        if guess_count == 0:
            return 0.0, 0.0, 0.0, 0.0
        precision = overlap_count / float(guess_count)
        recall = overlap_count / float(gold_count)
        if precision == 0.0 or recall == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        f = 2 * (precision * recall) / (precision + recall)
        accuracy = float(correct_labels) / total_labels
        return f, precision, recall, accuracy
    
    def eval_instance(pred, ref):
        """
        update statics for one instance
        """
        total_labels = len(pred)
        correct_labels = np.sum(np.equal(pred, ref))
        gold_chunks = iobes_to_spans(ref, r_l_map)
        gold_count = len(gold_chunks)
    
        guess_chunks = iobes_to_spans(pred, r_l_map)
        guess_count = len(guess_chunks)
    
        overlap_chunks = gold_chunks & guess_chunks
        overlap_count = len(overlap_chunks)
    
        return correct_labels, total_labels, gold_count, guess_count, overlap_count            
    
    for ref_tg, pred_tg in tqdm(zip(ref_tgs, pred_tgs)):
        ref_tg_id = np.array([l_map[tg] for tg in ref_tg])
        pred_tg_id = np.array([l_map[tg] for tg in pred_tg])
        correct_labels_i, total_labels_i, gold_count_i, guess_count_i, overlap_count_i = eval_instance(pred_tg_id, ref_tg_id)
        correct_labels += correct_labels_i
        total_labels += total_labels_i
        gold_count += gold_count_i
        guess_count += guess_count_i
        overlap_count += overlap_count_i

    return f1_score()
    
if __name__ == '__main__':
    dataset = 'ddi_tiny'
    res = preprocess_ddi(data_path='../data/'+ dataset + '/xml')
#    save_data(res, output_path='../data/' + dataset + '/train.ddi')
    
#    data_path = '../data/DrugDDI/ddi.txt'
    
    
    # shuffle and split data
    train_data, val_data, test_data = train_val_test_split(res, random_state=RANDOM_STATE)
    save_data(train_data, output_path='../data/' + dataset + '/train.ddi')
    save_data(val_data, output_path='../data/' + dataset + '/val.ddi')
    save_data(test_data, output_path='../data/' + dataset + '/test.ddi')
    
    
    
            
    
    
            
        